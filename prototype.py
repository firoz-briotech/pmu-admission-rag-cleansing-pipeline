"""Scrape-data cleansing pipeline with a SQLite-backed audit log.

The script is split into small sub-logics so every step can be audited on its
own:

    1. imports + constants
    2. configuration (env vars -> Settings)
    3. audit table + custom logging handler
    4. Gemini client
    5. single-file processing
    6. pipeline orchestration
    7. CLI entry point

Each sub-logic wraps its work in try/except and writes one audit row per
outcome: `info` when it succeeds, `error` when it fails (plus `warning` and
`debug` where useful).
"""

import argparse
import logging
import os
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- 1. CONSTANTS ---
LOGGER_NAME = "webscrap.audit"
AUDIT_TABLE = "audit_log"
DEFAULT_AUDIT_DB = "audit_logs.db"

# log_type stored in the audit table, derived from the logging level.
LOG_TYPE_BY_LEVEL = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "critical",
}

# Rows that report progress need no follow-up; issues stay unresolved.
AUTO_RESOLVED_TYPES = {"debug", "info"}

AUDIT_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {AUDIT_TABLE} (
    audit_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    log_type    TEXT    NOT NULL,
    log_message TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    resolved    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_{AUDIT_TABLE}_session
    ON {AUDIT_TABLE} (session_id);
"""

# --- 2. CONFIGURATION ---

# This is where you define what "cleaning" means for your scrape data
SYSTEM_PROMPT = """
You are a data cleaning assistant. The user will give you raw scraped text from a website crawler.
Your job is to extract ONLY the main content — the actual information in the center of the page.
Remove: navigation menus, headers, footers, ads, breadcrumbs, repeated links, boilerplate text.
Keep: article body, descriptions, main text, factual content.
Return ONLY the cleaned text, nothing else. No explanations, no preamble.
"""


@dataclass(frozen=True)
class Settings:
    """Everything the pipeline needs, read once at start-up."""

    session_id: str
    project_id: str
    location: str
    model: str
    raw_dir: Path
    cleaned_dir: Path
    audit_db: Path
    log_level: int
    console_logging: bool


def parse_log_level(raw: str | None) -> int:
    """Translate LOG_LEVEL=DEBUG|INFO|... into a logging level, defaulting to INFO."""
    if not raw:
        return logging.INFO
    level = logging.getLevelName(raw.strip().upper())
    return level if isinstance(level, int) else logging.INFO


def load_settings(session_id: str) -> Settings:
    """Sub-logic: read configuration from the environment / .env file."""
    return Settings(
        session_id=session_id,
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", ""),
        model=os.getenv("GEMINI_MODEL", ""),
        raw_dir=Path("scrap_data"),
        cleaned_dir=Path("cleansed_data"),
        audit_db=Path(os.getenv("AUDIT_DB_PATH", DEFAULT_AUDIT_DB)),
        log_level=parse_log_level(os.getenv("LOG_LEVEL")),
        console_logging=os.getenv("AUDIT_CONSOLE", "1") != "0",
    )


def validate_settings(settings: Settings, audit: "AuditLog") -> bool:
    """Sub-logic: fail fast with one clear error row if config is incomplete."""
    missing = [
        name
        for name, value in (
            ("GOOGLE_CLOUD_PROJECT", settings.project_id),
            ("GOOGLE_CLOUD_LOCATION", settings.location),
            ("GEMINI_MODEL", settings.model),
        )
        if not value
    ]
    if missing:
        audit.error(f"Configuration invalid: missing environment variables {', '.join(missing)} (check .env)")
        return False
    audit.info(
        f"Configuration loaded (project={settings.project_id}, location={settings.location}, model={settings.model})"
    )
    return True


def prepare_directories(settings: Settings) -> None:
    """Sub-logic: make sure input exists and output can be written."""
    if not settings.raw_dir.is_dir():
        raise FileNotFoundError(f"raw directory '{settings.raw_dir}' does not exist")
    settings.cleaned_dir.mkdir(parents=True, exist_ok=True)


# --- 3. AUDIT LOGGING ---


class SqliteAuditStore:
    """Persistence for the audit table. Swap this class for BigQuery/Postgres later."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        if db_path.parent != Path(""):
            db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.executescript(AUDIT_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self):
        """Open a connection, commit on success and always close it (no leaks)."""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def insert(
        self,
        session_id: str,
        log_type: str,
        log_message: str,
        timestamp: str,
        resolved: int,
    ) -> int:
        """Append one audit row and return its auto-incremented audit_id."""
        with self._connection() as conn:
            cursor = conn.execute(
                f"""
                INSERT INTO {AUDIT_TABLE}
                    (session_id, log_type, log_message, timestamp, resolved)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, log_type, log_message, timestamp, resolved),
            )
            return int(cursor.lastrowid)

    def mark_resolved(self, audit_id: int) -> bool:
        """Flip resolved to 1; return True when a pending row was updated."""
        with self._connection() as conn:
            cursor = conn.execute(
                f"UPDATE {AUDIT_TABLE} SET resolved = 1 WHERE audit_id = ? AND resolved = 0",
                (audit_id,),
            )
            return cursor.rowcount > 0


class AuditTableHandler(logging.Handler):
    """Custom handler: every log record becomes one row in the audit table."""

    def __init__(self, store: SqliteAuditStore, session_id: str, level: int = logging.NOTSET):
        super().__init__(level)
        self.store = store
        self.session_id = session_id
        self.last_audit_id: int | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_type = LOG_TYPE_BY_LEVEL.get(
                record.levelno,
                "error" if record.levelno >= logging.ERROR else "info",
            )
            timestamp = datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="seconds")
            self.last_audit_id = self.store.insert(
                session_id=getattr(record, "session_id", self.session_id),
                log_type=log_type,
                log_message=record.getMessage(),
                timestamp=timestamp,
                resolved=1 if log_type in AUTO_RESOLVED_TYPES else 0,
            )
        except Exception:
            # A broken audit sink must never crash the pipeline.
            self.handleError(record)


class AuditLog:
    """Thin facade over the standard logging module plus the audit table."""

    def __init__(self, settings: Settings, store: SqliteAuditStore):
        self.session_id = settings.session_id
        self._store = store
        self._logger = logging.getLogger(f"{LOGGER_NAME}.{settings.session_id}")
        self._logger.setLevel(settings.log_level)
        self._logger.propagate = False
        self._logger.handlers.clear()

        self._table = AuditTableHandler(store, settings.session_id, settings.log_level)
        self._logger.addHandler(self._table)

        if settings.console_logging:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(settings.log_level)
            console.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)-5s [%(session_id)s] %(message)s",
                    "%H:%M:%S",
                )
            )
            self._logger.addHandler(console)

    def _write(self, level: int, message: str, exc_info: bool = False) -> int | None:
        self._logger.log(level, message, exc_info=exc_info, extra={"session_id": self.session_id})
        return self._table.last_audit_id

    def debug(self, message: str) -> int | None:
        return self._write(logging.DEBUG, message)

    def info(self, message: str) -> int | None:
        return self._write(logging.INFO, message)

    def warning(self, message: str) -> int | None:
        return self._write(logging.WARNING, message)

    def error(self, message: str) -> int | None:
        return self._write(logging.ERROR, message)

    def failure(self, message: str, exc: BaseException) -> int | None:
        """Log an error row with a custom message that summarises the exception."""
        return self._write(logging.ERROR, f"{message}: {type(exc).__name__}: {exc}")

    def resolve(self, audit_id: int) -> bool:
        resolved = self._store.mark_resolved(audit_id)
        if resolved:
            self.info(f"Audit entry {audit_id} marked as resolved")
        else:
            self.warning(f"Audit entry {audit_id} not found or already resolved")
        return resolved


# --- 4. GEMINI CLIENT ---


def create_client(settings: Settings, audit: AuditLog) -> genai.Client:
    """Sub-logic: build the Gemini Enterprise client."""
    try:
        client = genai.Client(
            enterprise=True,
            project=settings.project_id,
            location=settings.location,
        )
    except Exception as exc:
        audit.failure("Failed to create Gemini client", exc)
        raise
    audit.info(f"Gemini client created (project={settings.project_id}, location={settings.location})")
    return client


# --- 5. SINGLE-FILE PROCESSING ---


def clean_file(client: genai.Client, settings: Settings, audit: AuditLog, raw_path: Path) -> str:
    """Sub-logic: read one raw file and send it to Gemini."""
    try:
        raw_text = raw_path.read_text(encoding="utf-8")
        audit.debug(f"Sending '{raw_path.name}' to {settings.model} ({len(raw_text)} chars)")

        response = client.models.generate_content(
            model=settings.model,
            contents=f"{SYSTEM_PROMPT}\n\n--- RAW TEXT ---\n{raw_text}",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="LOW"),  # LOW is faster/cheaper for cleaning
            ),
        )
        cleaned = response.text
        if not cleaned or not cleaned.strip():
            raise ValueError("the model returned an empty response")
        return cleaned
    except Exception as exc:
        audit.failure(f"Failed to clean '{raw_path.name}'", exc)
        raise


def process_file(client: genai.Client, settings: Settings, audit: AuditLog, raw_path: Path) -> str:
    """Sub-logic: clean one file and write it out. Returns processed|skipped|failed."""
    output_path = settings.cleaned_dir / raw_path.name

    if output_path.exists():
        audit.info(f"Skipped '{raw_path.name}': already cleaned at '{output_path}'")
        return "skipped"

    started = time.perf_counter()
    try:
        cleaned = clean_file(client, settings, audit, raw_path)
    except Exception:
        return "failed"

    try:
        output_path.write_text(cleaned, encoding="utf-8")
    except OSError as exc:
        audit.failure(f"Failed to write '{output_path}'", exc)
        return "failed"

    duration = time.perf_counter() - started
    audit.info(f"Cleaned '{raw_path.name}' -> '{output_path}' ({len(cleaned)} chars in {duration:.2f}s)")
    return "processed"


# --- 6. PIPELINE ORCHESTRATION ---


def discover_raw_files(settings: Settings, audit: AuditLog) -> list[Path]:
    """Sub-logic: list the raw input files."""
    try:
        raw_files = sorted(settings.raw_dir.glob("*.txt"))
    except OSError as exc:
        audit.failure(f"Failed to list files in '{settings.raw_dir}'", exc)
        raise

    if raw_files:
        audit.info(f"Discovered {len(raw_files)} raw file(s) in '{settings.raw_dir}'")
    else:
        audit.warning(f"No .txt files found in '{settings.raw_dir}'")
    return raw_files


def run_pipeline(settings: Settings, audit: AuditLog) -> int:
    """Sub-logic: drive the whole run and return the process exit code."""
    started = time.perf_counter()
    audit.info(
        f"Pipeline started (session_id={settings.session_id}, model={settings.model}, "
        f"raw_dir='{settings.raw_dir}', cleaned_dir='{settings.cleaned_dir}')"
    )

    try:
        client = create_client(settings, audit)
        raw_files = discover_raw_files(settings, audit)
    except Exception:
        audit.error("Pipeline aborted before processing any file")
        return 1

    counts = {"processed": 0, "skipped": 0, "failed": 0}
    for raw_path in raw_files:
        counts[process_file(client, settings, audit, raw_path)] += 1

    duration = time.perf_counter() - started
    summary = (
        f"Pipeline finished (processed={counts['processed']}, skipped={counts['skipped']}, "
        f"failed={counts['failed']}, duration={duration:.2f}s)"
    )
    if counts["failed"]:
        audit.error(summary)
        return 1
    audit.info(summary)
    return 0


# --- 7. CLI ENTRY POINT ---


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean scraped text files and audit every step.")
    parser.add_argument(
        "--resolve",
        type=int,
        metavar="AUDIT_ID",
        help="mark an audit entry as resolved and exit",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    session_id = uuid.uuid4().hex
    settings = load_settings(session_id)

    try:
        store = SqliteAuditStore(settings.audit_db)
    except Exception:
        # Audit table unavailable: fall back to the plain logging module.
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        logging.getLogger(LOGGER_NAME).exception("Audit table unavailable at '%s'", settings.audit_db)
        return 1

    audit = AuditLog(settings, store)

    if args.resolve is not None:
        if args.resolve <= 0:
            audit.error(f"Invalid audit id '{args.resolve}': expected a positive integer")
            return 1
        audit.resolve(args.resolve)
        return 0

    audit.info(
        f"Application run started (session_id={session_id}, model={settings.model}, "
        f"log_level={logging.getLevelName(settings.log_level)})"
    )

    if not validate_settings(settings, audit):
        return 1

    try:
        prepare_directories(settings)
    except OSError as exc:
        audit.failure("Failed to prepare pipeline directories", exc)
        return 1
    audit.info(f"Directories ready (raw='{settings.raw_dir}', cleaned='{settings.cleaned_dir}')")

    return run_pipeline(settings, audit)


if __name__ == "__main__":
    sys.exit(main())
