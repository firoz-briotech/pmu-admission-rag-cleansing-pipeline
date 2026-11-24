"""Scrape-data cleansing pipeline with a SQLite-backed audit log (GCS backend).

The script is split into small sub-logics so every step can be audited on its
own:

    1. imports + constants + configuration
    2. audit table + custom logging handler
    3. Gemini client
    4. GCS object processing
    5. pipeline orchestration
    6. CLI entry point

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
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.cloud import storage

# --- 1. CONSTANTS & CONFIGURATION ---
load_dotenv()

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

# This is where you define what "cleaning" means for your scrape data
SYSTEM_PROMPT = """
You are a data cleaning assistant. The user will give you raw scraped text from a website crawler.
Your job is to extract ONLY the main content — the actual information in the center of the page.
Remove: navigation menus, headers, footers, ads, breadcrumbs, repeated links, boilerplate text.
Keep: article body, descriptions, main text, factual content.
Return ONLY the cleaned text, nothing else. No explanations, no preamble.
"""


def parse_log_level(raw: str | None) -> int:
    """Translate LOG_LEVEL=DEBUG|INFO|... into a logging level, defaulting to INFO."""
    level = logging.getLevelName((raw or "").strip().upper())
    return level if isinstance(level, int) else logging.INFO


# Configuration is read once, straight from the environment / .env file.
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "")
MODEL = os.getenv("GEMINI_MODEL", "")
INPUT_BUCKET = os.getenv("GCS_INPUT_BUCKET", "pmu-admission-rag-ingestion-input")
OUTPUT_BUCKET = os.getenv("GCS_OUTPUT_BUCKET", "pmu-admission-rag-ingestion-output")
AUDIT_DB_PATH = Path(os.getenv("AUDIT_DB_PATH", DEFAULT_AUDIT_DB))
LOG_LEVEL = parse_log_level(os.getenv("LOG_LEVEL"))
CONSOLE_LOGGING = os.getenv("AUDIT_CONSOLE", "1") != "0"


def validate_settings(audit: "AuditLog") -> bool:
    """Sub-logic: fail fast with one clear error row if config is incomplete."""
    missing = [
        name
        for name, value in (
            ("GOOGLE_CLOUD_PROJECT", PROJECT_ID),
            ("GOOGLE_CLOUD_LOCATION", LOCATION),
            ("GEMINI_MODEL", MODEL),
            ("GCS_INPUT_BUCKET", INPUT_BUCKET),
            ("GCS_OUTPUT_BUCKET", OUTPUT_BUCKET),
        )
        if not value
    ]
    if missing:
        audit.error(f"Configuration invalid: missing environment variables {', '.join(missing)} (check .env)")
        return False
    audit.info(
        f"Configuration loaded (project={PROJECT_ID}, location={LOCATION}, model={MODEL})"
    )
    return True


# --- 2. AUDIT LOGGING ---


class SqliteAuditStore:
    """Persistence for the audit table. Swap this class for BigQuery/Postgres later."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.executescript(AUDIT_SCHEMA)

    @contextmanager
    def _connection(self):
        """Open a connection, commit on success and always close it (no leaks)."""
        conn = sqlite3.connect(self.db_path, timeout=10)
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

    def __init__(
        self,
        session_id: str,
        store: SqliteAuditStore,
        log_level: int = LOG_LEVEL,
        console_logging: bool = CONSOLE_LOGGING,
    ):
        self.session_id = session_id
        self._store = store
        self._logger = logging.getLogger(f"{LOGGER_NAME}.{session_id}")
        self._logger.setLevel(log_level)
        self._logger.propagate = False
        self._logger.handlers.clear()

        self._table = AuditTableHandler(store, session_id, log_level)
        self._logger.addHandler(self._table)

        if console_logging:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(log_level)
            console.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)-5s [%(session_id)s] %(message)s",
                    "%H:%M:%S",
                )
            )
            self._logger.addHandler(console)

    def _write(self, level: int, message: str) -> int | None:
        self._logger.log(level, message, extra={"session_id": self.session_id})
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


# --- 3. GEMINI CLIENT ---


def create_client(audit: AuditLog) -> genai.Client:
    """Sub-logic: build the Gemini Enterprise client."""
    try:
        client = genai.Client(
            enterprise=True,
            project=PROJECT_ID,
            location=LOCATION,
        )
    except Exception as exc:
        audit.failure("Failed to create Gemini client", exc)
        raise
    audit.info(f"Gemini client created (project={PROJECT_ID}, location={LOCATION})")
    return client


# --- 4. GCS OBJECT PROCESSING ---


def clean_text(client: genai.Client, audit: AuditLog, file_name: str, raw_text: str) -> str:
    """Sub-logic: send one raw object's text to Gemini."""
    try:
        audit.debug(f"Sending file '{file_name}' to {MODEL} ({len(raw_text)} chars)")

        response = client.models.generate_content(
            model=MODEL,
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
        audit.failure(f"Failed to clean file '{file_name}'", exc)
        raise


def discover_gcs_files(audit: AuditLog) -> tuple[storage.Bucket, storage.Bucket, list[str]]:
    """List input objects and return only objects absent from the output bucket."""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        input_bucket = storage_client.bucket(INPUT_BUCKET)
        output_bucket = storage_client.bucket(OUTPUT_BUCKET)
        input_names = sorted(
            blob.name for blob in storage_client.list_blobs(input_bucket)
            if not blob.name.endswith("/")
        )
        output_names = {
            blob.name for blob in storage_client.list_blobs(output_bucket)
            if not blob.name.endswith("/")
        }
    except Exception as exc:
        audit.failure(
            f"Failed to compare GCS buckets '{INPUT_BUCKET}' and '{OUTPUT_BUCKET}'",
            exc,
        )
        raise

    pending = [name for name in input_names if name not in output_names]
    audit.info(
        f"Compared GCS buckets (input_files={len(input_names)}, output_files={len(output_names)}, "
        f"pending_files={len(pending)}, input_bucket='{INPUT_BUCKET}', "
        f"output_bucket='{OUTPUT_BUCKET}')"
    )
    if not input_names:
        audit.warning(f"No raw files found in input bucket '{INPUT_BUCKET}'")
    elif not pending:
        audit.info("No new files require cleansing")
    else:
        for name in pending:
            audit.info(f"Queued file '{name}' for cleansing")
    return input_bucket, output_bucket, pending


def process_gcs_file(
    client: genai.Client,
    audit: AuditLog,
    input_bucket: storage.Bucket,
    output_bucket: storage.Bucket,
    file_name: str,
) -> str:
    """Download, cleanse, and upload one GCS object using the same object name."""
    started = time.perf_counter()
    try:
        raw_text = input_bucket.blob(file_name).download_as_text(encoding="utf-8")
        cleaned = clean_text(client, audit, file_name, raw_text)
        output_bucket.blob(file_name).upload_from_string(
            cleaned, content_type="text/plain; charset=utf-8"
        )
    except Exception as exc:
        audit.failure(f"Failed to process file '{file_name}'", exc)
        return "failed"

    duration = time.perf_counter() - started
    audit.info(
        f"Cleansed file '{file_name}' from gs://{INPUT_BUCKET}/{file_name} "
        f"to gs://{OUTPUT_BUCKET}/{file_name} ({len(cleaned)} chars in {duration:.2f}s)"
    )
    return "processed"


# --- 5. PIPELINE ORCHESTRATION ---


def run_pipeline(audit: AuditLog) -> int:
    """Sub-logic: drive the whole run and return the process exit code."""
    started = time.perf_counter()
    audit.info(
        f"Pipeline started (session_id={audit.session_id}, model={MODEL}, "
        f"input_bucket='{INPUT_BUCKET}', output_bucket='{OUTPUT_BUCKET}')"
    )

    try:
        client = create_client(audit)
        input_bucket, output_bucket, raw_files = discover_gcs_files(audit)
    except Exception:
        audit.error("Pipeline aborted before processing any file")
        return 1

    counts = {"processed": 0, "failed": 0}
    for raw_file in raw_files:
        counts[process_gcs_file(client, audit, input_bucket, output_bucket, raw_file)] += 1

    duration = time.perf_counter() - started
    summary = (
        f"Pipeline finished (processed={counts['processed']}, failed={counts['failed']}, "
        f"duration={duration:.2f}s)"
    )
    if counts["failed"]:
        audit.error(summary)
        return 1
    audit.info(summary)
    return 0


# --- 6. CLI ENTRY POINT ---


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean scraped text files and audit every step.")
    parser.add_argument(
        "--resolve",
        type=int,
        metavar="AUDIT_ID",
        help="mark an audit entry as resolved and exit",
    )
    args = parser.parse_args(argv)

    session_id = uuid.uuid4().hex

    try:
        store = SqliteAuditStore(AUDIT_DB_PATH)
    except Exception:
        # Audit table unavailable: fall back to the plain logging module.
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        logging.getLogger(LOGGER_NAME).exception("Audit table unavailable at '%s'", AUDIT_DB_PATH)
        return 1

    audit = AuditLog(session_id, store)

    if args.resolve is not None:
        if args.resolve <= 0:
            audit.error(f"Invalid audit id '{args.resolve}': expected a positive integer")
            return 1
        audit.resolve(args.resolve)
        return 0

    audit.info(
        f"Application run started (session_id={session_id}, model={MODEL}, "
        f"log_level={logging.getLevelName(LOG_LEVEL)})"
    )

    if not validate_settings(audit):
        return 1

    audit.info(
        f"GCS storage ready (input='gs://{INPUT_BUCKET}', output='gs://{OUTPUT_BUCKET}')"
    )

    return run_pipeline(audit)


if __name__ == "__main__":
    sys.exit(main())
