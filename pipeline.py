"""Scrape-data cleansing pipeline with structured JSON audit logging (GCS backend).

The script is split into small sub-logics so every step can be audited on its
own:

    1. imports + constants + configuration
    2. structured JSON audit log
    3. Gemini client
    4. GCS object processing
    5. pipeline orchestration
    6. CLI entry point

Each sub-logic wraps its work in try/except and writes one structured log line
per outcome: `info` when it succeeds, `error` when it fails (plus `warning` and
`debug` where useful). Each line is emitted as single-line JSON on stdout so
Cloud Logging can parse it when the job runs on Cloud Run.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.cloud import storage

# --- 1. CONSTANTS & CONFIGURATION ---
load_dotenv()

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
LOG_LEVEL = parse_log_level(os.getenv("LOG_LEVEL"))
SEVERITY_BY_LEVEL = {
    logging.DEBUG:    "DEBUG",
    logging.INFO:     "INFO",
    logging.WARNING:  "WARNING",
    logging.ERROR:    "ERROR",
    logging.CRITICAL: "CRITICAL",
}


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


class AuditLog:
    def __init__(self, session_id: str, log_level: int = LOG_LEVEL):
        self.session_id = session_id
        self._log_level = log_level

    def _emit(self, level: int, message: str, **extra) -> None:
        if level < self._log_level:
            return
        entry = {
            "severity": SEVERITY_BY_LEVEL.get(level, "DEFAULT"),
            "message": message,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "logging.googleapis.com/labels": {"session_id": self.session_id},
            **extra,
        }
        print(json.dumps(entry, default=str), file=sys.stdout, flush=True)

    def debug(self, message: str) -> None:
        self._emit(logging.DEBUG, message)

    def info(self, message: str) -> None:
        self._emit(logging.INFO, message)

    def warning(self, message: str) -> None:
        self._emit(logging.WARNING, message)

    def error(self, message: str) -> None:
        self._emit(logging.ERROR, message)

    def failure(self, message: str, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._emit(
            logging.ERROR,
            f"{message}: {type(exc).__name__}: {exc}",
            stack_trace=tb,
        )


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


def discover_gcs_files(audit: AuditLog, force_reprocess: bool = False) -> tuple[storage.Bucket, storage.Bucket, list[str]]:
    """List input objects and return only objects absent from the output bucket."""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        input_bucket = storage_client.bucket(INPUT_BUCKET)
        output_bucket = storage_client.bucket(OUTPUT_BUCKET)

        if force_reprocess:
            audit.warning("force-reprocess enabled: deleting all output bucket files")
            blobs_to_delete = list(storage_client.list_blobs(output_bucket))
            for blob in blobs_to_delete:
                blob.delete()
            audit.warning(f"Deleted {len(blobs_to_delete)} files from output bucket '{OUTPUT_BUCKET}'")
            output_names = set()  # Empty set → all input files become pending

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


def run_pipeline(audit: AuditLog, force_reprocess: bool = False) -> int:
    """Sub-logic: drive the whole run and return the process exit code."""
    started = time.perf_counter()
    audit.info(
        f"Pipeline started (session_id={audit.session_id}, model={MODEL}, "
        f"input_bucket='{INPUT_BUCKET}', output_bucket='{OUTPUT_BUCKET}')"
    )

    try:
        client = create_client(audit)
        input_bucket, output_bucket, raw_files = discover_gcs_files(audit, force_reprocess)
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

    #for testing purpose only
    parser.add_argument(
    "--force-reprocess",
    action="store_true",
    help="Delete all output bucket files before running (testing only)"
    )
    args = parser.parse_args(argv)

    session_id = uuid.uuid4().hex

    audit = AuditLog(session_id)

    audit.info(
        f"Application run started (session_id={session_id}, model={MODEL}, "
        f"log_level={logging.getLevelName(LOG_LEVEL)})"
    )

    if not validate_settings(audit):
        return 1

    audit.info(
        f"GCS storage ready (input='gs://{INPUT_BUCKET}', output='gs://{OUTPUT_BUCKET}')"
    )

    return run_pipeline(audit, force_reprocess=args.force_reprocess)


if __name__ == "__main__":
    sys.exit(main())
