# Cloud Run Job image for the cleansing pipeline.
FROM python:3.13-slim

# Unbuffered + UTF-8 stdout so every JSON log line is flushed and encodable.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so this layer stays cached when only pipeline.py changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser pipeline.py .
USER appuser

# No .env in the image: configuration comes from Cloud Run env vars.
CMD ["python", "pipeline.py"]
