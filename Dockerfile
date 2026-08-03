# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Build-time system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging metadata + source
COPY pyproject.toml ./
COPY src/      ./src/
COPY configs/  ./configs/
COPY prompts/  ./prompts/
COPY web/      ./web/

# Install into system Python (no --prefix) so every subsequent RUN can import
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# Pre-download embedding model — non-fatal so CI build succeeds even if
# HuggingFace Hub is unreachable or the model download is slow
ENV HF_HOME=/app/.cache/huggingface
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('Embedding model cached.')" \
    || echo "⚠️  Model pre-cache skipped (will download on first request)"

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-cached model (may be empty if download was skipped — that's fine)
COPY --from=builder /app/.cache /app/.cache

# Copy application source
COPY src/      ./src/
COPY configs/  ./configs/
COPY prompts/  ./prompts/
COPY web/      ./web/

# Persistent data dirs
RUN mkdir -p data/raw data/memory data/uploads

# Point HF to the pre-cached model
ENV HF_HOME=/app/.cache/huggingface

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
