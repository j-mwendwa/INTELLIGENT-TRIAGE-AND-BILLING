# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build-time system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only packaging metadata first so pip layer is cached unless deps change
COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY prompts/ ./prompts/
COPY web/ ./web/

# Install everything (including dev extras for tests in CI; runtime only here)
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -e ".[dev]"

# Pre-download embedding model so ACA cold starts don't hit HuggingFace Hub
ENV HF_HOME=/install/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('Embedding model cached.')"

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages + cached model from builder
COPY --from=builder /install /usr/local
COPY --from=builder /install/.cache/huggingface /app/.cache/huggingface

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
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
