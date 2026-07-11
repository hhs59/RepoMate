# repomate — Code If You Can
# Multi-stage Dockerfile with GPU support
# Base image: PyTorch GPU (match your host — CUDA, ROCm, etc.)

ARG GPU_BASE=pytorch/pytorch:latest

# ─── Stage 1: Builder ───────────────────────────────────────────────────────
FROM ${GPU_BASE} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml requirements.txt ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[core,ingest,rag,serve,train,ui]"

# ─── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM ${GPU_BASE} AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/conda/lib/python3.11/site-packages/ /opt/conda/lib/python3.11/site-packages/
COPY --from=builder /build/src/ ./src/
COPY pyproject.toml ./
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENV PYTHONPATH=/app
ENV REPOMATE_DATA_DIR=/app/data
ENV REPOMATE_LOG_LEVEL=INFO

EXPOSE 8000 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=120s \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
