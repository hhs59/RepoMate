#!/usr/bin/env bash
set -e

# repomate entrypoint — starts vLLM, FastAPI
# Traps signals for clean shutdown

trap 'kill 0' SIGINT SIGTERM

# ─── 1. Optionally init a demo repo ─────────────────────────────────────────
if [ -n "$REPOMATE_DEMO_REPO" ] && [ -z "$(ls -A /app/data/ 2>/dev/null)" ]; then
    echo "[entrypoint] Data dir empty — running repomate init $REPOMATE_DEMO_REPO"
    repomate init "$REPOMATE_DEMO_REPO" || echo "[entrypoint] Init failed, continuing anyway"
fi

# ─── 2. Start vLLM server ──────────────────────────────────────────────────
HYBRID_FLAG=""
if [ "$REPOMATE_HYBRID" = "1" ] && [ -d /app/data ]; then
    ADAPTER_DIR=$(find /app/data -maxdepth 2 -name "adapters" -type d | head -1)
    if [ -n "$ADAPTER_DIR" ] && [ -n "$(ls -A "$ADAPTER_DIR" 2>/dev/null)" ]; then
        echo "[entrypoint] Hybrid mode: adapter found at $ADAPTER_DIR"
        HYBRID_FLAG="--enable-lora --lora-modules repomate-ft=$ADAPTER_DIR"
    else
        echo "[entrypoint] Hybrid requested but no adapter found — RAG-only mode"
    fi
fi

VLLM_PORT=${REPOMATE_VLLM_PORT:-8000}
BASE_MODEL=${REPOMATE_BASE_MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}

echo "[entrypoint] Starting vLLM server on port $VLLM_PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.85 \
    $HYBRID_FLAG &
VLLM_PID=$!

# ─── 3. Wait for vLLM ──────────────────────────────────────────────────────
echo "[entrypoint] Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
        echo "[entrypoint] vLLM ready after ${i}s"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "[entrypoint] WARNING: vLLM not ready after 120s — API will use cloud LLM fallback"
    fi
    sleep 1
done

# ─── 4. Start FastAPI ──────────────────────────────────────────────────────
API_PORT=${REPOMATE_API_PORT:-8080}
echo "[entrypoint] Starting FastAPI on port $API_PORT..."
python -m uvicorn src.serve.app:app --host 0.0.0.0 --port "$API_PORT" &
API_PID=$!

echo "[entrypoint] repomate is serving!"
echo "  UI:    http://localhost:$API_PORT"
echo "  API:   http://localhost:$API_PORT/health"
echo "  vLLM:  http://localhost:$VLLM_PORT"

# ─── 5. Wait for any process to exit ───────────────────────────────────────
wait
