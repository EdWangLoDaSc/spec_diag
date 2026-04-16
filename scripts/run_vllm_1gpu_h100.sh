#!/usr/bin/env bash
# Start vLLM generator on exactly 1 GPU.
# Model defaults to local path:
#   /home/apulis-dev/models/Qwen25-7B-Instruct

set -euo pipefail

SPEC_DIAG_ROOT="${SPEC_DIAG_ROOT:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-spec_diag}"
MODEL="${MODEL:-/home/apulis-dev/models/Qwen25-7B-Instruct/V0/code/Qwen2.5-7B-Instruct}"
SERVED_NAME="${SERVED_NAME:-generator}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
HF_HOME="${HF_HOME:-$SPEC_DIAG_ROOT/.hf_cache}"
LOG_DIR="${LOG_DIR:-$SPEC_DIAG_ROOT/logs/vllm_split}"

mkdir -p "$LOG_DIR" "$HF_HOME"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${VLLM_LOG_FILE:-$LOG_DIR/vllm_${TS}.log}"
PID_FILE="${VLLM_PID_FILE:-$LOG_DIR/vllm.pid}"

set +u
if [ -f /share/anaconda3/etc/profile.d/conda.sh ]; then
  source /share/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
  source "$HOME/.conda/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found" >&2
  exit 1
fi
conda activate "$CONDA_ENV"
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
N_VIS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
if [ "$N_VIS" -ne 1 ]; then
  echo "ERROR: vLLM script needs exactly 1 visible GPU; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME

echo "==== vLLM split launcher ===="
echo "host      : $(hostname)"
echo "model     : $MODEL"
echo "served as : $SERVED_NAME"
echo "port      : $VLLM_PORT"
echo "gpus      : $CUDA_VISIBLE_DEVICES"
echo "log       : $LOG_FILE"
echo "============================="
nvidia-smi || true

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --dtype bfloat16 \
  --trust-remote-code \
  2>&1 | tee "$LOG_FILE" &

VLLM_PID=$!
echo "$VLLM_PID" > "$PID_FILE"
echo "vLLM pid=$VLLM_PID (pid file: $PID_FILE)"

echo "Waiting for health endpoint..."
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM is healthy at http://127.0.0.1:${VLLM_PORT}/v1"
    wait "$VLLM_PID"
    exit $?
  fi
  sleep 2
done

echo "ERROR: vLLM did not become healthy within timeout." >&2
kill "$VLLM_PID" 2>/dev/null || true
exit 3
