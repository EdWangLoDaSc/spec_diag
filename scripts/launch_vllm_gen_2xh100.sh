#!/usr/bin/env bash
# 启动 vLLM generator (Qwen3-8B) —— 2x H100 80GB, tensor-parallel=2
# 用法:
#   bash scripts/launch_vllm_gen_2xh100.sh               # 前台
#   nohup bash scripts/launch_vllm_gen_2xh100.sh &       # 后台
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/launch_vllm_gen_2xh100.sh
#   MODEL=Qwen/Qwen2.5-14B-Instruct VLLM_PORT=8000 bash ...

set -euo pipefail

# ---- 路径 ----
SPEC_DIAG_ROOT="${SPEC_DIAG_ROOT:-/data/user/dingcao/hanyang/spec_diag}"
LOG_DIR="$SPEC_DIAG_ROOT/logs/vllm"
mkdir -p "$LOG_DIR"

# ---- conda ----
CONDA_ENV="${CONDA_ENV:-/share/anaconda3/envs/vllm}"
set +u
if [ -f /share/anaconda3/etc/profile.d/conda.sh ]; then
  source /share/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
  source "$HOME/.conda/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found" >&2; exit 1
fi
conda activate "$CONDA_ENV"
set -u

# ---- GPU 选卡 (默认 0,1；外部可覆盖) ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TP_SIZE=2
N_VIS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
if [ "$N_VIS" -ne "$TP_SIZE" ]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' has $N_VIS gpus, TP_SIZE=$TP_SIZE" >&2
  exit 1
fi

# ---- runtime env ----
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="${HF_HOME:-$SPEC_DIAG_ROOT/.hf_cache}"
mkdir -p "$HF_HOME"

# ---- 模型 / 端口 / 长度 ----
MODEL="${MODEL:-Qwen/Qwen3-8B}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
SERVED_NAME="${SERVED_NAME:-generator}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/vllm_2xh100_${TS}.log"
PID_FILE="$LOG_DIR/vllm_2xh100.pid"

echo "==== vLLM 2x H100 ===="
echo "host      : $(hostname)"
echo "gpus      : $CUDA_VISIBLE_DEVICES  (TP=$TP_SIZE)"
echo "model     : $MODEL  (served as '$SERVED_NAME')"
echo "port      : $VLLM_PORT"
echo "max_len   : $MAX_MODEL_LEN"
echo "gpu_util  : $GPU_MEM_UTIL"
echo "log       : $LOG_FILE"
echo "base_url  : http://$(hostname):$VLLM_PORT/v1"
echo "======================"
nvidia-smi || true

# ---- 启动 ----
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --dtype bfloat16 \
  --trust-remote-code \
  2>&1 | tee "$LOG_FILE" &

VLLM_PID=$!
echo "$VLLM_PID" > "$PID_FILE"
echo "vllm pid=$VLLM_PID (saved to $PID_FILE)"
wait "$VLLM_PID"
