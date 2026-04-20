#!/bin/bash
#SBATCH -p acd_u
#SBATCH -J proj1_1v3g
#SBATCH -o /data/user/dingcao/hanyang/proj1/spec_diag/slurm_logs/proj1_%j.out
#SBATCH -e /data/user/dingcao/hanyang/proj1/spec_diag/slurm_logs/proj1_%j.err
#SBATCH -n 16
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
#SBATCH -D /data/user/dingcao/hanyang/proj1/spec_diag

set -euo pipefail

module load anaconda3 || true
module load cuda/12.4 || true
source "$HOME/.conda/etc/profile.d/conda.sh" 2>/dev/null || \
  source /share/anaconda3/etc/profile.d/conda.sh
conda activate selfevolve

# ---------- paths / config ----------
PROJ_ROOT="${PROJ_ROOT:-/data/user/dingcao/hanyang/proj1/spec_diag}"
MODEL="${MODEL:-/data/user/dingcao/hanyang/proj1/models/Qwen3-8B}"
SERVED_NAME="${SERVED_NAME:-generator}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$PROJ_ROOT/logs/run_${SLURM_JOB_ID}_${TS}}"
VLLM_LOG="$RUN_DIR/vllm.log"
TRAIN_LOG="$RUN_DIR/train.log"
mkdir -p "$RUN_DIR" "$PROJ_ROOT/slurm_logs"

export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="${HF_HOME:-$PROJ_ROOT/.hf_cache}"
mkdir -p "$HF_HOME"

echo "==== proj1/spec_diag  1 vLLM + 3 GRPO on 4 GPU ===="
echo "host  : $(hostname)"
echo "jobid : ${SLURM_JOB_ID:-<nojob>}"
echo "cwd   : $(pwd)"
echo "model : $MODEL"
echo "env   : selfevolve"
echo "run   : $RUN_DIR"
nvidia-smi || true
echo "==================================================="

# ---------- 1) vLLM on GPU 0 ----------
(
  export CUDA_VISIBLE_DEVICES=0
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
    > "$VLLM_LOG" 2>&1
) &
VLLM_PID=$!
echo "vLLM pid=$VLLM_PID -> $VLLM_LOG"

cleanup() {
  echo "[cleanup] killing vLLM pid=$VLLM_PID"
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---------- 2) wait for vLLM health ----------
echo "[wait] vLLM health @ http://127.0.0.1:${VLLM_PORT}/v1/models ..."
READY=0
for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    READY=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM process died early, see $VLLM_LOG" >&2
    exit 10
  fi
  sleep 2
done
if [ "$READY" -ne 1 ]; then
  echo "ERROR: vLLM not healthy within 600s, see $VLLM_LOG" >&2
  exit 11
fi
echo "[ok] vLLM healthy."

# ---------- 3) GRPO on GPUs 1,2,3 ----------
export CUDA_VISIBLE_DEVICES=1,2,3
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export SPEC_DIAG_MODEL="$SERVED_NAME"

# 训练入口：按需用 TRAIN_CMD 覆盖（sbatch --export=ALL,TRAIN_CMD="..."）
TRAIN_CMD="${TRAIN_CMD:-python -m spec_diag.train}"

echo "[run] $TRAIN_CMD"
set +e
bash -c "$TRAIN_CMD" 2>&1 | tee "$TRAIN_LOG"
RC=${PIPESTATUS[0]}
set -e

echo "[done] train rc=$RC"
exit "$RC"
