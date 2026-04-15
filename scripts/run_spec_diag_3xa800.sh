#!/usr/bin/env bash
# spec_diag end-to-end experiment on 3x A800 80GB
# ================================================
# Layout:
#   GPU 0         → vLLM generator (Qwen3-8B, TP=1)       background
#   GPU 1, GPU 2  → spec_diag GRPO student training        foreground
#
# This script is designed to run inside a single interactive srun
# allocation (or a single sbatch job) that already holds 3 A800 cards.
# It is NOT a sbatch submitter itself — use scripts/submit_proj.sh or
# an equivalent one-liner to allocate the cards, then run this inside.
#
# Usage:
#   # Inside an allocation with CUDA_VISIBLE_DEVICES=0,1,2:
#   bash scripts/run_spec_diag_3xa800.sh
#
#   # Override the student model or training knobs (Hydra syntax,
#   # forwarded verbatim to spec_diag.train):
#   STUDENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
#   TRAIN_OVERRIDES="spec_diag.feeder.warmup_tasks=64 trainer.total_epochs=2" \
#   bash scripts/run_spec_diag_3xa800.sh
#
# Env vars (all optional):
#   SPEC_DIAG_ROOT      repo root (default: twoji path below)
#   CONDA_ENV           conda env name (default: spec_diag)
#   STUDENT_MODEL       HF id or local path passed to
#                       actor_rollout_ref.model.path
#   VLLM_PORT           vLLM port (default 8000)
#   VLLM_HEALTH_TIMEOUT_S  how long to wait for /v1/models (default 600)
#   GENERATOR_MODEL     model vLLM serves (default Qwen/Qwen3-8B)
#   TRAIN_OVERRIDES     extra Hydra args appended to the training command
#   KEEP_VLLM_ON_EXIT   1 = leave vLLM running after training (default 0)

set -euo pipefail

# ----------------------------------------------------------- paths / env
SPEC_DIAG_ROOT="${SPEC_DIAG_ROOT:-/hpc2hdd/home/dingcao/hanyang/proj1/spec_diag}"
CONDA_ENV="${CONDA_ENV:-spec_diag}"
LOG_ROOT="$SPEC_DIAG_ROOT/logs/e2e_3xa800"
mkdir -p "$LOG_ROOT" "$SPEC_DIAG_ROOT/logs/vllm"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_ROOT/run_${TS}.log"
VLLM_LOG="$SPEC_DIAG_ROOT/logs/vllm/vllm_1xa800_${TS}.log"
TRAIN_LOG="$LOG_ROOT/train_${TS}.log"

# ----------------------------------------------------------- GPU check
# This script assumes exactly 3 visible GPUs. Split them 1 + 2.
VISIBLE="${CUDA_VISIBLE_DEVICES:-}"
if [ -z "$VISIBLE" ]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES is unset. Run this inside an srun/sbatch"
  echo "       allocation that holds 3 A800 cards, or export it manually."
  exit 2
fi
N_GPUS=$(echo "$VISIBLE" | awk -F',' '{print NF}')
if [ "$N_GPUS" -ne 3 ]; then
  echo "ERROR: this orchestrator needs exactly 3 visible GPUs, got $N_GPUS"
  echo "       CUDA_VISIBLE_DEVICES='$VISIBLE'"
  exit 2
fi

IFS=',' read -r GPU_VLLM GPU_TRAIN0 GPU_TRAIN1 <<< "$VISIBLE"
echo "==== spec_diag 3x A800 orchestrator ===="
echo "host          : $(hostname)"
echo "visible       : $VISIBLE"
echo "vllm gpu      : $GPU_VLLM"
echo "train gpus    : $GPU_TRAIN0,$GPU_TRAIN1"
echo "run log       : $RUN_LOG"
echo "vllm log      : $VLLM_LOG"
echo "train log     : $TRAIN_LOG"
echo "========================================"

# From here on, mirror stdout/stderr into the run log.
exec > >(tee -a "$RUN_LOG") 2>&1

# ----------------------------------------------------------- conda
set +u
if [ -f /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh ]; then
  source /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
  source "$HOME/.conda/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found" >&2; exit 1
fi
conda activate "$CONDA_ENV"
set -u

# ----------------------------------------------------------- knobs
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HEALTH_TIMEOUT_S="${VLLM_HEALTH_TIMEOUT_S:-600}"
GENERATOR_MODEL="${GENERATOR_MODEL:-Qwen/Qwen3-8B}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_OVERRIDES="${TRAIN_OVERRIDES:-}"
KEEP_VLLM_ON_EXIT="${KEEP_VLLM_ON_EXIT:-0}"

export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="${HF_HOME:-$SPEC_DIAG_ROOT/.hf_cache}"
export RAY_DEDUP_LOGS=0

# ----------------------------------------------------------- cleanup trap
VLLM_PID=""
cleanup() {
  set +e
  if [ -n "$VLLM_PID" ] && [ "$KEEP_VLLM_ON_EXIT" != "1" ]; then
    if kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[cleanup] stopping vLLM pid=$VLLM_PID"
      kill "$VLLM_PID" 2>/dev/null || true
      # give it a moment to release the GPU, then SIGKILL if needed
      for _ in 1 2 3 4 5; do
        kill -0 "$VLLM_PID" 2>/dev/null || break
        sleep 1
      done
      kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT INT TERM

# ------------------------------------------------- 1. launch vLLM (bg)
echo "---- [1/3] launching vLLM generator on GPU $GPU_VLLM ----"
(
  CUDA_VISIBLE_DEVICES="$GPU_VLLM" \
  MODEL="$GENERATOR_MODEL" \
  VLLM_PORT="$VLLM_PORT" \
  SPEC_DIAG_ROOT="$SPEC_DIAG_ROOT" \
  CONDA_ENV="$CONDA_ENV" \
  bash "$SPEC_DIAG_ROOT/scripts/launch_vllm_gen_1xa800.sh"
) > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "vllm pid=$VLLM_PID log=$VLLM_LOG"

# ------------------------------------------------- 2. wait for health
echo "---- [2/3] waiting for vLLM /v1/models (timeout ${VLLM_HEALTH_TIMEOUT_S}s) ----"
BASE_URL="http://127.0.0.1:$VLLM_PORT/v1"
waited=0
until curl -sf "$BASE_URL/models" >/dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM process died before becoming healthy. Tail of log:"
    tail -n 60 "$VLLM_LOG" || true
    exit 3
  fi
  if [ "$waited" -ge "$VLLM_HEALTH_TIMEOUT_S" ]; then
    echo "ERROR: vLLM did not become healthy within ${VLLM_HEALTH_TIMEOUT_S}s"
    tail -n 60 "$VLLM_LOG" || true
    exit 3
  fi
  sleep 5
  waited=$((waited + 5))
done
echo "vLLM healthy after ${waited}s"
curl -s "$BASE_URL/models" | head -c 300; echo

# ------------------------------------------------- 3. launch training (fg)
echo "---- [3/3] launching spec_diag GRPO training on GPUs $GPU_TRAIN0,$GPU_TRAIN1 ----"
export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_API_KEY="dummy"
export SPEC_DIAG_MODEL="generator"

# NOTE: CUDA_VISIBLE_DEVICES must only expose the 2 training GPUs to verl.
# verl's `trainer.n_gpus_per_node` should match len(CUDA_VISIBLE_DEVICES).
CUDA_VISIBLE_DEVICES="$GPU_TRAIN0,$GPU_TRAIN1" \
python -m spec_diag.train \
  actor_rollout_ref.model.path="$STUDENT_MODEL" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.project_name='spec_diag' \
  trainer.experiment_name="phase0_3xa800_${TS}" \
  trainer.logger='["console"]' \
  $TRAIN_OVERRIDES \
  2>&1 | tee "$TRAIN_LOG"

echo "---- training finished; cleanup trap will stop vLLM ----"
