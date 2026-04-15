#!/usr/bin/env bash
# spec_diag end-to-end experiment on 3x A800 80GB
# ================================================
# Layout:
#   GPU 0         → vLLM generator (Qwen3-8B, TP=1)      background
#   GPU 1, GPU 2  → spec_diag GRPO student training       foreground
#
# This script is designed to run inside a single interactive srun
# allocation (or a single sbatch job) that already holds 3 A800 cards.
#
# Artifacts: every run gets its own timestamped directory
#   $SPEC_DIAG_ROOT/logs/e2e_3xa800/run_YYYYmmdd_HHMMSS/
#     ├── run.log            orchestrator stdout/stderr
#     ├── env.txt            nvidia-smi + pip freeze + git rev + env snapshot
#     ├── vllm.log           vLLM server stdout/stderr
#     ├── vllm_health.log    periodic curl /v1/models beacon
#     ├── gpu.log            periodic nvidia-smi --query-gpu samples
#     ├── train.log          Python training stdout/stderr (tee)
#     ├── train_python.log   Python logging module output (feeder, trainer, …)
#     ├── config_resolved.yaml  Hydra config after resolution
#     └── status.txt         final exit code + tail of train.log
#
# This layout is the ONLY place to look when something goes wrong. Every
# previous run stays put; nothing is overwritten.
#
# Usage:
#   bash scripts/run_spec_diag_3xa800.sh
#
#   STUDENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
#   TRAIN_OVERRIDES="spec_diag.feeder.warmup_tasks=64 trainer.total_epochs=2" \
#   bash scripts/run_spec_diag_3xa800.sh
#
# Env vars (all optional):
#   SPEC_DIAG_ROOT             repo root (default: 二期 path below)
#   CONDA_ENV                  conda env name (default: spec_diag)
#   STUDENT_MODEL              HF id / local path for the policy
#   GENERATOR_MODEL            model vLLM serves (default Qwen/Qwen3-8B)
#   VLLM_PORT                  vLLM port (default 8000)
#   VLLM_HEALTH_TIMEOUT_S      how long to wait for /v1/models (default 600)
#   TRAIN_OVERRIDES            extra Hydra args appended to the training command
#   GPU_POLL_INTERVAL_S        nvidia-smi sample cadence (default 15)
#   HEALTH_POLL_INTERVAL_S     vLLM health beacon cadence (default 30)
#   KEEP_VLLM_ON_EXIT          1 = leave vLLM running after training (default 0)

set -euo pipefail

# =========================================================== 0. paths + knobs
SPEC_DIAG_ROOT="${SPEC_DIAG_ROOT:-/hpc2hdd/home/dingcao/hanyang/proj1/spec_diag}"
CONDA_ENV="${CONDA_ENV:-spec_diag}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$SPEC_DIAG_ROOT/logs/e2e_3xa800/run_${TS}"
mkdir -p "$RUN_DIR"

RUN_LOG="$RUN_DIR/run.log"
ENV_FILE="$RUN_DIR/env.txt"
VLLM_LOG="$RUN_DIR/vllm.log"
VLLM_HEALTH_LOG="$RUN_DIR/vllm_health.log"
GPU_LOG="$RUN_DIR/gpu.log"
TRAIN_LOG="$RUN_DIR/train.log"
STATUS_FILE="$RUN_DIR/status.txt"

# Convenience symlink to the latest run so the user can `cd logs/.../latest`.
ln -sfn "$RUN_DIR" "$SPEC_DIAG_ROOT/logs/e2e_3xa800/latest"

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HEALTH_TIMEOUT_S="${VLLM_HEALTH_TIMEOUT_S:-600}"
GPU_POLL_INTERVAL_S="${GPU_POLL_INTERVAL_S:-15}"
HEALTH_POLL_INTERVAL_S="${HEALTH_POLL_INTERVAL_S:-30}"
GENERATOR_MODEL="${GENERATOR_MODEL:-Qwen/Qwen3-8B}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_OVERRIDES="${TRAIN_OVERRIDES:-}"
KEEP_VLLM_ON_EXIT="${KEEP_VLLM_ON_EXIT:-0}"

# From here on, mirror orchestrator stdout/stderr into run.log. Must happen
# AFTER we've created RUN_DIR but BEFORE any informational echo.
exec > >(tee -a "$RUN_LOG") 2>&1

echo "==== spec_diag 3x A800 orchestrator ===="
echo "host          : $(hostname)"
echo "timestamp     : $TS"
echo "run dir       : $RUN_DIR"
echo "student model : $STUDENT_MODEL"
echo "generator     : $GENERATOR_MODEL  (vLLM port $VLLM_PORT)"
echo "overrides     : $TRAIN_OVERRIDES"
echo "========================================"

# =========================================================== 1. GPU split
VISIBLE="${CUDA_VISIBLE_DEVICES:-}"
if [ -z "$VISIBLE" ]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES is unset. Run inside an srun/sbatch"
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
echo "vllm gpu      : $GPU_VLLM"
echo "train gpus    : $GPU_TRAIN0,$GPU_TRAIN1"

# =========================================================== 2. conda
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

export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="${HF_HOME:-$SPEC_DIAG_ROOT/.hf_cache}"
export RAY_DEDUP_LOGS=0

# =========================================================== 3. env snapshot
# One-shot dump of everything that normally takes 20 min to reconstruct
# after an HPC job blows up. Keep it self-contained — no external tools.
{
  echo "==== run metadata ===="
  echo "timestamp          : $(date -Iseconds)"
  echo "hostname           : $(hostname)"
  echo "user               : ${USER:-unknown}"
  echo "pwd                : $(pwd)"
  echo "SPEC_DIAG_ROOT     : $SPEC_DIAG_ROOT"
  echo "CONDA_ENV          : $CONDA_ENV"
  echo "CUDA_VISIBLE_DEVICES: $VISIBLE"
  echo "SLURM_JOB_ID       : ${SLURM_JOB_ID:-<not-slurm>}"
  echo "SLURM_NODELIST     : ${SLURM_NODELIST:-<not-slurm>}"
  echo
  echo "==== git ===="
  (cd "$SPEC_DIAG_ROOT" && git rev-parse HEAD 2>/dev/null) || echo "(not a git repo or git missing)"
  (cd "$SPEC_DIAG_ROOT" && git status --short 2>/dev/null) || true
  echo
  echo "==== python / cuda ===="
  python --version 2>&1 || true
  which python
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())" 2>&1 || true
  nvcc --version 2>&1 | tail -n 1 || true
  echo
  echo "==== nvidia-smi ===="
  nvidia-smi || true
  echo
  echo "==== key pip packages ===="
  pip list 2>/dev/null | grep -Ei '^(verl|vllm|ray|torch|transformers|hydra-core|openai|pebble|flash-attn)\b' || true
  echo
  echo "==== env passthrough ===="
  for k in OPENAI_BASE_URL OPENAI_API_KEY SPEC_DIAG_MODEL HF_HOME LD_LIBRARY_PATH VLLM_PORT; do
    echo "$k=${!k:-<unset>}"
  done
} > "$ENV_FILE" 2>&1
echo "env snapshot  : $ENV_FILE"

# =========================================================== 4. cleanup trap
VLLM_PID=""
GPU_MON_PID=""
HEALTH_MON_PID=""
FINAL_EXIT_CODE=0

cleanup() {
  local rc="${1:-$?}"
  FINAL_EXIT_CODE="$rc"
  set +e
  # Stop monitors first so they don't race with vLLM teardown.
  for p in "$GPU_MON_PID" "$HEALTH_MON_PID"; do
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
      kill "$p" 2>/dev/null || true
    fi
  done
  if [ -n "$VLLM_PID" ] && [ "$KEEP_VLLM_ON_EXIT" != "1" ]; then
    if kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[cleanup] stopping vLLM pid=$VLLM_PID"
      kill "$VLLM_PID" 2>/dev/null || true
      for _ in 1 2 3 4 5; do
        kill -0 "$VLLM_PID" 2>/dev/null || break
        sleep 1
      done
      kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
  fi
  # Write final status file.
  {
    echo "run_dir       : $RUN_DIR"
    echo "finished      : $(date -Iseconds)"
    echo "exit_code     : $FINAL_EXIT_CODE"
    echo "keep_vllm     : $KEEP_VLLM_ON_EXIT"
    echo
    echo "---- last 80 lines of train.log ----"
    tail -n 80 "$TRAIN_LOG" 2>/dev/null || echo "(train.log not found)"
    echo
    echo "---- last 20 lines of vllm.log ----"
    tail -n 20 "$VLLM_LOG" 2>/dev/null || echo "(vllm.log not found)"
  } > "$STATUS_FILE"
  echo "[cleanup] status written to $STATUS_FILE"
}
trap 'cleanup $?' EXIT
trap 'exit 130' INT TERM

# =========================================================== 5. gpu monitor
# Background nvidia-smi sampler. Each sample has a timestamp header plus the
# --query-gpu columns we usually need for post-mortem. Interval keeps the
# file small (a few MB per hour).
gpu_monitor() {
  : > "$GPU_LOG"
  while :; do
    {
      echo "---- $(date -Iseconds) ----"
      nvidia-smi \
        --query-gpu=index,name,pstate,temperature.gpu,utilization.gpu,memory.used,memory.free \
        --format=csv,noheader 2>&1
    } >> "$GPU_LOG"
    sleep "$GPU_POLL_INTERVAL_S"
  done
}
gpu_monitor &
GPU_MON_PID=$!
echo "gpu monitor   : pid=$GPU_MON_PID interval=${GPU_POLL_INTERVAL_S}s log=$GPU_LOG"

# =========================================================== 6. vLLM (bg)
echo "---- [1/4] launching vLLM generator on GPU $GPU_VLLM ----"
(
  CUDA_VISIBLE_DEVICES="$GPU_VLLM" \
  MODEL="$GENERATOR_MODEL" \
  VLLM_PORT="$VLLM_PORT" \
  VLLM_LOG_FILE="$VLLM_LOG" \
  VLLM_PID_FILE="$RUN_DIR/vllm.pid" \
  SPEC_DIAG_ROOT="$SPEC_DIAG_ROOT" \
  CONDA_ENV="$CONDA_ENV" \
  bash "$SPEC_DIAG_ROOT/scripts/launch_vllm_gen_1xa800.sh"
) &
VLLM_PID=$!
echo "vllm pid      : $VLLM_PID log=$VLLM_LOG"

# =========================================================== 7. health wait
echo "---- [2/4] waiting for vLLM /v1/models (timeout ${VLLM_HEALTH_TIMEOUT_S}s) ----"
BASE_URL="http://127.0.0.1:$VLLM_PORT/v1"
waited=0
until curl -sf "$BASE_URL/models" >/dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM process died before becoming healthy. Tail of vllm.log:"
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

# =========================================================== 8. health beacon
# Background beacon that pings /v1/models at a fixed cadence. The training
# process doesn't care about this file; it's purely for post-mortem ("did
# vLLM drop out at minute 42?").
health_beacon() {
  : > "$VLLM_HEALTH_LOG"
  while :; do
    local now status
    now="$(date -Iseconds)"
    if curl -sf -m 5 "$BASE_URL/models" >/dev/null 2>&1; then
      status="OK"
    else
      status="FAIL"
    fi
    echo "$now  $status" >> "$VLLM_HEALTH_LOG"
    sleep "$HEALTH_POLL_INTERVAL_S"
  done
}
health_beacon &
HEALTH_MON_PID=$!
echo "health beacon : pid=$HEALTH_MON_PID interval=${HEALTH_POLL_INTERVAL_S}s log=$VLLM_HEALTH_LOG"

# =========================================================== 9. training
echo "---- [3/4] launching spec_diag GRPO training on GPUs $GPU_TRAIN0,$GPU_TRAIN1 ----"
export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_API_KEY="dummy"
export SPEC_DIAG_MODEL="generator"
# Tell the Python launcher where to dump its logging + resolved config.
export SPEC_DIAG_RUN_DIR="$RUN_DIR"

set +e
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
TRAIN_RC=${PIPESTATUS[0]}
set -e

echo "---- [4/4] training finished rc=$TRAIN_RC ----"
exit "$TRAIN_RC"
