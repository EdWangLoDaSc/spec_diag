#!/usr/bin/env bash
# Run spec_diag GRPO training on exactly 4 GPUs (GPU 1,2,3,4 by default).
#
# Hardware topology (from nvidia-smi topo -m):
#   - 5 GPUs with full NVLink mesh (NV18 between every pair)
#   - GPU 0/1/2 → NUMA 0 (CPU 0-47)
#   - GPU 3/4   → NUMA 1 (CPU 48-95)
#
# The 3-GPU script (0,1,2) stays entirely in NUMA 0 → works.
# Any 4-GPU selection crosses NUMA → NCCL bootstrap / socket listen/connect
# hangs across NUMA boundary (even though NVLink is fine).
#
# Workaround (added below):
#   NCCL_SHM_DISABLE=1   # Disable cross-NUMA shared memory (primary fix)
#   NCCL_P2P_LEVEL=NVL   # Force P2P over NVLink only
#   NCCL_IB_DISABLE=1    # No IB on single node
#
# Recommended launch order:
#   1. bash scripts/step_1_vllm.sh          # vLLM on GPU 0
#   2. bash scripts/step2_grpo_4gpu.sh      # GRPO on 1-4

set -euo pipefail

SPEC_DIAG_ROOT="${SPEC_DIAG_ROOT:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-spec_diag}"
STUDENT_MODEL="${STUDENT_MODEL:-/data/user/dingcao/hanyang/proj1/models/Qwen25-7B-Instruct/V0/code/Qwen2.5-7B-Instruct}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
SPEC_DIAG_MODEL="${SPEC_DIAG_MODEL:-generator}"
TRAIN_OVERRIDES="${TRAIN_OVERRIDES:-}"
RUN_DIR_BASE="${RUN_DIR_BASE:-$SPEC_DIAG_ROOT/logs/grpo_split}"

# 4-GPU tuned defaults
ACTOR_MB_PER_GPU="${ACTOR_MB_PER_GPU:-2}"
ROLLOUT_MB_PER_GPU="${ROLLOUT_MB_PER_GPU:-2}"
REF_MB_PER_GPU="${REF_MB_PER_GPU:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ACTOR_MINI_BATCH_SIZE="${ACTOR_MINI_BATCH_SIZE:-32}"
ROLLOUT_UPDATE_BUCKET_MB="${ROLLOUT_UPDATE_BUCKET_MB:-4096}"
CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-pkg://spec_diag.rewards.spec_diag_score}"

mkdir -p "$RUN_DIR_BASE"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RUN_DIR_BASE/run_${TS}"
mkdir -p "$RUN_DIR"
TRAIN_LOG="$RUN_DIR/train.log"

# ==================== GPU & NCCL Configuration ====================
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"

# NCCL fixes for cross-NUMA (GPU1-2 on NUMA0, GPU3-4 on NUMA1)
export NCCL_SHM_DISABLE=1          # Primary fix: disable cross-NUMA shm
export NCCL_P2P_LEVEL=NVL          # Force NVLink P2P (all pairs have NV18)
export NCCL_IB_DISABLE=1           # No IB on single node
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN             # Change to INFO if you want full bootstrap logs

# Ray / OpenMP performance settings
export RAY_DEDUP_LOGS=0
export OMP_NUM_THREADS=8

N_VIS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
if [ "$N_VIS" -ne 4 ]; then
  echo "ERROR: GRPO script needs exactly 4 visible GPUs; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

# CRITICAL: Ray hangs at init_workers() if torch cannot see exactly 4 GPUs.
_N_CUDA="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
if [ "$_N_CUDA" -ne 4 ]; then
  echo "ERROR: torch.cuda.device_count()=${_N_CUDA}, but we requested 4 GPUs." >&2
  echo "       CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  echo "       Make sure vLLM is running on GPU 0." >&2
  exit 2
fi
unset _N_CUDA

if ! curl -sf "${OPENAI_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: vLLM endpoint not healthy at ${OPENAI_BASE_URL}" >&2
  echo "Start it first with: bash scripts/step_1_vllm.sh" >&2
  exit 3
fi

cd "$SPEC_DIAG_ROOT"

export OPENAI_BASE_URL OPENAI_API_KEY SPEC_DIAG_MODEL
export SPEC_DIAG_RUN_DIR="$RUN_DIR"
export PYTHONUNBUFFERED=1

# Auto-detect verl hydra config dir from the installed package.
if [ -z "${VERL_HYDRA_CONFIG_DIR:-}" ]; then
  VERL_HYDRA_CONFIG_DIR="$(
    python -c 'import pathlib, verl; print(pathlib.Path(verl.__file__).resolve().parent / "trainer" / "config")' 2>/dev/null || true
  )"
fi
VERL_HYDRA_CONFIG_DIR="${VERL_HYDRA_CONFIG_DIR:-$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))' 2>/dev/null)/verl/trainer/config}"

if [ ! -d "$VERL_HYDRA_CONFIG_DIR" ]; then
  echo "ERROR: verl hydra config dir not found: $VERL_HYDRA_CONFIG_DIR" >&2
  exit 4
fi

if [ "$ACTOR_MINI_BATCH_SIZE" -gt "$TRAIN_BATCH_SIZE" ]; then
  echo "ERROR: ACTOR_MINI_BATCH_SIZE ($ACTOR_MINI_BATCH_SIZE) must be <= TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE)" >&2
  exit 5
fi

echo "==== GRPO split launcher (4 GPU - NUMA aware) ===="
echo "host         : $(hostname)"
echo "student model: $STUDENT_MODEL"
echo "base url     : $OPENAI_BASE_URL"
echo "gpus         : $CUDA_VISIBLE_DEVICES"
echo "run dir      : $RUN_DIR"
echo "NCCL_SHM_DISABLE   = $NCCL_SHM_DISABLE   (cross-NUMA fix)"
echo "NCCL_P2P_LEVEL     = $NCCL_P2P_LEVEL"
echo "NCCL_IB_DISABLE    = $NCCL_IB_DISABLE"
echo "torch.cuda.device_count() = $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "NUMA topology    : GPU0/1/2(NUMA0) + GPU3/4(NUMA1) with full NV18 mesh"
echo "====================================="
nvidia-smi || true

set +e
python -m spec_diag.train \
  "hydra.searchpath=[file://$VERL_HYDRA_CONFIG_DIR]" \
  actor_rollout_ref.model.path="$STUDENT_MODEL" \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_MB_PER_GPU" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="$ROLLOUT_UPDATE_BUCKET_MB" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_MB_PER_GPU" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_MB_PER_GPU" \
  reward.custom_reward_function.path="$CUSTOM_REWARD_PATH" \
  reward.custom_reward_function.name=compute_score \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$ACTOR_MINI_BATCH_SIZE" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.project_name='spec_diag' \
  trainer.experiment_name="split_1vllm_4grpo_${TS}" \
  trainer.logger='["console","tensorboard"]' \
  trainer.test_freq=20 \
  '+actor_rollout_ref.model.override_config.attn_implementation=eager' \
  '+critic.model.override_config.attn_implementation=eager' \
  '+ray_kwargs.ray_init.log_to_driver=True' \
  ${TRAIN_OVERRIDES} \
  2>&1 | tee "$TRAIN_LOG"
RC=${PIPESTATUS[0]}
set -e

echo "GRPO exit code: $RC"
echo "train log: $TRAIN_LOG"
echo "Full detailed log: $TRAIN_LOG"
exit "$RC"
