#!/usr/bin/env bash
# Run spec_diag GRPO training on exactly 3 GPUs.
# Assumes vLLM endpoint is already running (default: http://127.0.0.1:8000/v1).

set -euo pipefail

SPEC_DIAG_ROOT="${SPEC_DIAG_ROOT:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-spec_diag}"
STUDENT_MODEL="${STUDENT_MODEL:-/home/apulis-dev/models/Qwen25-7B-Instruct/V0/code/Qwen2.5-7B-Instruct}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
SPEC_DIAG_MODEL="${SPEC_DIAG_MODEL:-generator}"
TRAIN_OVERRIDES="${TRAIN_OVERRIDES:-}"
RUN_DIR_BASE="${RUN_DIR_BASE:-$SPEC_DIAG_ROOT/logs/grpo_split}"
ACTOR_MB_PER_GPU="${ACTOR_MB_PER_GPU:-1}"
ROLLOUT_MB_PER_GPU="${ROLLOUT_MB_PER_GPU:-1}"
REF_MB_PER_GPU="${REF_MB_PER_GPU:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"
ACTOR_MINI_BATCH_SIZE="${ACTOR_MINI_BATCH_SIZE:-24}"
ROLLOUT_UPDATE_BUCKET_MB="${ROLLOUT_UPDATE_BUCKET_MB:-3072}"
CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-pkg://spec_diag.rewards.spec_diag_score}"

mkdir -p "$RUN_DIR_BASE"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RUN_DIR_BASE/run_${TS}"
mkdir -p "$RUN_DIR"
TRAIN_LOG="$RUN_DIR/train.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
N_VIS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
if [ "$N_VIS" -ne 3 ]; then
  echo "ERROR: GRPO script needs exactly 3 visible GPUs; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

if ! curl -sf "${OPENAI_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: vLLM endpoint not healthy at ${OPENAI_BASE_URL}" >&2
  echo "Start it first with scripts/run_vllm_1gpu_h100.sh" >&2
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

echo "==== GRPO split launcher (3 GPU) ===="
echo "host         : $(hostname)"
echo "student model: $STUDENT_MODEL"
echo "base url     : $OPENAI_BASE_URL"
echo "gpus         : $CUDA_VISIBLE_DEVICES"
echo "run dir      : $RUN_DIR"
echo "hydra extra  : file://$VERL_HYDRA_CONFIG_DIR"
echo "actor mb/gpu : $ACTOR_MB_PER_GPU"
echo "rollout mb   : $ROLLOUT_MB_PER_GPU"
echo "ref mb       : $REF_MB_PER_GPU"
echo "train batch  : $TRAIN_BATCH_SIZE"
echo "actor mini   : $ACTOR_MINI_BATCH_SIZE"
echo "rollout bucket mb: $ROLLOUT_UPDATE_BUCKET_MB"
echo "reward fn    : $CUSTOM_REWARD_PATH::compute_score"
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
  trainer.n_gpus_per_node=3 \
  trainer.nnodes=1 \
  trainer.project_name='spec_diag' \
  trainer.experiment_name="split_1vllm_3grpo_${TS}" \
  trainer.logger='["console","tensorboard"]' \
  '+actor_rollout_ref.model.override_config.attn_implementation=eager' \
  '+critic.model.override_config.attn_implementation=eager' \
  ${TRAIN_OVERRIDES} \
  2>&1 | tee "$TRAIN_LOG"
RC=${PIPESTATUS[0]}
set -e

echo "GRPO exit code: $RC"
echo "train log: $TRAIN_LOG"
exit "$RC"
