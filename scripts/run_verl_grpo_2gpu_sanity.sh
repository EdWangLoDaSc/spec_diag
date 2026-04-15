#!/usr/bin/env bash
# verl GRPO 2-GPU sanity check
# ============================
# Purpose: run verl's stock GRPO example on GSM8K with a small Qwen model to
# confirm verl + FSDP + vLLM rollout + reward manager all work on this box.
# This is NOT spec_diag — it doesn't touch any of our code. Run this first.
#
# Targets:
#   - 2x H100 80GB (tensor_parallel=1, FSDP across 2 GPUs)
#   - Qwen2.5-1.5B-Instruct by default (smallest you'd call "realistic")
#   - ~a few minutes of training, not a full run
#
# Usage:
#   bash scripts/run_verl_grpo_2gpu_sanity.sh
#
# Override model or paths:
#   MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/run_verl_grpo_2gpu_sanity.sh
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_verl_grpo_2gpu_sanity.sh
#   DATA_DIR=/data/gsm8k bash scripts/run_verl_grpo_2gpu_sanity.sh
#
# Prereqs (once):
#   1. pip install -e ../verl
#   2. pip install -e '.[dev]'
#   3. pip install 'vllm>=0.8.4' datasets pyarrow
#   4. HuggingFace access to the model (either HF token or pre-staged weights)

set -euo pipefail

# ---- paths ----
SPEC_DIAG_ROOT="${SPEC_DIAG_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VERL_DIR="${VERL_DIR:-$SPEC_DIAG_ROOT/../verl}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
LOG_DIR="${LOG_DIR:-$SPEC_DIAG_ROOT/logs/verl_sanity}"
mkdir -p "$LOG_DIR" "$DATA_DIR"

# ---- model / training knobs ----
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
if [ "$N_GPUS" -ne 2 ]; then
  echo "ERROR: this script expects exactly 2 GPUs; CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' has $N_GPUS" >&2
  exit 1
fi

# ---- sanity: verl importable ----
if ! python -c "import verl" 2>/dev/null; then
  echo "ERROR: verl is not importable. pip install -e $VERL_DIR" >&2
  exit 1
fi
if ! python -c "import vllm" 2>/dev/null; then
  echo "ERROR: vllm is not importable. pip install 'vllm>=0.8.4'" >&2
  exit 1
fi

# ---- GSM8K data prep (once) ----
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
  echo "==== preprocessing GSM8K -> $DATA_DIR ===="
  python "$VERL_DIR/examples/data_preprocess/gsm8k.py" --local_dir "$DATA_DIR"
fi

# ---- runtime env ----
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="${HF_HOME:-$SPEC_DIAG_ROOT/.hf_cache}"
mkdir -p "$HF_HOME"
# keep Ray chatty but not flood
export RAY_DEDUP_LOGS=0

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/grpo_2gpu_${TS}.log"

echo "==== verl GRPO 2-GPU sanity ===="
echo "host      : $(hostname)"
echo "gpus      : $CUDA_VISIBLE_DEVICES  (n=$N_GPUS)"
echo "model     : $MODEL"
echo "data      : $DATA_DIR"
echo "verl      : $VERL_DIR"
echo "log       : $LOG_FILE"
echo "================================"
nvidia-smi || true

# ---- launch ----
# Rationale for knobs vs verl's reference run_qwen2-7b.sh:
#   - 2 GPUs instead of 8         → n_gpus_per_node=2, rollout.tp=1
#   - Qwen2.5-1.5B instead of 7B  → smaller, fits easily, fast smoke signal
#   - train_batch_size=64         → small, 1 sanity run != full training
#   - ppo_mini_batch_size=32      → divides batch evenly
#   - micro_batch_size=8          → safe on 1.5B w/ grad ckpt
#   - rollout.n=5                 → GRPO group size (unchanged)
#   - rollout.gpu_memory_utilization=0.6 → leave room for FSDP shards
#   - total_epochs=1              → full 1 epoch ~ a few hundred updates,
#                                    but you can Ctrl-C once you see reward
#                                    curves moving. We're only asserting the
#                                    *loop* works.
#   - test_freq=10, save_freq=-1  → eval occasionally, don't write ckpts
#   - logger=[console]            → no wandb required

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_sanity' \
    trainer.experiment_name="grpo_2gpu_$(basename "$MODEL")" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    2>&1 | tee "$LOG_FILE"
