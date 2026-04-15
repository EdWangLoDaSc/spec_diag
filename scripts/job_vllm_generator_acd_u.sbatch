#!/bin/bash
# 出题侧：vLLM 托管 generator.yaml 中的模型（默认 Qwen2.5-7B-Instruct）
# 与训练作业分开提交，避免争用同一张卡时可改分区或单卡独占。

#SBATCH -p acd_u
#SBATCH -J vllm_gen7b
#SBATCH -o /data/user/dingcao/hanyang/spec_diag/slurm_logs/vllm_%j.out
#SBATCH -e /data/user/dingcao/hanyang/spec_diag/slurm_logs/vllm_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -D /data/user/dingcao/hanyang/spec_diag

set -euo pipefail

SPEC_DIAG_ROOT="/data/user/dingcao/hanyang/spec_diag"
mkdir -p "$SPEC_DIAG_ROOT/slurm_logs"

if command -v module >/dev/null 2>&1; then
  module load anaconda3 2>/dev/null || true
  module load cuda/12.4 2>/dev/null || true
fi

# 共享机上的 vllm 环境（与 conda env list 一致；若无则改为 conda activate vllm017 等）
if [ -f /share/anaconda3/etc/profile.d/conda.sh ]; then
  # shellcheck source=/dev/null
  source /share/anaconda3/etc/profile.d/conda.sh
  conda activate /share/anaconda3/envs/vllm
elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/.conda/etc/profile.d/conda.sh"
  conda activate vllm
else
  echo "ERROR: cannot find conda.sh" >&2
  exit 1
fi

# 端口可按平台策略修改；客户端填 OPENAI_BASE_URL=http://<节点>:8000/v1
export VLLM_PORT="${VLLM_PORT:-8000}"
MODEL_NAME="${GENERATOR_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

echo "Starting vLLM model=$MODEL_NAME port=$VLLM_PORT on $(hostname)"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --trust-remote-code
