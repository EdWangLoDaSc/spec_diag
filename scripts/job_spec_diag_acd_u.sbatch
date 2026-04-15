#!/bin/bash
#SBATCH -p acd_u
#SBATCH -J spec_diag
#SBATCH -o /data/user/dingcao/hanyang/spec_diag/slurm_logs/%j.out
#SBATCH -e /data/user/dingcao/hanyang/spec_diag/slurm_logs/%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -D /data/user/dingcao/hanyang/spec_diag

set -euo pipefail

HANYANG_BASE="/data/user/dingcao/hanyang"
SPEC_DIAG_ROOT="$HANYANG_BASE/spec_diag"
VERL_DIR="$HANYANG_BASE/verl"
export VERL_DIR

mkdir -p "$SPEC_DIAG_ROOT/slurm_logs"

# CUDA（按集群 module 名自行改）
if command -v module >/dev/null 2>&1; then
  module load anaconda3 2>/dev/null || true
  module load cuda/12.4 2>/dev/null || true
fi

# conda：spec_diag 用户环境
if [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/.conda/etc/profile.d/conda.sh"
elif [ -f /share/anaconda3/etc/profile.d/conda.sh ]; then
  # shellcheck source=/dev/null
  source /share/anaconda3/etc/profile.d/conda.sh
fi
conda activate /data/user/dingcao/.conda/envs/spec_diag

cd "$SPEC_DIAG_ROOT"
echo "PWD=$(pwd) HOST=$(hostname) JOB_ID=${SLURM_JOB_ID:-}"
echo "VERL_DIR=$VERL_DIR"

# 入口：可改成训练脚本 / verl 示例
python -m spec_diag.main
