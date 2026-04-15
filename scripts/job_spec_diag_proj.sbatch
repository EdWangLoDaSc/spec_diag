#!/bin/bash
# spec_diag 主任务 (acd_u) —— 单节点单卡，python -m spec_diag.main
# 提交：bash scripts/submit_proj.sh    或    sbatch scripts/job_spec_diag_proj.sbatch

#SBATCH -p acd_u
#SBATCH -J spec_diag
#SBATCH -o /data/user/dingcao/hanyang/spec_diag/slurm_logs/%x_%j.out
#SBATCH -e /data/user/dingcao/hanyang/spec_diag/slurm_logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH -D /data/user/dingcao/hanyang/spec_diag

set -euo pipefail

# ---- 路径 ----
HANYANG_BASE="/data/user/dingcao/hanyang"
SPEC_DIAG_ROOT="$HANYANG_BASE/spec_diag"
VERL_DIR="$HANYANG_BASE/verl"
CONDA_ENV="/data/user/dingcao/.conda/envs/spec_diag"
export HANYANG_BASE SPEC_DIAG_ROOT VERL_DIR

mkdir -p "$SPEC_DIAG_ROOT/slurm_logs"

# ---- module ----
if command -v module >/dev/null 2>&1; then
  module load anaconda3 2>/dev/null || true
  module load cuda/12.4 2>/dev/null || true
fi

# ---- conda 激活（set +u 避开激活脚本里的未定义变量）----
set +u
if [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/.conda/etc/profile.d/conda.sh"
elif [ -f /share/anaconda3/etc/profile.d/conda.sh ]; then
  # shellcheck source=/dev/null
  source /share/anaconda3/etc/profile.d/conda.sh
else
  echo "ERROR: conda.sh not found" >&2
  exit 1
fi
conda activate "$CONDA_ENV"
set -u

# ---- python / 日志 ----
export PYTHONPATH="$SPEC_DIAG_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# 如果后续要接 vLLM 出题服务：export OPENAI_BASE_URL="http://<gen_node>:8000/v1"

# ---- 调试信息 ----
echo "==== ENV ===="
echo "JOB_ID=${SLURM_JOB_ID:-}  HOST=$(hostname)  PWD=$(pwd)"
echo "CONDA_ENV=$CONDA_ENV"
python -c "import sys; print('python:', sys.executable); print('ver:', sys.version)"
nvidia-smi || true
echo "=============="

# ---- 主程序 ----
cd "$SPEC_DIAG_ROOT"
echo "[$(date)] start spec_diag.main"
python -m spec_diag.main
echo "[$(date)] done"
