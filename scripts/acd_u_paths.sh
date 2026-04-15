#!/usr/bin/env bash
# 集群上统一根路径（登录节点 / 计算节点 source 本文件或由其派生的作业脚本引用）
# Usage: source "$(dirname "$0")/acd_u_paths.sh"   OR   由 .sbatch 内嵌相同变量

export HANYANG_BASE="${HANYANG_BASE:-/data/user/dingcao/hanyang}"
export SPEC_DIAG_ROOT="${SPEC_DIAG_ROOT:-$HANYANG_BASE/spec_diag}"
export VERL_DIR="${VERL_DIR:-$HANYANG_BASE/verl}"

# conda：用户环境 spec_diag；共享 vLLM 可用下面之一
export CONDA_SH_USER="${CONDA_SH_USER:-$HOME/.conda/etc/profile.d/conda.sh}"
export CONDA_SH_SHARE="${CONDA_SH_SHARE:-/share/anaconda3/etc/profile.d/conda.sh}"

activate_conda_env() {
  local env_path_or_name="$1"
  if [ -f "$CONDA_SH_USER" ]; then
    # shellcheck source=/dev/null
    source "$CONDA_SH_USER"
  elif [ -f "$CONDA_SH_SHARE" ]; then
    # shellcheck source=/dev/null
    source "$CONDA_SH_SHARE"
  else
    echo "ERROR: conda.sh not found. Set CONDA_SH_USER or CONDA_SH_SHARE." >&2
    return 1
  fi
  # shellcheck disable=SC1090
  conda activate "$env_path_or_name"
}
