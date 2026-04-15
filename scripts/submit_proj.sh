#!/usr/bin/env bash
# 登录节点执行：把 spec_diag 主任务丢到 acd_u
#   bash scripts/submit_proj.sh
set -euo pipefail

SPEC_DIAG_ROOT="/data/user/dingcao/hanyang/spec_diag"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_FILE="$SCRIPT_DIR/job_spec_diag_proj.sbatch"

# slurm 在脚本启动前就要打开 -o/-e 指向的文件，目录必须先存在
mkdir -p "$SPEC_DIAG_ROOT/slurm_logs"

JID=$(sbatch --parsable "$SBATCH_FILE")
echo "submitted: jobid=$JID"
echo "  queue : squeue -j $JID"
echo "  stdout: tail -F $SPEC_DIAG_ROOT/slurm_logs/spec_diag_${JID}.out"
echo "  stderr: tail -F $SPEC_DIAG_ROOT/slurm_logs/spec_diag_${JID}.err"
echo "  cancel: scancel $JID"
