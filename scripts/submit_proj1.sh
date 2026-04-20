#!/usr/bin/env bash
# 登录节点执行：把 proj1 1vLLM+3GRPO 任务丢到 acd_u
#   bash scripts/submit_proj1.sh
set -euo pipefail

PROJ_ROOT="/data/user/dingcao/hanyang/proj1/spec_diag"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_FILE="$SCRIPT_DIR/job_proj1_1vllm_3grpo.sbatch"

# slurm 在脚本启动前就要打开 -o/-e 指向的文件，目录必须先存在
mkdir -p "$PROJ_ROOT/slurm_logs"

JID=$(sbatch --parsable "$SBATCH_FILE")
echo "submitted: jobid=$JID"
echo "  queue : squeue -j $JID"
echo "  stdout: tail -F $PROJ_ROOT/slurm_logs/proj1_${JID}.out"
echo "  stderr: tail -F $PROJ_ROOT/slurm_logs/proj1_${JID}.err"
echo "  cancel: scancel $JID"
