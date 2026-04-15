#!/usr/bin/env bash
# 在登录节点执行：提交 spec_diag 或 vLLM 出题服务到 acd_u
set -euo pipefail

BASE="/data/user/dingcao/hanyang"
SPEC_DIAG="$BASE/spec_diag"
mkdir -p "$SPEC_DIAG/slurm_logs"

usage() {
  echo "Usage: $0 {main|vllm|both}" >&2
  echo "  main  - sbatch job_spec_diag_acd_u.sbatch (python -m spec_diag.main)" >&2
  echo "  vllm  - sbatch job_vllm_generator_acd_u.sbatch (OpenAI-compatible API)" >&2
  echo "  both  - 依次提交 vllm 与 main（按需改依赖顺序）" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

case "${1:-}" in
  main)
    sbatch "$SCRIPT_DIR/job_spec_diag_acd_u.sbatch"
    ;;
  vllm)
    sbatch "$SCRIPT_DIR/job_vllm_generator_acd_u.sbatch"
    ;;
  both)
    sbatch "$SCRIPT_DIR/job_vllm_generator_acd_u.sbatch"
    sbatch "$SCRIPT_DIR/job_spec_diag_acd_u.sbatch"
    ;;
  *)
    usage
    ;;
esac

echo "Submitted. squeue -u \"\$USER\" ; logs: $SPEC_DIAG/slurm_logs/"
