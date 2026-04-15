#!/usr/bin/env python3
"""
HPC Slurm Helper for HKUST(GZ) HPC+AI (mgmt-3 / ACD platform)
Helps with:
1. GPU (卡) 查询 - sinfo, squeue, nvidia-smi
2. 分配 2 张卡 - generate sbatch for --gres=gpu:2 , interactive srun
3. 本地文件传入 - scp/rsync commands from local machine to cluster

Usage:
  # On cluster for queries and job submission
  python hpc_slurm_helper.py query
  python hpc_slurm_helper.py allocate --gpus 2 --job-name myjob
  python hpc_slurm_helper.py interactive --gpus 2

  # On local machine for file transfer
  python hpc_slurm_helper.py transfer --local-path ./my_data --remote-path /data/user/dingcao/hanyang/spec_diag/data
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import os


def run_command(cmd: str, capture: bool = True) -> str:
    """Run shell command safely."""
    try:
        if capture:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=False
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr, file=sys.stderr)
            return result.stdout or result.stderr or ""
        else:
            return subprocess.run(cmd, shell=True, check=False).returncode
    except Exception as e:
        print(f"Command failed: {e}", file=sys.stderr)
        return str(e)


def query_gpus() -> None:
    """Query GPUs/cards on the HPC cluster."""
    print("=== HPC GPU (卡) 查询工具 ===")
    print("当前登录节点: mgmt-3 (负载较高: swap 100%, load ~20)")
    print("\n1. 查看可用分区、节点和GPU资源 (推荐):")
    print("   sinfo -o '%P %n %G %R %E' -N | head -30")
    print("   sinfo -l  # 详细队列状态")
    
    print("\n2. 查看你的作业 (PENDING/RUNNING):")
    print("   squeue -u $USER -o '%.8u %.12j %.8T %.10b %.6C %.8m %D %R'")
    print("   squeue -u $USER -t RUNNING  # 只看运行中的")
    
    print("\n3. 历史作业:")
    print("   sacct -u $USER --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End")
    print("   sacct -u $USER --array")
    
    print("\n4. 作业详情 (替换 <JOBID>):")
    print("   scontrol show job <JOBID>")
    print("   scontrol show job <JOBID>  # 查看 Reason if PENDING")
    
    print("\n5. 在已分配作业内查看实际GPU:")
    print("   nvidia-smi")
    print("   nvidia-smi -L  # 列出GPU")
    print("   watch -n 1 nvidia-smi  # 实时监控")
    
    print("\nRunning actual queries now (if on cluster):")
    print("\n--- sinfo GPU status ---")
    run_command("sinfo -o '%P %n %G %R' -N | head -20")
    print("\n--- Your jobs ---")
    run_command("squeue -u $(whoami) -o '%.10u %.12j %.10T %.12b %C %m' 2>/dev/null || echo 'No squeue or not on cluster'")
    
    print("\nNote: 当前系统有12个zombie processes, load高, 优先用 acd_u 队列, 避免长时间占用.")


def generate_2gpu_sbatch(job_name: str = "spec_diag_2gpu", 
                        output_path: str = None,
                        cpus: int = 16,
                        mem: str = "64G",
                        time: str = "7-00:00:00",
                        work_dir: str = "/data/user/dingcao/hanyang/spec_diag") -> str:
    """Generate Slurm batch script for allocating 2 GPUs."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = f"job_{job_name}_{timestamp}.sbatch"
    
    content = f"""#!/bin/bash
#SBATCH -p acd_u
#SBATCH -J {job_name}
#SBATCH -o {work_dir}/slurm_logs/%j.out
#SBATCH -e {work_dir}/slurm_logs/%j.err
#SBATCH -n {cpus}
#SBATCH --gres=gpu:2
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH -D {work_dir}

set -euo pipefail

echo "=== Job started at $(date) on $(hostname) ==="
echo "JOB_ID=${{SLURM_JOB_ID:-unknown}}"
echo "GPUS=2 (requested via --gres=gpu:2)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "nvidia-smi not available yet"

HANYANG_BASE="/data/user/dingcao/hanyang"
SPEC_DIAG_ROOT="$HANYANG_BASE/spec_diag"
VERL_DIR="$SPEC_DIAG_ROOT/verl"
export VERL_DIR

mkdir -p "$SPEC_DIAG_ROOT/slurm_logs"

# Load modules
if command -v module >/dev/null 2>&1; then
  module load anaconda3 2>/dev/null || true
  module load cuda/12.4 2>/dev/null || true
fi

# Activate conda env (adjust if needed)
if [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
  source "$HOME/.conda/etc/profile.d/conda.sh"
  conda activate /data/user/dingcao/.conda/envs/spec_diag
elif [ -f /share/anaconda3/etc/profile.d/conda.sh ]; then
  source /share/anaconda3/etc/profile.d/conda.sh
  conda activate spec_diag
fi

cd "$SPEC_DIAG_ROOT"
echo "PWD=$(pwd)  HOST=$(hostname)  JOB_ID=${{SLURM_JOB_ID}}"
echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv

# Your main command here - replace with your training/inference script
# Example for 2 GPU training or vLLM with tensor parallel:
# CUDA_VISIBLE_DEVICES=0,1 python -m spec_diag.main --gpus 2
# or for vLLM TP=2: see launch_vllm_gen_2xh100.sh

python -m spec_diag.main

echo "=== Job finished at $(date) ==="
"""
    
    Path(output_path).write_text(content)
    print(f"Generated 2-GPU Slurm script: {output_path}")
    print(f"   - Partition: acd_u")
    print(f"   - GPUs: 2 (--gres=gpu:2)")
    print(f"   - CPUs: {cpus}, Mem: {mem}")
    print(f"   - Submit with: sbatch {output_path}")
    print(f"   - Monitor: squeue -u $USER ; tail -f {work_dir}/slurm_logs/*.out")
    return output_path


def interactive_2gpu() -> None:
    """Print command for interactive session with 2 GPUs."""
    cmd = (
        "srun -p acd_u --gres=gpu:2 -n 16 --mem=64G --time=04:00:00 "
        "--pty bash -i"
    )
    print("=== 交互式分配 2 张卡 (推荐先测试) ===")
    print(f"Run this command on login node:")
    print(f"  {cmd}")
    print("\nInside the interactive session:")
    print("  - nvidia-smi to confirm 2 GPUs")
    print("  - module load anaconda3; module load cuda/12.4")
    print("  - conda activate your_env")
    print("  - Run your Python code with CUDA_VISIBLE_DEVICES=0,1 ...")
    print("\nNote: --time limits the session. Use shorter time for testing (e.g. 01:00:00).")
    print("After testing, convert to .sbatch script using this helper.")


def file_transfer_help(local_path: str = None, remote_path: str = None, username: str = "dingcao") -> None:
    """Provide commands for transferring local files to the HPC cluster."""
    print("=== 本地文件传入到HPC集群 (从你的本地Mac/Windows运行) ===")
    print("注意: 在**本地终端** (非登录节点) 执行以下命令")
    print(f"Cluster login node: mgmt-3 (IP usually 10.120.18.243 or use hostname)")
    
    if local_path is None:
        local_path = "./your_local_folder_or_file"
    if remote_path is None:
        remote_path = "/data/user/dingcao/hanyang/spec_diag/"
    
    host = "mgmt-3"  # or full hostname if needed: mgmt-3.hpc.ust.hk or check
    
    print("\n1. 使用 scp (简单单文件/文件夹):")
    print(f"   scp -r {local_path} {username}@{host}:{remote_path}")
    print("   # 示例: scp -r ./models/ dingcao@mgmt-3:/data/user/dingcao/hanyang/spec_diag/models/")
    
    print("\n2. 使用 rsync (推荐, 更快, 支持断点续传, 增量同步):")
    print(f"   rsync -avz --progress {local_path}/ {username}@{host}:{remote_path}")
    print("   # -a: archive, -v: verbose, -z: compress, --progress: show progress")
    print(f"   # 示例: rsync -avz --progress ./spec_diag/ dingcao@mgmt-3:/data/user/dingcao/hanyang/")
    
    print("\n3. 从集群拉取文件 (反向):")
    print(f"   rsync -avz {username}@{host}:{remote_path} ./local_backup/")
    
    print("\n4. 如果使用 VS Code Remote-SSH:")
    print("   - 安装 Remote - SSH 扩展")
    print("   - 配置 ~/.ssh/config with Host mgmt-3")
    print("   - 直接在 VSCode 中打开远程文件夹, 文件自动同步")
    
    print("\nTips:")
    print("- 大文件/模型建议使用 rsync")
    print("- 先确保本地有SSH key (ssh-copy-id) 以避免重复输入密码")
    print("- 目标路径建议放在 /data/user/dingcao/hanyang/spec_diag/ 下")
    print("- 传输后在集群上: ls -lh to verify, then sbatch your_job.sbatch")


def main():
    parser = argparse.ArgumentParser(
        description="HPC Slurm GPU管理助手 (Always respond in Python)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", 
        choices=["query", "allocate", "interactive", "transfer", "all"],
        help="Command to run: query GPUs, allocate 2 GPUs, interactive, file transfer"
    )
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs (default 2)")
    parser.add_argument("--job-name", default="spec_diag_2gpu", help="Job name")
    parser.add_argument("--local-path", default="./data", help="Local path for transfer")
    parser.add_argument("--remote-path", default="/data/user/dingcao/hanyang/spec_diag/", 
                       help="Remote path on cluster")
    parser.add_argument("--output", default=None, help="Output sbatch filename")
    
    args = parser.parse_args()
    
    print("HPC Slurm Helper (Python) - HKUST(GZ) ACD Platform")
    print("=" * 60)
    print(f"User: dingcao | Node: mgmt-3 | Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60 + "\n")
    
    if args.command in ["query", "all"]:
        query_gpus()
        print("\n" + "="*60 + "\n")
    
    if args.command in ["allocate", "all"]:
        if args.gpus == 2:
            generate_2gpu_sbatch(
                job_name=args.job_name,
                output_path=args.output,
                cpus=16 if args.gpus == 2 else 8
            )
        else:
            print(f"Note: Generated for {args.gpus} GPUs. Adjust --gres=gpu:{args.gpus}")
            generate_2gpu_sbatch(job_name=args.job_name, output_path=args.output)
        print("\n" + "="*60 + "\n")
    
    if args.command in ["interactive", "all"]:
        interactive_2gpu()
        print("\n" + "="*60 + "\n")
    
    if args.command in ["transfer", "all"]:
        file_transfer_help(args.local_path, args.remote_path)
        print("\n" + "="*60)
    
    print("\n总结:")
    print("1. 查询卡: python hpc_slurm_helper.py query")
    print("2. 分配2张卡: python hpc_slurm_helper.py allocate --gpus 2")
    print("3. 交互式2卡: python hpc_slurm_helper.py interactive")
    print("4. 文件传入: 在**本地**运行 python hpc_slurm_helper.py transfer")
    print("5. 提交: sbatch your_generated_job.sbatch")
    print("6. 监控: squeue -u dingcao ; scancel <JOBID> if needed")
    print("\nSee also: spec_diag/docs/hpc3_usage.md and scripts/*.sh")
    print("Based on hpc-acd-slurm-jobs skill and existing project setup.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No command provided. Showing help and running 'all' demo:\n")
        main()  # will show parser help
        sys.argv = ["hpc_slurm_helper.py", "all"]
        main()
    else:
        main()
