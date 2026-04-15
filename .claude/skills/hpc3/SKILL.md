---
name: hpc3
description: 港科广 HPC 三期 (ACD) 平台使用助手。TRIGGER when 用户在本项目里涉及 Slurm 作业提交/查看/终止 (sbatch, srun, squeue, sacct, scancel, scontrol)、队列选择 (acd_u / acd_ue / emergency_acd)、多机 NCCL/IB 环境变量、数组作业、或在受限网络下用校内镜像仓 (harbor.internal.com, nexus.hpc.hkust-gz.edu.cn) 装 pip/conda 依赖。
---

# HPC3 (港科广 ACD) 使用技能

完整文档：`docs/hpc3_usage.md`（先读它，再按下面清单操作）。

## 快速决策

1. **要跑作业？** → 写 `#SBATCH` 脚本，用 `sbatch` 提交，不要用命令行 flag 堆参数。模板见下。
2. **要装依赖？** → 一定用校内镜像仓 `harbor.internal.com`，不要直连公网 PyPI / Anaconda。
3. **选队列** → 日常 `acd_u`；需要独占 `acd_ue`；紧急 `emergency_acd`。默认上限 7 天，到期前在科大 Go 开 IT 工单延期（仅一次）。
4. **多机训练** → 必须 `module load anaconda3 cuda/12.4` 并完整导出 NCCL_* 变量（见 `docs/hpc3_usage.md` §2.3），否则 IB/NVLink 不走。

## 本项目已有脚本（优先复用，不要另起炉灶）

- `scripts/submit_acd_u.sh` — 提交入口
- `scripts/job_spec_diag_acd_u.sbatch` — spec_diag 训练作业
- `scripts/job_vllm_generator_acd_u.sbatch` — vLLM generator 作业
- `scripts/acd_u_paths.sh` — 统一路径/环境
- 默认 conda env: `/data/user/dingcao/.conda/envs/spec_diag`
- 日志目录: `/data/user/dingcao/hanyang/spec_diag/slurm_logs/`

新作业优先改/复制已有 sbatch，不要重造。

## 单卡 sbatch 模板

```bash
#!/bin/bash
#SBATCH -p acd_u
#SBATCH -J <job_name>
#SBATCH -o /data/user/dingcao/hanyang/spec_diag/slurm_logs/%j.out
#SBATCH -e /data/user/dingcao/hanyang/spec_diag/slurm_logs/%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -D /data/user/dingcao/hanyang/spec_diag
set -euo pipefail
module load anaconda3 || true
module load cuda/12.4 || true
source "$HOME/.conda/etc/profile.d/conda.sh" 2>/dev/null || \
  source /share/anaconda3/etc/profile.d/conda.sh
conda activate /data/user/dingcao/.conda/envs/spec_diag
python -m spec_diag.main
```

## 多机模板要点

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
```
然后完整粘贴 `docs/hpc3_usage.md` §2.3 里的 NCCL_* 环境变量块（`NCCL_SOCKET_IFNAME=vlan0.2135`、`NCCL_IB_HCA=mlx5_0,...`、`NCCL_IB_GID_INDEX=3` 等）。不要自己简化。

## 数组作业

```bash
#SBATCH --array=1-10           # 或 1,3,5 / 1-10:2
#SBATCH -o output_%A_%a.txt    # %A=主 jobid, %a=子索引
PARAM=$SLURM_ARRAY_TASK_ID
```

## 交互式

```bash
srun -p acd_u -n 4 --mem=8G --gres=gpu:1 --time=01:00:00 --pty bash
```

## 管理命令速查

| 目的 | 命令 |
|---|---|
| 看自己排队/运行中 | `squeue -u $USER` |
| 历史作业 | `sacct -u $USER [--array]` |
| 作业详情 | `scontrol show job <jobid>` |
| 数组子作业 | `scontrol show job <jobid>_<task_id>` |
| 挂起 / 恢复 | `scontrol suspend <jobid>` / `scontrol resume <jobid>` |
| 取消 | `scancel <jobid>` |
| 队列/节点 | `sinfo -l` |

## 校内镜像仓（装依赖必用）

前置：`ping harbor.internal.com` 必须通，否则登录节点查 IP 加 hosts。

**PyPI 全局配置**：
```bash
pip config set global.index-url http://harbor.internal.com:8081/repository/pypi-hkust/simple
pip config set install.trusted-host harbor.internal.com
```

**PyPI 临时**：
```bash
pip install <pkg> \
  --index-url http://harbor.internal.com:8081/repository/pypi-hkust/simple \
  --trusted-host harbor.internal.com
```

**Conda 通道**（先 `conda config --remove channels defaults`）：
```
http://harbor.internal.com:8081/repository/conda-hkust/{main,free,msys2,pro,r}
```

镜像仓 Web 门户: https://nexus.hpc.hkust-gz.edu.cn/ 。支持 Maven / npm / docker / conda / PyPI / yum。

## 注意事项

- 作业默认最长 7 天；延期一次、总计不得 > 7 天，走科大 Go IT 工单。
- Web 历史作业只保留结束后 7 天内的记录，跑完要拿数据就及时下载。
- `-D <path>` 决定作业工作目录，默认为提交目录——本项目统一用 `/data/user/dingcao/hanyang/spec_diag`。
- 所有路径尽量走 `/data/user/...`，不要写家目录（配额小）。
