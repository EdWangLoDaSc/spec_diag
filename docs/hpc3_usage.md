# HPC3（港科广 ACD 平台）使用文档

港科大广州 HPC 三期 AI 智算平台（ACD）使用总结，基于官方知识库三篇文档整理：作业提交与管理、集群队列、校内软件镜像仓。

官方门户：
- 文档：https://docs.hpc.hkust-gz.edu.cn/docs/hpc3/
- 镜像仓：https://nexus.hpc.hkust-gz.edu.cn/

---

## 1. 集群队列

使用 `sinfo -l` 查看队列与节点状态。HPC3 上用于本项目的 GPU(ACD) 队列：

| 队列 | 资源 | 优先级 | 价格 | 用户配额 | 默认时长 |
|---|---|---|---|---|---|
| `acd_u` | GPU(ACD) | 共享(低) | 低 | CPU 128 核 / GPU 16 卡 | 7 天 |
| `acd_ue` | GPU(ACD) | 独占(中) | 中 | CPU 128 核 / GPU 16 卡 | 7 天 |
| `emergency_acd` | GPU(ACD) | 紧急(高) | 高 | CPU 128 核 / GPU 16 卡 | 7 天 |

- 默认最长运行 7 天；到期前可在科大 Go 提 IT 工单申请延长（只能延长一次，总计不得超过 7 天）。
- 本仓库默认使用 `acd_u`（见 `scripts/job_spec_diag_acd_u.sbatch`）。

---

## 2. 作业提交

### 2.1 命令行直接提交

```bash
sbatch -p acd_u --input=input.sh -o output_%j.txt -e err_%j.txt \
       -n 8 --gres=gpu:1 job_script.sh
```

常用参数：

| 参数 | 含义 |
|---|---|
| `-p acd_u` | 指定队列 |
| `-o output_%j.txt` | 标准输出文件，`%j` = jobid |
| `-e err_%j.txt` | 标准错误文件 |
| `-n 8` | 申请 CPU 总核心数 |
| `--gres=gpu:1` | 申请 GPU 卡数 |
| `-w ACD1-1,ACD1-2` | 指定节点（`sinfo` 查） |
| `-x "~ACD1-1"` | 排除节点 |
| `-D /apps` | 指定作业执行路径（默认为提交路径） |

### 2.2 脚本模式（推荐）

`sbatch my_job.sh`，脚本头部用 `#SBATCH` 写死参数：

```bash
#!/bin/bash
#SBATCH -p acd_u
#SBATCH -o output_%j.txt
#SBATCH -e err_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -D /apps

echo "Job started at $(date)"
python your_script.py
echo "Job ended at $(date)"
```

### 2.3 多节点并行作业

关键 SBATCH 项：`--nodes=2 --ntasks-per-node=4 --gres=gpu:4`。必须 `module load anaconda3`、`module load cuda/12.4`，并设置 NCCL 环境变量（InfiniBand + NVLink）。标准模板（来自官方文档）：

```bash
#!/bin/bash
#SBATCH -p acd
#SBATCH --job-name=speed_test
#SBATCH -o /data/user/<user>/slurm_log/%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

module load anaconda3
module load cuda/12.4

export NCCL_SOCKET_IFNAME=vlan0.2135
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=138
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=4
export NCCL_NVLS_ENABLE=0
export NCCL_NVLS_PLUGIN=1
export NCCL_NVLS_LANES=2
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_MIN_CTAS=32
export NCCL_MAX_CTAS=128
export NCCL_IB_RETRY_CNT=7
export NCCL_MIN_NCHANNELS=64
export NCCL_MAX_NCHANNELS=256
export NCCL_NCHANNELS_PER_NET_PEER=32
export NCCL_BUFFSIZE=33554432
export NCCL_LL_BUFFSIZE=33554432
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_P2P_LEVEL=nvl
export NCCL_ALGO=nvlstree,ring
export NCCL_LL_MAX_NCHANNELS=4
export NCCL_CROSS_NIC=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_SHM_DISABLE=0
export NCCL_COLLNET_ENABLE=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
export NCCL_IB_TIMEOUT=3600

python your_script.py
```

### 2.4 数组作业

```bash
#SBATCH --array=1-10           # 10 个子作业；也可 1,3,5 或 1-10:2
#SBATCH -o output_%A_%a.txt    # %A=主 jobid, %a=子索引
PARAM=$SLURM_ARRAY_TASK_ID
python your_script.py $PARAM
```

### 2.5 交互式作业

```bash
srun -p acd_u -n 4 --mem=8G --gres=gpu:1 --time=01:00:00 --pty bash
```

---

## 3. 作业查看与管理

| 操作 | 命令 |
|---|---|
| 当前作业 | `squeue -u <user> [-t PENDING,RUNNING,SUSPENDED]` |
| 历史作业 | `sacct -u <user> [--array]` |
| 作业详情 | `scontrol show job <jobid>` |
| 数组作业详情 | `scontrol show job <jobid>_<task_id>` |
| 挂起 / 恢复 | `scontrol suspend <jobid>` / `scontrol resume <jobid>` |
| 终止 | `scancel <jobid>` |

Web 端：HPC 三期门户 → 作业管理 → 作业列表 / 历史作业（显示结束 7 天内记录）。作业详情页含 输出 / 数据 / 监控 / 详情 / 性能 五个 tab，运行中点"性能"可看 GPU / CPU / 内存使用率。

---

## 4. 校内软件镜像仓

HPC3 节点出口受限，**必须用校内镜像仓**装依赖。前提：`ping harbor.internal.com` 能通；不通时手工写 `/etc/hosts`（IP 在登录节点查）。

支持 Maven / npm / docker / conda / PyPI / yum。

### 4.1 PyPI

全局（venv 内）：

```bash
pip config set global.index-url http://harbor.internal.com:8081/repository/pypi-hkust/simple
pip config set install.trusted-host harbor.internal.com
```

临时：

```bash
pip install <pkg> \
  --index-url http://harbor.internal.com:8081/repository/pypi-hkust/simple \
  --trusted-host harbor.internal.com
```

查版本：`pip index versions <pkg> --index-url ... --trusted-host harbor.internal.com`。

### 4.2 Conda

```bash
conda config --remove channels defaults
conda config --add channels http://harbor.internal.com:8081/repository/conda-hkust/main
conda config --add channels http://harbor.internal.com:8081/repository/conda-hkust/free
conda config --add channels http://harbor.internal.com:8081/repository/conda-hkust/msys2
conda config --add channels http://harbor.internal.com:8081/repository/conda-hkust/pro
conda config --add channels http://harbor.internal.com:8081/repository/conda-hkust/r
```

`conda search <pkg>` 查可装版本。

---

## 5. 本项目约定

- 提交入口：`scripts/submit_acd_u.sh`，作业脚本 `scripts/job_spec_diag_acd_u.sbatch`、`scripts/job_vllm_generator_acd_u.sbatch`，路径 `scripts/acd_u_paths.sh`。
- 默认队列 `acd_u`，conda 环境 `/data/user/dingcao/.conda/envs/spec_diag`。
- 进入作业前 `module load anaconda3 && module load cuda/12.4`。
- 日志统一落到 `/data/user/dingcao/hanyang/spec_diag/slurm_logs/%j.{out,err}`。
