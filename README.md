# spec_diag

Dynamic curriculum framework: a frozen LLM **Generator** (ReAct + memory) feeds
code-reasoning tasks to a GRPO-trained **Student**, built on top of
[verl](https://github.com/verl-project/verl.git).

Full design: `../idea-iter/experiment_plan.md`.
HPC3 (港科广 ACD) cheat sheet: `docs/hpc3_usage.md`.

---

## What's working right now (Phase 0)

This is the state as of the latest commit. **Read this before running
anything** so you know what to expect.

| Component | Status | Notes |
|---|---|---|
| `CodeExecutor` (Python domain) | ✅ real | wraps `PythonExecutor` from AZR, impls `check_validity` + `eval_student` + `compute_gold_output` |
| `DynamicDataset` Ray actor | ✅ real | FIFO + uniform / recency / mixed sampling, thread-safe |
| `ReActGenerator.cold_start(n)` | ✅ real | OpenAI client → vLLM, returns `n` validated code tasks with `gold_output` filled in |
| `ReActGenerator.generate(memory, n)` | ⛔ Phase 1 stub | raises NotImplementedError |
| `GeneratorMemory`, `student_profiler` | ⛔ Phase 1 stubs | not on the Phase 0 path |
| `ExecutorRewardManager` | ✅ real | verl `AbstractRewardManager` subclass, registered as `"spec_diag_executor"` |
| `DynamicGRPOTrainer` | 🟡 scaffold | wraps verl `RayPPOTrainer` and overrides the dataset path; **not yet smoke-tested on GPU** |
| `main.py` smoke test | ✅ real | end-to-end pipeline check **without** verl / training |

**Bottom line**: `python -m spec_diag.main` verifies that vLLM generator →
DynamicDataset → CodeExecutor → reward grading all work together. Actually
driving a GRPO training run still requires finishing the verl config +
wiring a generator-feeding loop onto the training step callback; that is
the Phase 0 closing task, not yet done.

---

## Repo layout

```
spec_diag/
├── pyproject.toml
├── README.md                       ← you are here
├── docs/
│   └── hpc3_usage.md               HPC3 (ACD) slurm + mirror quick-ref
├── spec_diag/                      python package
│   ├── main.py                     Phase 0 smoke-test entry point
│   ├── configs/
│   │   ├── generator.yaml          ReAct generator config (default: Qwen3-8B)
│   │   └── spec_diag_grpo.yaml     hydra override on top of verl's ppo_trainer
│   ├── generator/
│   │   ├── react_generator.py      ReActGenerator.cold_start (real) + generate (stub)
│   │   ├── memory.py               GeneratorMemory dataclass (Phase 1 stub)
│   │   └── student_profiler.py     LLM diagnosis (Phase 1 stub)
│   ├── executors/
│   │   ├── base.py                 Executor ABC
│   │   ├── code_executor.py        CodeExecutor (Phase 0, real)
│   │   ├── python_executor.py      AZR PythonExecutor, bug-fixed
│   │   ├── templates.py, checks.py, parsers.py
│   ├── dataset/
│   │   └── dynamic_dataset.py      DynamicDatasetImpl + @ray.remote wrapper
│   ├── rewards/
│   │   └── executor_reward.py      verl reward_manager plug-in
│   └── trainer/
│       └── dynamic_grpo_trainer.py RayPPOTrainer-bound scaffold
├── scripts/
│   ├── launch_vllm_gen_2xh100.sh   vLLM server, 2× H100 80GB, TP=2
│   ├── launch_vllm_gen_4xh100.sh   vLLM server, 4× H100 80GB, TP=4
│   ├── setup_env.sh                env bootstrap helper
│   └── run_verl_sanity.sh          stock verl GRPO smoke test
└── tests/
```

`verl` is **not vendored**. Clone it side-by-side.

Pinned verl commit for reproducibility:
```
114ad569f73882b927c21f637d000117703066a3
```

---

## Server setup

```bash
# 1. Clone side-by-side
git clone https://github.com/EdWangLoDaSc/spec_diag.git
git clone https://github.com/verl-project/verl.git
cd verl && git checkout 114ad569f73882b927c21f637d000117703066a3 && cd ..

# 2. Env
conda create -n spec_diag python=3.10 -y
conda activate spec_diag

# 3. Install verl + spec_diag (editable)
pip install -e ./verl
pip install -e './spec_diag[dev]'

# 4. Install vLLM (required — serves the generator)
pip install 'vllm>=0.8.4'      # Qwen3 needs ≥ 0.8.4

# 5. Sanity check
python -c "import verl, spec_diag, vllm; print(verl.__version__ if hasattr(verl,'__version__') else verl.__file__); print(vllm.__version__)"
```

If your verl clone is not at `../verl`, export `VERL_DIR`:
```bash
export VERL_DIR=/path/to/verl
```

### HPC3-specific

On HPC3 nodes you cannot hit public PyPI/HF directly. See
`docs/hpc3_usage.md` for the mirror setup (`harbor.internal.com` for
pip/conda). Pre-stage the Qwen3-8B weights on a machine with network and
rsync them into `$HF_HOME` (default `$SPEC_DIAG_ROOT/.hf_cache`) before
launching vLLM.

---

## The two-process architecture

spec_diag runs **two decoupled processes** that talk over HTTP:

```
  ┌──────────────────────┐       OpenAI API        ┌─────────────────────┐
  │  vLLM server         │  ───────────────────▶   │  training / smoke   │
  │  (generator, frozen) │                         │  (student, Ray)     │
  │  Qwen3-8B            │                         │  python -m spec_diag│
  │  launch_vllm_gen_*.sh│                         │  .main              │
  └──────────────────────┘                         └─────────────────────┘
       holds N GPUs                                     holds its own GPUs
       never restarts during training                  can crash & rerun freely
```

The generator is served as an OpenAI-compatible HTTP endpoint by vLLM
under the alias `generator` (via `--served-model-name`). The training /
smoke-test process talks to it with the `openai` Python SDK, pointed at
`$OPENAI_BASE_URL`.

**Why split**: the frozen 8B model would otherwise reload on every
training restart (~16 GB of weights), and it would fight the student for
GPU memory. Keep them apart.

---

## Run: Phase 0 smoke test

### Step 1 — start the vLLM generator

Pick one of the two launchers depending on how many GPUs you want to
give the generator. Qwen3-8B bf16 is only ~16 GB so 1 H100 is enough;
2/4 cards just buy you longer KV cache and higher throughput.

```bash
chmod +x scripts/launch_vllm_gen_{2,4}xh100.sh

# foreground (blocks the terminal, good for first run to see logs)
bash scripts/launch_vllm_gen_2xh100.sh

# or background + log file
nohup bash scripts/launch_vllm_gen_4xh100.sh > /dev/null 2>&1 &
tail -F /data/user/dingcao/hanyang/spec_diag/logs/vllm/vllm_4xh100_*.log
```

Override any of these via env vars:
```bash
CUDA_VISIBLE_DEVICES=2,3 VLLM_PORT=8001 \
MODEL=Qwen/Qwen3-8B MAX_MODEL_LEN=32768 \
bash scripts/launch_vllm_gen_2xh100.sh
```

Wait until you see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Confirm it's up:
```bash
curl http://localhost:8000/v1/models
# expect: {"object":"list","data":[{"id":"generator",...}]}
```

Stop it later:
```bash
kill $(cat /data/user/dingcao/hanyang/spec_diag/logs/vllm/vllm_2xh100.pid)
```

### Step 2 — run the spec_diag pipeline smoke test

In a **second** terminal (or after detaching the vLLM one):

```bash
cd spec_diag
conda activate spec_diag

export OPENAI_BASE_URL="http://localhost:8000/v1"   # or the generator node hostname
export OPENAI_API_KEY="dummy"                       # vLLM ignores the key
export SPEC_DIAG_MODEL="generator"                  # matches --served-model-name
export SPEC_DIAG_N=4                                # how many cold-start tasks to request

python -m spec_diag.main
```

Expected output (abridged):
```
============================================================
[spec_diag] Phase 0 smoke test
[spec_diag] OPENAI_BASE_URL=http://localhost:8000/v1
[spec_diag] model=generator
[spec_diag] n_cold_start=4
============================================================
[ray] DynamicDataset actor spawned
[generator] got 4 valid tasks from cold_start
  [0] tags=['sort'] inputs='[3, 1, 2]' gold='[1, 2, 3]'
  [1] tags=['string'] inputs="'abc'" gold="'cba'"
  ...
[dataset] add_batch → buffer size = 4
[dataset] stats: {'size': 4, 'min_step': 0, 'max_step': 0, 'tag_counts': {...}}
[dataset] sampled 3 tasks for grading
  task tags=['sort']   r(correct)=1.00 r(wrong)=0.00
  task tags=['string'] r(correct)=1.00 r(wrong)=0.00
  task tags=['arith']  r(correct)=1.00 r(wrong)=0.00
============================================================
[spec_diag] smoke test done.
============================================================
```

The smoke test passes if:
- `got N valid tasks` with N > 0,
- every `r(correct) = 1.00`,
- every `r(wrong) = 0.00`.

If any of those fail, scroll up in the log for `[WARN]` lines.

---

## Troubleshooting

**`openai.APIConnectionError` / `Connection refused`**
vLLM isn't up yet, or the wrong port. Check `curl $OPENAI_BASE_URL/models`.

**`got 0 valid tasks from cold_start`**
The generator returned malformed JSON or every task failed determinism /
validity checks. Try `SPEC_DIAG_N=8` and lower temperature in
`spec_diag/configs/generator.yaml`. Also check the raw response by
bumping the log level (TODO: expose via env var).

**`r(correct) < 1.0` for all sampled tasks**
`gold_output` formatting mismatch between `compute_gold_output` (uses
`repr(f(input))`) and the grader's `eval(gold) == eval(agent)` fast path.
Usually means a task is nondeterministic or produces an unprinted object.

**`r(wrong) > 0`**
Grader is too loose. Should not happen with `"__definitely_wrong__"` as
the fake response — if it does, something is wrong in
`PythonExecutor.eval_output_prediction`.

**vLLM OOM loading Qwen3-8B**
Lower `GPU_MEM_UTIL=0.85` in the launch script, or drop `MAX_MODEL_LEN`.

**`Qwen3ForCausalLM` unknown architecture**
vLLM too old. `pip install -U 'vllm>=0.8.4'`.

**Pebble / Ray hang during validity check**
`CodeExecutor` uses `pebble.ProcessPool`. If it interacts badly with Ray
fork semantics, drop `validity_workers` to 1 in
`ReActGenerator(config={"validity_workers": 1})` (edit `main.py`).

---

## Config

### `spec_diag/configs/generator.yaml`
```yaml
model:
  name: "Qwen/Qwen3-8B"   # served by vLLM under alias "generator"
  backend: "vllm"
  temperature: 0.7
  max_tokens: 4096
react:
  observe_window: 50           # Phase 1
  failure_samples_per_tag: 3   # Phase 1
  tasks_per_round: 32          # Phase 1
memory:
  profile_refresh_every: 4     # Phase 1
  exemplar_pool_size: 16       # Phase 2
```

Change the model here, **and** in the launch script env (or pass
`MODEL=...` to the launcher). The `served-model-name` alias is
`generator`, so your code only needs to reference that.

### Env vars picked up by `main.py`
| Var | Default | Meaning |
|---|---|---|
| `OPENAI_BASE_URL` | `http://localhost:8000/v1` | where vLLM listens |
| `OPENAI_API_KEY` | `dummy` | unused by vLLM |
| `SPEC_DIAG_MODEL` | `generator` (yaml fallback) | model name in API call |
| `SPEC_DIAG_N` | `4` | cold-start task count |

### Env vars picked up by `launch_vllm_gen_{2,4}xh100.sh`
| Var | 2×H100 default | 4×H100 default |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-8B` | `Qwen/Qwen3-8B` |
| `VLLM_PORT` | `8000` | `8000` |
| `MAX_MODEL_LEN` | `16384` | `32768` |
| `GPU_MEM_UTIL` | `0.90` | `0.88` |
| `CUDA_VISIBLE_DEVICES` | `0,1` | `0,1,2,3` |
| `CONDA_ENV` | `/share/anaconda3/envs/vllm` | same |
| `HF_HOME` | `$SPEC_DIAG_ROOT/.hf_cache` | same |
| `SERVED_NAME` | `generator` | same |

---

## Standalone verl GRPO smoke test

Before touching spec_diag, you can verify that your verl + GPU install
works with stock configs:

```bash
bash scripts/run_verl_sanity.sh
```

This points you at verl's own `examples/grpo_trainer/*.sh`. If this
fails, nothing downstream will work — fix it first.

---

## What Phase 0 closing still needs

The smoke test above only exercises the generator → dataset → reward
pipeline. To close Phase 0 you also need:

1. A complete verl PPO config: `spec_diag_grpo.yaml` currently only flips
   `adv_estimator=grpo`. Still missing: `data.train_files`,
   `actor_rollout_ref.model.*`, `trainer.*`, tokenizer wiring, device
   mapping. Copy the closest example from
   `verl/examples/grpo_trainer/*.yaml` and edit.
2. A **feeding loop**: something that calls
   `generator.cold_start(...)` or (Phase 1) `generator.generate(...)`
   every K training steps and `dataset.add_batch(...)`. Simplest path is
   a trainer step callback in `DynamicGRPOTrainer`.
3. Run once with a tiny student (e.g. Qwen2.5-0.5B-Instruct) to verify
   rewards actually flow back into the update.

None of these are blockers for the smoke test, but they're blockers for
"real training".

---

## Phase status

- [x] Scaffolding
- [🟡] Phase 0 pipeline smoke test (generator → dataset → executor → reward): **runnable**
- [ ] Phase 0 training loop (verl GRPO driven by DynamicDataset): **config + feeding loop pending**
- [ ] Phase 1: ReAct Generator with memory + student profiler (code domain)
- [ ] Phase 2: SQL domain

---

## Attribution

- `spec_diag/executors/python_executor.py` (and helpers `templates.py`,
  `checks.py`, `parsers.py`) are copied from
  [Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner),
  adapted from QwQ. Bug fixes applied on top (see git log).
- Training framework: [verl](https://github.com/verl-project/verl).
- Generator: [Qwen3](https://github.com/QwenLM/Qwen3).
