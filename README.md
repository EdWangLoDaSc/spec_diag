# spec_diag

Dynamic curriculum framework: a frozen LLM **Generator** (ReAct + memory) feeds
tasks to a GRPO-trained **Student**, built on top of
[verl](https://github.com/verl-project/verl.git).

See `docs/experiment_plan.md` (or `../idea-iter/experiment_plan.md` in the
monorepo layout) for the full spec.

---

## Repo layout

```
spec_diag/
├── pyproject.toml
├── spec_diag/              # python package
│   ├── generator/          # ReAct Generator + memory (Phase 1)
│   ├── executors/          # PythonExecutor + Executor ABC (Phase 0)
│   ├── dataset/            # DynamicDataset Ray Actor (Phase 0)
│   ├── rewards/            # verl reward_manager plug-in (Phase 0)
│   ├── trainer/            # RayPPOTrainer subclass (Phase 0)
│   ├── configs/            # Hydra configs composed on top of verl
│   └── main.py             # orchestrator entry point
├── scripts/
│   ├── setup_env.sh        # doc-only env bootstrap
│   └── run_verl_sanity.sh  # smoke test launcher for verl GRPO example
└── tests/
```

`verl` is **not vendored**. It lives in a separate clone (see setup below).

Pinned verl commit for reproducibility:
```
114ad569f73882b927c21f637d000117703066a3
```

---

## Server setup

On a fresh machine, clone **both** this repo and verl side-by-side:

```bash
# 1. Clone spec_diag and verl
git clone https://github.com/EdWangLoDaSc/spec_diag.git
git clone https://github.com/verl-project/verl.git
cd verl && git checkout 114ad569f73882b927c21f637d000117703066a3 && cd ..

# 2. Create env
conda create -n spec_diag python=3.10 -y
conda activate spec_diag

# 3. Install verl (editable)
pip install -e ./verl

# 4. Install spec_diag (editable, with dev extras)
pip install -e './spec_diag[dev]'

# 5. Sanity check
python -c "import verl, spec_diag; print(verl.__file__); print(spec_diag.__file__)"
```

If your verl clone is not at `../verl` relative to `spec_diag/`, export
`VERL_DIR` before running the helper scripts:

```bash
export VERL_DIR=/path/to/verl
```

`scripts/setup_env.sh` and `scripts/run_verl_sanity.sh` honor `VERL_DIR`.

---

## Smoke test verl's stock GRPO

```bash
cd spec_diag
bash scripts/run_verl_sanity.sh
```

This does **not** touch any spec_diag code — it points you at verl's own
`examples/grpo_trainer/*.sh` scripts. Pick the smallest one and run it first
to confirm your GPU + verl install is healthy. **If this fails, nothing
downstream matters.**

---

## Phase status

- [x] Scaffolding
- [ ] Phase 0: DynamicDataset + PythonExecutor wired into verl reward_manager + GRPO smoke test
- [ ] Phase 1: ReAct Generator on code domain
- [ ] Phase 2: SQL domain

---

## Attribution

- `spec_diag/executors/python_executor.py` (and its helpers `templates.py`,
  `checks.py`, `parsers.py`) are copied from
  [Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner),
  which in turn adapted them from QwQ.
- Training framework: [verl](https://github.com/verl-project/verl).
