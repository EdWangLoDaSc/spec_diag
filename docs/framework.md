# SPEC-Diag: Self-Evolving Diagnostic Curriculum for Code Reasoning

## 1. Overview

SPEC-Diag is a self-evolving training framework where a frozen **Generator LLM** designs targeted code reasoning tasks for a **Student model** trained via GRPO (Group Relative Policy Optimization). The Generator observes the Student's per-capability performance and adapts the curriculum in real time, focusing on weak areas while maintaining diversity.

```
┌──────────────────┐     OpenAI API      ┌──────────────────────┐
│   vLLM Server    │ ◄──────────────────► │  Training Process    │
│  Generator LLM   │                      │  (Student + verl)    │
│  (frozen, GPU 0) │                      │  (GPU 1,2,3)         │
└──────────────────┘                      └──────────────────────┘
         │                                          │
    Designs tasks                            Trains Student
    based on feedback ◄──── RewardTracker ◄──── Scores responses
```

### Key Contributions

1. **Dynamic Curriculum**: Tasks are generated on-the-fly based on Student performance, not from a fixed dataset.
2. **ReAct-based Generation**: Generator uses [Observe]→[Think]→[Act] framework with memory of Student's capability trajectory.
3. **Multi-task Training**: Four task types (output prediction, input prediction, error prediction, function deduction) train diverse code reasoning abilities.
4. **Adaptive Difficulty**: Task difficulty adjusts automatically based on the Student's overall pass rate.
5. **Cross-process Feedback Loop**: Named Ray actor bridges reward signals from verl's training loop back to the Generator.

---

## 2. Architecture

### 2.1 Two-Process Design

The system runs as two decoupled processes communicating over HTTP:

| Process | Role | GPU | Model |
|---------|------|-----|-------|
| **vLLM Server** | Serves Generator LLM for task generation + student profiling | GPU 0 | Qwen3-8B (frozen) |
| **Training** | GRPO training of Student model via verl | GPU 1,2,3 | Qwen2.5-7B-Instruct |

**Rationale**: Separating Generator from Student avoids GPU memory contention and allows independent restarts. The frozen Generator (~16GB) would otherwise fight the Student for memory during gradient updates.

### 2.2 Component Overview

```
Generator (vLLM)          Feedback Bridge           Student (verl)
────────────────          ───────────────           ──────────────
ReActGenerator            RewardTracker             RayPPOTrainer
  ├── cold_start()        (Named Ray Actor)           ├── Rollout
  ├── generate()            ├── record()              ├── Reward
  └── _validate_specs()     ├── get_report()          ├── GRPO Update
                            └── set_current_step()    └── CRUXEval Eval
GeneratorMemory
  ├── capability_trajectory
  ├── recent_failures
  ├── student_profile
  └── difficulty_hint

StudentProfiler
  └── build_student_profile()
```

---

## 3. Task Types

The framework trains the Student on four complementary code reasoning tasks:

### 3.1 code_o — Output Prediction
- **Student sees**: Function code + input
- **Student predicts**: `repr(f(input))`
- **Evaluation**: `eval(gold_output) == eval(student_output)` via `PythonExecutor.eval_output_prediction()`
- **Example prompt**: "Given `def f(x): return sorted(x)` and input `[3,1,2]`, predict the output."

### 3.2 code_i — Input Prediction
- **Student sees**: Function code + output
- **Student predicts**: A valid input that produces that output
- **Evaluation**: `gold_output == f(student_input)` via `PythonExecutor.eval_input_prediction()`
- **Difficulty**: Requires reverse-engineering the algorithm — harder than output prediction.

### 3.3 code_e — Error Prediction
- **Student sees**: Function code + input
- **Student predicts**: Error type (e.g., `TypeError`, `ValueError`, `IndexError`) or `NoError`
- **Evaluation**: Case-insensitive string match of first token
- **Constraint**: Generated code must *actually* raise an error on the given input (validated during generation).

### 3.4 code_f — Function Deduction (feat/code-f-task-type branch)
- **Student sees**: Input/output pairs (first half) + hint message
- **Student predicts**: The function `def f(...)` that produces those outputs
- **Evaluation**: Run student's function on hidden pairs (second half), `gold_output == f(student_input)` per pair, return mean accuracy.
- **Difficulty**: Highest — requires synthesizing a program from examples.

### Target Distribution
Prompts request approximately: 40% code_o, 20% code_i, 20% code_e, 20% code_f.

---

## 4. Task Generation Pipeline

### 4.1 Cold Start (Phase 0)

Before any training feedback exists, the Generator produces random tasks:

```python
cold_start(n=64)
  → _fanout_chat(system, user, n)          # concurrent LLM calls (4 tasks/call × 16 calls)
    → _chat(system_prompt, user_prompt)    # OpenAI API to vLLM
    → _parse_json_list(raw_response)       # strip <think> tags, extract JSON
  → _validate_specs(specs, n)              # filter + compute gold answers
  → DynamicDataset.add_batch(tasks, step)  # push to buffer
```

### 4.2 Memory-Conditioned Generation (Phase 1)

After training begins, the Generator uses Student performance feedback:

```python
generate(memory, n=64)
  → memory.snapshot_prompt_context()    # get pass rates, failures, difficulty hint
  → format ReAct prompt:
      [Observe] student profile + capability summary + failures
      [Difficulty] adaptive hint based on overall pass rate
      [Think] design tasks for weak areas
      [Act] output JSON task list
  → _fanout_chat(system, user, n)
  → _validate_specs(specs, n)
  → fallback to cold_start(n) if 0 valid tasks
```

### 4.3 Validation Pipeline (`_validate_specs`)

Every generated task passes through:

| Check | What it does | Rejection reason |
|-------|-------------|------------------|
| Format | Code and inputs are strings | `format_bad` |
| Dedup | `(code.strip(), inputs.strip())` key | `duplicate` |
| Stub detection | Scan for `pass`, `# TODO`, `# implement` | `validity_fail` |
| Banned keywords | AST-level check for 16 banned modules (random, time, os, etc.) | `validity_fail` |
| Determinism | Run code twice, compare outputs | `validity_fail` |
| Gold answer | Execute code to compute ground truth | `gold_fail` |
| code_e: must error | Reject `error_type == "NoError"` | `gold_fail` |
| code_i: non-trivial | Reject `gold_output == "None"` | `gold_fail` |

**Tag normalization**: Tags are lowercased and whitespace/hyphens replaced with underscores to prevent fragmentation (e.g., `"error handling"` → `"error_handling"`).

### 4.4 Reference Snippets

Each generation prompt includes few-shot reference snippets sampled from `data/seed_io.jsonl` (256 entries from AZR's 7B seed data). This anchors the difficulty level and output format:

```
<snippet_0>
```python
def f(nums):
    # complex algorithm...
```
```input
[1, 2, 3, 4, 5]
```
```output
[(4, 1), (2, 3)]
```
</snippet_0>
```

Cold start uses 6 references; `generate()` uses 3 (to leave room for memory context).

---

## 5. Feedback Loop (Phase 1)

### 5.1 RewardTracker — Cross-Process Bridge

The key challenge: verl's `RayPPOTrainer` has no callback hooks, and `compute_score()` runs in a separate `RewardLoopWorker` Ray actor. Solution: a **named Ray actor** (`spec_diag_reward_tracker`) that bridges the two processes.

```
RewardLoopWorker process              SpecDiagTaskRunner process
──────────────────────                ─────────────────────────
compute_score()                       _FeederThread
  │                                     │
  ├── eval_student(task, response)      ├── set_current_step(step)
  ├── score = 0.0 or 1.0               ├── get_report(since_step)
  └── record.remote(                    │     → {per_tag_pass_rates,
        tags, score,        ──────►     │        per_tag_counts,
        task, response)    fire&forget  │        failures}
                                        │
                           RewardTracker (Named Ray Actor)
                             ├── _scores[tag] → [float]
                             ├── _steps[tag]  → [int]
                             ├── _failures[tag] → [{task, response, score}]
                             └── _current_step
```

**Design decisions**:
- **Fire-and-forget**: `record.remote()` never calls `ray.get()` — reward computation is never blocked.
- **Retry on lookup failure**: Up to 20 retries if `ray.get_actor()` fails (actor may not be created yet at startup).
- **Memory cap**: Max 2000 scores per tag to prevent unbounded growth.
- **Tag normalization**: Applied at record time to reduce fragmentation.

### 5.2 GeneratorMemory

In-process (lives in the feeder thread) data structure that accumulates Student performance:

```python
@dataclass
class GeneratorMemory:
    task_history: list[dict]              # per-round {per_tag_pass_rates, total_tasks}
    capability_trajectory: dict[str, list[float]]  # tag → [rate_t0, rate_t1, ...]
    recent_failures: list[dict]           # [{tags, task, response, score}], deduped, max 30
    student_profile: str                  # natural language diagnosis (refreshed every 4 rounds)
    exemplar_pool: list[dict]             # Phase 2 (future)
```

**`snapshot_prompt_context()`** produces the dict injected into the ReAct prompt:
- Filters tags with < 3 observations (removes single-sample noise)
- Caps at 30 tags sorted by pass rate
- Computes `overall_pass_rate` and `difficulty_hint`
- Classifies tags as weak (< 50%) or strong (> 80%)

### 5.3 Student Profiler

Every 4 memory updates, the feeder thread calls `build_student_profile()` which uses the same vLLM endpoint to generate a 2-3 sentence natural language diagnosis:

> "The student struggles with recursive tree traversal (28% pass rate) and dictionary manipulation (35%), likely due to difficulty tracking nested state. String operations are strong (82%)."

This is injected into the ReAct prompt's `[Observe]` section for richer context.

### 5.4 Adaptive Difficulty

The `difficulty_hint` adjusts based on `overall_pass_rate`:

| Pass Rate | Hint |
|-----------|------|
| < 30% | "Generate EASIER tasks — simpler logic, shorter code" |
| 30-70% | "Maintain current difficulty — target 30-70% success" |
| > 70% | "Generate HARDER tasks — deeper recursion, trickier edge cases" |

### 5.5 Feeder Thread Lifecycle

```
_FeederThread (daemon, polls every 5s)
│
├── Warmup: cold_start 256 tasks synchronously (before training)
│
└── Loop (every 5s):
    if steps_since_last_generate >= 8:
      ├── _update_memory(step)
      │     ├── set_current_step.remote(step)
      │     ├── get_report.remote(since_step)
      │     ├── memory.update(report)
      │     └── [every 4 rounds] build_student_profile()
      │
      ├── if memory has data → generator.generate(memory, 64)
      │   else              → generator.cold_start(64)
      │
      ├── DynamicDataset.add_batch(tasks, step)
      └── _save_tasks(tasks + llm_calls → $RUN_DIR/tasks/)
```

---

## 6. Training Pipeline

### 6.1 verl Integration

SPEC-Diag wraps verl's `RayPPOTrainer` via `DynamicGRPOTrainer`:

```python
DynamicGRPOTrainer
  ├── build(run_dir)
  │     ├── _warmup_buffer(256)       # synchronous cold_start
  │     ├── _start_feeder(run_dir)    # daemon thread
  │     ├── _start_checkpoint()       # periodic buffer save
  │     ├── _build_val_dataset()      # CRUXEval 1600 samples
  │     └── RayPPOTrainer(train_dataset=DynamicMapDataset, val_dataset=CRUXEvalDataset)
  │
  ├── init_workers()                  # actor/ref/rollout worker groups
  │
  └── fit()
        ├── RayPPOTrainer.fit()       # verl GRPO loop
        └── finally: stop feeder + checkpoint
```

### 6.2 DynamicMapDataset

Map-style dataset backed by `DynamicDataset` Ray actor:

- `__len__` = `samples_per_epoch` (1024) — fixed, for DataLoader compatibility
- `__getitem__(i)` calls `sample_batch.remote(1, strategy)` — **sampling with replacement**, index `i` is ignored
- `num_workers = 0` — required because Ray actor handles don't survive fork
- Blocks up to 300s if buffer is empty (safety net against dead feeder)

### 6.3 GRPO Training

Standard verl GRPO with these overrides:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `adv_estimator` | `grpo` | Group relative advantage |
| `rollout.n` | 8 | 8 rollouts per prompt |
| `train_batch_size` | 24 | Per GPU batch |
| `use_kl_loss` | true | Prevent mode collapse |
| `kl_loss_coef` | 0.001 | Gentle KL penalty |
| `total_epochs` | 4 | 128 steps total |
| `max_prompt_length` | 1024 | |
| `max_response_length` | 512 | |

### 6.4 Reward Function

`compute_score(data_source, solution_str, ground_truth)` routes by `data_source`:

| data_source | Evaluation | Score |
|-------------|-----------|-------|
| `spec_diag_code` | `CodeExecutor.eval_student()` → routes by `task_type` | 0.0 or 1.0 |
| `cruxeval_o` | `eval_output_prediction()` | 0.0 or 1.0 |
| `cruxeval_i` | `eval_input_prediction()` | 0.0 or 1.0 |
| `humaneval` | evalplus `check_correctness()` | 0.0 or 1.0 |

The `CodeExecutor` is a module-level singleton to avoid creating/destroying `ProcessPool` on every call (256 calls per training step with GRPO n=8).

---

## 7. Evaluation

### 7.1 CRUXEval (Primary)

[CRUXEval](https://huggingface.co/datasets/cruxeval-org/cruxeval) — 800 Python code reasoning problems, directly measuring the trained capabilities:

- **CRUXEval-O** (800 samples): code + input → predict output = code_o capability
- **CRUXEval-I** (800 samples): code + output → predict input = code_i capability
- Total: 1600 validation samples

Evaluated every `test_freq=30` training steps via verl's `_validate()`:
1. Training pauses
2. Current Student generates answers for all 1600 samples
3. `compute_score` grades each with `CodeExecutor`
4. verl computes `mean@1` per data_source, logs to TensorBoard

TensorBoard metrics:
- `val-core/cruxeval_o/acc/mean@1`
- `val-core/cruxeval_i/acc/mean@1`

### 7.2 HumanEval (Secondary, Optional)

164 HumanEval problems via evalplus. Tests whether code reasoning ability transfers to code generation. Requires `pip install evalplus`.

---

## 8. Task Quality Controls

### 8.1 Banned Keywords (AST-level)

16 keywords checked via `ast.parse()` + tree walk:

```
logging, random, multiprocessing, pebble, subprocess, threading,
datetime, time, hashlib, hmac, bcrypt, os.sys, os.path, sys.exit,
os.environ, calendar
```

### 8.2 Determinism Check

Every task is executed twice via `CHECK_DETERMINISM_TEMPLATE`:
```python
returns = f({inputs})
if returns != f({inputs}):
    raise Exception('Non-deterministic code')
```

### 8.3 Stub Detection

Rejects functions containing:
- `# TODO`, `# implement`, `# your code`, `NotImplementedError`
- Body ending in `pass`, `return -1`, `return None`, `...`

### 8.4 Gold Answer Verification

- **code_o / code_i**: `compute_gold_output()` executes `f(inputs)` and stores `repr()` result
- **code_e**: `compute_error_type()` executes code and extracts error type via `parse_error(status)`; rejects `NoError`
- **code_f**: `compute_io_pairs()` runs gold function on all inputs; requires ≥ 2 valid pairs

---

## 9. Logging & Observability

### 9.1 Run Directory Structure

```
$RUN_DIR/
├── train.log                 # full console output (verl + spec_diag)
├── train_python.log          # Python logger output
├── config_resolved.yaml      # resolved Hydra config
├── tasks/
│   ├── batch_00001_step8.json
│   │   {
│   │     "batch": 1,
│   │     "step": 8,
│   │     "mode": "memory-conditioned",
│   │     "n_tasks": 29,
│   │     "tasks": [...],
│   │     "llm_calls": [
│   │       {"system": "...", "user": "...", "raw_response": "..."}
│   │     ]
│   │   }
│   └── ...
└── tensorboard_log/
    └── spec_diag/...
```

### 9.2 Validation Stats Logging

Each `_validate_specs` call logs:
```
validate_specs: 26/46 passed (format_bad=0, dup=5, validity_fail=11, gold_fail=4,
                               types={'code_o': 18, 'code_e': 6, 'code_i': 2})
```

---

## 10. Configuration

### 10.1 Generator Config (`spec_diag/configs/generator.yaml`)

```yaml
model:
  name: "Qwen/Qwen3-8B"
  temperature: 0.7
  max_tokens: 4096
  seed_data_path: data/seed_io.jsonl    # 256 AZR reference snippets
  n_references: 6                       # few-shot references per prompt
  tasks_per_call: 4                     # tasks per LLM call
  max_generation_workers: 8             # concurrent LLM calls

react:
  failure_samples_per_tag: 3
  regenerate_every_steps: 8             # curriculum refresh interval

memory:
  profile_refresh_every: 4              # student profile refresh interval
```

### 10.2 Training Config (`spec_diag/configs/spec_diag_grpo.yaml`)

```yaml
algorithm:
  adv_estimator: grpo

data:
  train_batch_size: 24
  max_prompt_length: 1024
  max_response_length: 512
  return_raw_chat: true

spec_diag:
  samples_per_epoch: 1024
  sample_strategy: mixed                # uniform + recency-weighted
  feeder:
    feed_batch: 64
    low_watermark: 128
    warmup_tasks: 256

trainer:
  test_freq: 30                         # CRUXEval every 30 steps
  logger: ["console", "tensorboard"]
```

---

## 11. Observed Results

### Training Score Progression (Qwen3-8B Generator, 120 steps)

| Steps | Mean Score | Phase |
|-------|-----------|-------|
| 1-10 | 0.33 | Initial (cold_start tasks) |
| 11-30 | 0.44 | Warming up |
| 31-60 | 0.58 | Rapid improvement |
| 61-90 | 0.66 | Approaching plateau |
| 91-120 | 0.65 | Stable |

### CRUXEval Trend (Qwen2.5-7B Generator, 120 steps)

| Step | CRUXEval-O | CRUXEval-I |
|------|-----------|-----------|
| 30 | 23.3% | 25.4% |
| 60 | 28.5% | 35.8% |
| 90 | 29.8% | 35.0% |
| 120 | 29.0% | 32.6% |

### Task Generation Statistics

- Validation pass rate: ~42-60%
- Type distribution: code_o 53%, code_e 25%, code_f 14%, code_i 9%
- Phase 1 activation: alternating cold_start / memory-conditioned batches

---

## 12. Repository Structure

```
spec_diag/
├── data/
│   ├── seed_io.jsonl              # 256 AZR reference snippets
│   └── cruxeval.jsonl             # 800 CRUXEval problems
├── spec_diag/
│   ├── train.py                   # Hydra entry point + SpecDiagTaskRunner
│   ├── configs/
│   │   ├── generator.yaml         # Generator LLM config
│   │   └── spec_diag_grpo.yaml    # verl training config
│   ├── generator/
│   │   ├── react_generator.py     # cold_start + generate + _validate_specs
│   │   ├── memory.py              # GeneratorMemory dataclass
│   │   ├── student_profiler.py    # LLM-based diagnosis
│   │   └── prompts/__init__.py    # ReAct prompt templates
│   ├── executors/
│   │   ├── code_executor.py       # eval_student routing + gold answer computation
│   │   ├── python_executor.py     # AZR PythonExecutor (pebble ProcessPool)
│   │   ├── templates.py           # execution templates
│   │   ├── checks.py              # banned keyword AST check
│   │   └── parsers.py             # error type extraction
│   ├── rewards/
│   │   ├── spec_diag_score.py     # compute_score (verl reward function)
│   │   └── reward_tracker.py      # RewardTracker named Ray actor
│   ├── dataset/
│   │   └── dynamic_dataset.py     # DynamicDataset Ray actor (task buffer)
│   ├── trainer/
│   │   └── dynamic_grpo_trainer.py # DynamicGRPOTrainer + FeederThread
│   └── eval/
│       ├── cruxeval_dataset.py    # CRUXEval validation dataset
│       └── humaneval_dataset.py   # HumanEval validation dataset
├── scripts/
│   ├── step_1_vllm.sh             # Start vLLM (1 GPU)
│   ├── step2_grpo.sh              # Start training (2 GPU)
│   └── step2_grpo_3gpu.sh         # Start training (3 GPU)
└── verl/                          # Vendored verl framework
```
