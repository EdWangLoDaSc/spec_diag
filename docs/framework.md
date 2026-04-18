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
5. **Mastered Task Eviction**: Tasks the student has fully learned are automatically removed from the buffer, keeping the curriculum challenging.
6. **Cross-process Feedback Loop**: Named Ray actor bridges reward signals from verl's training loop back to the Generator.
7. **Multi-domain**: Independent code and math areas share the same framework, each with its own Generator, Executor, and evaluation.

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
                            ├── set_current_step()    └── CRUXEval Eval
MathGenerator (math area)   ├── get_mastered_keys()
  ├── cold_start()          └── per-task mastery
  └── generate()
                          DynamicDataset
GeneratorMemory             ├── task buffer
  ├── capability_trajectory   ├── sample_batch()
  ├── recent_failures         └── evict_mastered()
  ├── student_profile
  ├── difficulty_hint
  └── overall_pass_rate

StudentProfiler
  └── build_student_profile()
```

---

## 3. Task Types

### 3.1 Code Domain

#### code_o — Output Prediction
- **Student sees**: Function code + input
- **Student predicts**: `repr(f(input))`
- **Evaluation**: `eval(gold_output) == eval(student_output)` via `PythonExecutor.eval_output_prediction()`

#### code_i — Input Prediction
- **Student sees**: Function code + output
- **Student predicts**: A valid input that produces that output
- **Evaluation**: `gold_output == f(student_input)` via `PythonExecutor.eval_input_prediction()`
- **Difficulty**: Requires reverse-engineering the algorithm — harder than output prediction.

#### code_e — Error Prediction
- **Student sees**: Function code + input
- **Student predicts**: Error type (e.g., `TypeError`, `ValueError`, `IndexError`) or `NoError`
- **Evaluation**: Case-insensitive string match of first token
- **Constraint**: Generated code must *actually* raise an error on the given input (validated during generation).

#### code_f — Function Deduction (`feat/code-f-task-type` branch)
- **Student sees**: Input/output pairs (first half) + hint message
- **Student predicts**: The function `def f(...)` that produces those outputs
- **Evaluation**: Run student's function on hidden pairs (second half), `gold_output == f(student_input)` per pair, return mean accuracy.
- **Difficulty**: Highest — requires synthesizing a program from examples.

**Target distribution**: 50% code_o, 25% code_i, 25% code_e (main); 40/20/20/20 with code_f.

### 3.2 Math Domain (`feat/math-tasks` branch)

#### math_o — Math Problem Solving
- **Student sees**: Natural language math problem
- **Student responds**: Chain-of-thought reasoning + final answer in `\boxed{}`
- **Evaluation**: Extract `\boxed{}` → normalize LaTeX → string match + numeric comparison
- **Verification code**: Hidden `def f(): return <computation>` — must actually COMPUTE the answer (not hardcode a constant, validated via AST)
- **Topics**: Arithmetic, algebra, number theory, combinatorics, probability, geometry, sequences, modular arithmetic

**Independent from code area**: Separate `MathGenerator`, `MathExecutor`, prompts, config, seed data. Can run as a standalone experiment.

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
| math_o: no hardcoding | AST check rejects `def f(): return 42` (constant) | `validity_fail` |
| Auto-tagging | Replace LLM tags with AST-derived tags | — |

### 4.4 AST-Based Auto-Tagging

LLM-generated tags suffer from fragmentation ("recursion" / "tree_recursion" / "recursive_traversal" → 3 tags for the same capability). Instead, `auto_tag(code)` extracts tags deterministically from the AST:

| Category | Tags |
|----------|------|
| Control flow | `recursion`, `loop`, `nested_loop`, `conditional` |
| Data structures | `list`, `dict`, `set`, `string`, `tuple`, `stack_queue` |
| Algorithms | `sorting`, `search`, `math`, `bitwise` |
| Patterns | `comprehension`, `lambda`, `class`, `exception` |
| Size | `short` (<10 lines), `medium` (10-30), `long` (>30) |
| Cyclomatic complexity | `cc_simple` (≤3), `cc_moderate` (4-8), `cc_complex` (>8) |

Cyclomatic complexity counts decision points in the AST (if/for/while/except/and/or/assert/ternary). This decouples size from reasoning difficulty — a 5-line function with 3 nested conditionals is `short` + `cc_moderate`, not just "short".

The Generator is NOT asked to produce `capability_tags` — they are computed after validation from the code itself. Same code always produces the same tags. Combined with task-type prefixing (§5.1), the memory sees tags like `code_i:recursion`, `code_o:cc_complex`.

### 4.4 Reference Snippets

Each generation prompt includes few-shot reference snippets sampled from seed data. This anchors the difficulty level and output format:

- **Code**: `data/seed_io.jsonl` (256 entries from AZR's 7B seed data)
- **Math**: `data/math_seeds.jsonl` (20 reference problems)

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
                                        ├── get_mastered_keys(threshold=3)
                           RewardTracker│     → [(code, inputs), ...]
                             ├── _scores[tag]
                             ├── _failures[tag]
                             ├── _task_perfect_streak[key]
                             └── _mastered_keys
```

**Design decisions**:
- **Fire-and-forget**: `record.remote()` never calls `ray.get()` — reward computation is never blocked.
- **Task-type-prefixed tags**: Tags are reported with both prefixed and raw forms (e.g., `["sorting"]` → `["code_i:sorting", "sorting"]`). This lets memory track per-task-type pass rates, so the Generator can see "code_i:sorting is weak" and automatically increase code_i output without hardcoded quotas.
- **Retry on lookup failure**: Up to 20 retries if `ray.get_actor()` fails (actor may not be created yet at startup).
- **Memory cap**: Max 2000 scores per tag to prevent unbounded growth.
- **Tag normalization**: Applied at record time to reduce fragmentation.
- **n_rollouts**: Read from `config.actor_rollout_ref.rollout.n` (not hardcoded).

### 5.2 Task Eviction (Zone of Proximal Development)

Tasks at both extremes of difficulty are automatically removed from the buffer:

| Type | Condition | Threshold | Min residence |
|------|-----------|-----------|---------------|
| **Mastered** (too easy) | All `n_rollouts` score 1.0 for 3 consecutive batches | `3 × n_rollouts` perfect streak | ≥ 16 training steps in buffer |
| **Impossible** (too hard) | All `n_rollouts` score 0.0 for 2 consecutive batches | `2 × n_rollouts` zero streak | None |

**Asymmetric design**: Mastered eviction is conservative (3 batches + residence requirement) because a task that seems easy early may become challenging after the student's distribution shifts. Impossible eviction is aggressive (2 batches, no residence) because GRPO's learnability mask (§6.6) already prevents gradient noise from these tasks — eviction just saves sampling cost.

**Complementarity of Layer 1 and Layer 2**: A "borderline" task — all-wrong in batch 1, then 1/8 correct in batch 2 — is NOT evicted (streak reset), and Layer 2 masks batch 1 but lets batch 2 produce gradients. This is exactly the "critical difficulty" zone where learning happens most.

**Mechanism**:
1. `RewardTracker` maintains per-task `_task_perfect_streak`, `_task_zero_streak`, and `_task_first_seen_step`.
2. `get_mastered_keys(threshold=3, min_residence_steps=16)` / `get_impossible_keys(threshold=2)` return eviction candidates.
3. `DynamicDataset.evict_mastered(keys)` removes matching tasks.
4. Feeder thread calls `_evict_outliers()` before each regeneration, then Generator fills the gaps.

**Three layers of learnability control**:
```
Layer 1: Buffer eviction     — remove tasks after consecutive extreme batches
Layer 2: GRPO learnability   — zero advantage for all-wrong groups (§6.6)
Layer 3: Format reward       — penalize unstructured output (§6.5)
```

**Effect**: The buffer naturally converges to tasks in the zone of proximal development (mixed correctness, useful gradient signal).

### 5.3 GeneratorMemory

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
- Tags include task-type prefixes (e.g., `code_i:sorting`, `code_o:sorting`), enabling the Generator to see which *task type × capability* combinations are weak and adjust output accordingly — this replaces hardcoded task-type quotas with organic feedback-driven balancing

### 5.4 Student Profiler

Every 4 memory updates, the feeder thread calls `build_student_profile()` which uses the same vLLM endpoint to generate a 2-3 sentence natural language diagnosis:

> "The student struggles with recursive tree traversal (28% pass rate) and dictionary manipulation (35%), likely due to difficulty tracking nested state. String operations are strong (82%)."

This is injected into the ReAct prompt's `[Observe]` section for richer context. Qwen3 `<think>` tags are stripped from the response.

### 5.5 Adaptive Difficulty

The `difficulty_hint` adjusts based on `overall_pass_rate`:

| Pass Rate | Hint |
|-----------|------|
| < 30% | "Generate EASIER tasks — simpler logic, shorter code" |
| 30-70% | "Maintain current difficulty — target 30-70% success" |
| > 70% | "Generate HARDER tasks — deeper recursion, trickier edge cases" |

### 5.6 Feeder Thread Lifecycle

```
_FeederThread (daemon, polls every 5s)
│
├── Warmup: cold_start 256 tasks synchronously (before training)
│
└── Loop (every 5s):
    if steps_since_last_generate >= 8:
      ├── _evict_mastered()                # remove tasks student has mastered
      │     ├── get_mastered_keys(threshold=3)
      │     └── evict_mastered(keys)
      │
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

### 6.4 Student Prompts (CoT + `<answer>` Tags)

Student prompts follow AZR's "think step by step" convention. The student produces chain-of-thought reasoning followed by a structured answer tag:

```
Given the Code Snippet and Input, think step by step then deduce the output.
Put your final output in <answer> tags.
```

**Answer extraction** (`_extract_answer`) uses 7 fallback strategies:

| Priority | Strategy | Example |
|----------|----------|---------|
| 1 | `<answer>...</answer>` | `"Let me think...<answer>[1,2,3]</answer>"` |
| 2 | `\boxed{...}` | `"Therefore \boxed{42}"` |
| 3 | `` ```output/input``` `` block | `` "```output\n[1,2,3]\n```" `` |
| 4 | `` ```python``` `` block | For code_f function deduction |
| 5 | Last line prefix strip | `"Answer: 42"`, `"The answer is 5"` |
| 6 | Short response passthrough | `"42"` (< 50 chars, likely IS the answer) |
| 7 | Last non-empty line | Final fallback for long CoT |

### 6.5 Reward Function

`compute_score(data_source, solution_str, ground_truth)` routes by `data_source`:

| data_source | Evaluation | Score |
|-------------|-----------|-------|
| `spec_diag_code` | `CodeExecutor.eval_student()` → extract `<answer>` → routes by `task_type` | 0.0 or 1.0 |
| `spec_diag_math` | `MathExecutor.eval_student()` → `\boxed{}` extraction + normalization | 0.0 or 1.0 |
| `cruxeval_o` | `eval_output_prediction()` | 0.0 or 1.0 |
| `cruxeval_i` | `eval_input_prediction()` | 0.0 or 1.0 |
| `math500` | `grade_math_answer()` — LaTeX normalization + symbolic compare | 0.0 or 1.0 |
| `humaneval` | evalplus `check_correctness()` | 0.0 or 1.0 |

**Format reward**: After accuracy scoring, a format penalty is applied:
- Has `<answer>...</answer>` tags → full reward (student learned the format)
- No tags but answer extracted via fallback → reward × 0.5
- Empty response → -1.0

This teaches the student to produce structured output over time.

### 6.6 Learnability Filter (GRPO Advantage Masking)

In GRPO, each prompt gets `n` rollouts. When ALL rollouts score 0.0 (all-wrong group), the advantage is mathematically 0 but numerically noisy (`0 / epsilon`). This wastes gradient updates.

**Fix**: In `compute_grpo_outcome_advantage`, groups where `std < epsilon AND mean < epsilon` (all-wrong) have their advantages explicitly zeroed. Applied to both loop and vectorized GRPO variants.

```
All-wrong group  (std=0, mean=0): advantage = 0  → masked, no gradient noise
All-correct group (std=0, mean=1): advantage = 0  → naturally zero, no mask needed
Mixed group       (std>0):         advantage = (r-mean)/std → normal GRPO
```

The `CodeExecutor` and `MathExecutor` are module-level singletons to avoid creating/destroying `ProcessPool` on every call.

---

## 7. Evaluation

### 7.1 CRUXEval (Primary, Code Domain)

[CRUXEval](https://huggingface.co/datasets/cruxeval-org/cruxeval) — 800 Python code reasoning problems, directly measuring the trained capabilities:

- **CRUXEval-O** (800 samples): code + input → predict output = code_o capability
- **CRUXEval-I** (800 samples): code + output → predict input = code_i capability
- Total: 1600 validation samples

TensorBoard metrics:
- `val-core/cruxeval_o/acc/mean@1`
- `val-core/cruxeval_i/acc/mean@1`

### 7.2 MATH500 (Primary, Math Domain — `feat/math-tasks` branch)

[MATH500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) — 500 competition-level math problems:

- Subjects: Algebra (124), Intermediate Algebra (97), Prealgebra (82), Number Theory (62), Precalculus (56), Geometry (41), Counting & Probability (38)
- Student uses chain-of-thought, answer extracted from `\boxed{}`
- Grading: LaTeX normalization (`\frac`, `\dfrac`, `\text`, etc.) + numeric comparison

TensorBoard metric: `val-core/math500/acc/mean@1`

### 7.3 HumanEval (Secondary, Optional)

164 HumanEval problems via evalplus. Tests whether code reasoning ability transfers to code generation. Requires `pip install evalplus`.

### Evaluation Timing

Evaluated every `test_freq=30` training steps via verl's `_validate()`:
1. Training pauses
2. Current Student generates answers for all validation samples
3. `compute_score` grades each sample
4. verl computes `mean@1` per data_source, logs to TensorBoard
5. Training resumes

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
- **math_o**: `compute_gold_answer()` runs `def f(): return <computation>`; AST check rejects hardcoded constants

### 8.5 Math Answer Anti-Hardcoding

For math tasks, the verification code must COMPUTE the answer, not hardcode it:

```python
# REJECTED (ast.Constant return value)
def f(): return 42

# ACCEPTED (ast.BinOp return value)
def f(): return 2**10 + 3**5

# ACCEPTED (ast.Call return value)
def f(): return math.gcd(84, 126)
```

Checked via `_is_hardcoded_constant()` using AST analysis.

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

### 9.3 Mastery Eviction Logging

```
spec_diag feeder: evicted 3 mastered tasks (3 keys)
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

### 10.2 Math Generator Config (`spec_diag/configs/math_generator.yaml`)

```yaml
model:
  name: "Qwen/Qwen3-8B"
  temperature: 0.7
  max_tokens: 4096
  seed_data_path: data/math_seeds.jsonl  # 20 reference problems
  n_references: 4
```

### 10.3 Training Config (`spec_diag/configs/spec_diag_grpo.yaml`)

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

## 12. Branches

| Branch | Content |
|--------|---------|
| `main` | Code domain: code_o, code_i, code_e + CRUXEval eval |
| `feat/code-f-task-type` | + code_f + task eviction + learnability mask + CoT prompts + format reward + auto-tagging |
| `feat/math-tasks` | + Math domain: math_o + MATH500 eval + CoT + anti-hardcoding |

---

## 13. Repository Structure

```
spec_diag/
├── data/
│   ├── seed_io.jsonl              # 256 AZR code reference snippets
│   ├── cruxeval.jsonl             # 800 CRUXEval problems
│   ├── math_seeds.jsonl           # 20 math reference problems
│   └── math500.jsonl              # 500 MATH500 eval problems
├── docs/
│   └── framework.md               # this document
├── spec_diag/
│   ├── train.py                   # Hydra entry point + SpecDiagTaskRunner
│   ├── configs/
│   │   ├── generator.yaml         # Code Generator config
│   │   ├── math_generator.yaml    # Math Generator config
│   │   └── spec_diag_grpo.yaml    # verl training config
│   ├── generator/
│   │   ├── react_generator.py     # Code: cold_start + generate + _validate_specs
│   │   ├── math_generator.py      # Math: cold_start + generate + _validate_specs
│   │   ├── memory.py              # GeneratorMemory dataclass
│   │   ├── student_profiler.py    # LLM-based diagnosis
│   │   └── prompts/__init__.py    # ReAct prompt templates (code)
│   ├── executors/
│   │   ├── code_executor.py       # Code: eval routing + <answer> extraction + gold answer
│   │   ├── math_executor.py       # Math: eval + gold answer + anti-hardcoding
│   │   ├── auto_tagger.py         # AST-based capability tag extraction (~20 fixed tags)
│   │   ├── python_executor.py     # AZR PythonExecutor (pebble ProcessPool)
│   │   ├── templates.py           # execution templates
│   │   ├── checks.py              # banned keyword AST check
│   │   └── parsers.py             # error type extraction
│   ├── rewards/
│   │   ├── spec_diag_score.py     # compute_score + format reward (routes all domains)
│   │   ├── reward_tracker.py      # RewardTracker + mastery/impossible tracking
│   │   └── math_grading.py        # \boxed{} extraction + LaTeX normalization
│   ├── dataset/
│   │   └── dynamic_dataset.py     # DynamicDataset Ray actor (buffer + eviction)
│   ├── trainer/
│   │   └── dynamic_grpo_trainer.py # DynamicGRPOTrainer + FeederThread + eviction
│   └── eval/
│       ├── cruxeval_dataset.py    # CRUXEval validation dataset
│       ├── math500_dataset.py     # MATH500 validation dataset
│       └── humaneval_dataset.py   # HumanEval validation dataset
├── scripts/
│   ├── step_1_vllm.sh             # Start vLLM (1 GPU)
│   ├── step2_grpo.sh              # Start training (2 GPU)
│   └── step2_grpo_3gpu.sh         # Start training (3 GPU)
└── verl/                          # Vendored verl framework
```
