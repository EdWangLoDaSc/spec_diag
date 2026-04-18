"""Math Generator — generates math reasoning tasks via LLM.

Independent from the code ReActGenerator. Uses a separate prompt template,
separate seed data, and separate validation pipeline.

Each math task has:
  - "problem": natural language math problem (shown to student)
  - "code": Python function `def f(): return <answer>` (hidden, for verification)
  - "gold_answer": computed by running the code

The Generator designs problems where the answer is verifiable by executing
a short Python function. This keeps the evaluation pipeline consistent
with the code domain (reuse PythonExecutor) while testing math reasoning.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spec_diag.executors.math_executor import MathExecutor

if TYPE_CHECKING:
    from spec_diag.generator.memory import GeneratorMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_MATH_SYSTEM = """\
You design math reasoning problems for an AI student. Each problem must:
1. Be a self-contained math question in natural language
2. Have a unique, deterministic numerical answer
3. Come with a hidden Python verification function `def f(): return <answer>`

### Problem Topics (vary across these):
- Arithmetic: multi-step calculations, order of operations
- Algebra: solve equations, simplify expressions, systems of equations
- Number theory: primes, divisibility, GCD/LCM, modular arithmetic
- Combinatorics: counting, permutations, combinations, probability
- Sequences: arithmetic/geometric progressions, recursive sequences
- Geometry: areas, volumes, angles, coordinate geometry

### Requirements:
- The problem must require multi-step reasoning (not just one operation)
- The Python function `def f()` takes NO arguments and returns the answer
- The function may use `math` module but no other imports
- The function must be deterministic and complete within 10 seconds
- The answer should be a number (integer, fraction, or decimal)
- Do NOT include the answer in the problem statement

### Banned in verification code:
{banned_keywords}

Output STRICT JSON only. No prose, no markdown fences.\
"""

_MATH_USER_TEMPLATE = """\
{reference_section}\
Produce {n} distinct math problems as a JSON list. Each element:
  "problem": string. Clear math problem in natural language.
  "code": string. Python `def f(): return <answer>`. May use `import math`.
  "capability_tags": list[string]. 1-3 tags like "algebra", "number_theory", \
"combinatorics", "probability", "geometry", "sequences", "arithmetic", \
"modular_arithmetic".

Return ONLY the JSON list.\
"""

_MATH_REACT_SYSTEM = """\
You are a curriculum designer for training an AI student on math reasoning.

[Observe] Read the student's performance and adapt.
[Think] Design problems targeting weak areas.
[Act] Output exactly {n} math problems as a JSON list.

### Requirements:
- Each problem must require multi-step reasoning
- Python function `def f()` (no args) returns the correct answer
- May use `import math`, no other imports
- Banned keywords: {banned_keywords}
- Focus 60-70% on weak capability areas
- Vary difficulty based on student performance

{difficulty_hint}

Output STRICT JSON only. No prose, no markdown fences.\
"""

_MATH_REACT_USER = """\
[Observe]
Student profile: {student_profile}

Capability pass rates:
{capability_summary}

Weak areas (pass rate < 50%): {weak_tags}
Strong areas (pass rate > 80%): {strong_tags}

Recent failures:
{failure_examples}

{reference_section}\
[Think]
Design {n} math problems targeting the student's weak areas.

[Act]
Output {n} problems as a JSON list. Each element:
{{"problem": "...", "code": "def f(): return ...", \
"capability_tags": ["...", ...]}}\
"""

BANNED_KEYWORDS = [
    "logging", "random", "multiprocessing", "pebble", "subprocess",
    "threading", "datetime", "time", "hashlib", "hmac", "bcrypt",
    "os.sys", "os.path", "sys.exit", "os.environ", "calendar",
]


class MathGenerator:
    """LLM-based math problem designer."""

    def __init__(self, model_name: str, config: dict[str, Any]) -> None:
        self.model_name = model_name
        self.config = config or {}
        self._executor = MathExecutor(
            timeout_length=int(self.config.get("validity_timeout", 10)),
            max_workers=int(self.config.get("validity_workers", 4)),
            ast_check=True,
        )
        self._client = None
        self._seed_problems: list[dict[str, Any]] | None = None
        self._n_references = int(self.config.get("n_references", 4))
        self._tasks_per_call = int(self.config.get("tasks_per_call", 4))
        self._max_workers = int(self.config.get("max_generation_workers", 8))

    # ---- OpenAI client ----

    def _get_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI
        base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "dummy")
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        return self._client

    def _chat(self, system: str, user: str) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=float(self.config.get("temperature", 0.7)),
            max_tokens=int(self.config.get("max_tokens", 4096)),
        )
        return resp.choices[0].message.content or ""

    # ---- Seed references ----

    def _load_seeds(self) -> list[dict[str, Any]]:
        if self._seed_problems is not None:
            return self._seed_problems
        seed_path = self.config.get("seed_data_path")
        if not seed_path:
            self._seed_problems = []
            return self._seed_problems
        p = Path(seed_path)
        if not p.exists():
            logger.warning("math seed_data_path %s not found", p)
            self._seed_problems = []
            return self._seed_problems
        problems = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "problem" in entry and "code" in entry:
                        problems.append(entry)
                except json.JSONDecodeError:
                    continue
        logger.info("loaded %d math seed problems from %s", len(problems), p)
        self._seed_problems = problems
        return self._seed_problems

    def _sample_references(self, n: int | None = None) -> str:
        seeds = self._load_seeds()
        if not seeds:
            return ""
        k = min(n or self._n_references, len(seeds))
        chosen = random.sample(seeds, k)
        parts = ["\n### Reference Math Problems:\n"]
        for i, p in enumerate(chosen):
            parts.append(
                f"<problem_{i}>\n"
                f"Problem: {p['problem']}\n"
                f"Verification: `{p['code']}`\n"
                f"Answer: {p.get('gold_answer', '?')}\n"
                f"</problem_{i}>\n"
            )
        parts.append("Design problems at similar or higher difficulty.\n\n")
        return "".join(parts)

    # ---- Validation ----

    def _validate_specs(self, specs: list[Any], n: int) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        seen: set[str] = set()
        n_fmt = n_dup = n_val = n_gold = 0
        for spec in specs:
            if not isinstance(spec, dict):
                n_fmt += 1
                continue
            problem = spec.get("problem")
            code = spec.get("code")
            if not isinstance(problem, str) or not isinstance(code, str):
                n_fmt += 1
                continue
            if len(problem.strip()) < 10:
                n_fmt += 1
                continue
            key = problem.strip()[:100]
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            draft = {
                "domain": "math",
                "task_type": "math_o",
                "problem": problem,
                "code": code,
                "capability_tags": spec.get("capability_tags") or [],
            }
            if not self._executor.check_validity(draft):
                n_val += 1
                continue
            gold = self._executor.compute_gold_answer(draft)
            if gold is None or gold in ("None", ""):
                n_gold += 1
                continue
            draft["gold_answer"] = gold
            tasks.append(draft)
            if len(tasks) >= n:
                break
        total = len(specs)
        if total > 0:
            logger.info(
                "math validate_specs: %d/%d passed (fmt=%d, dup=%d, "
                "val=%d, gold=%d)",
                len(tasks), total, n_fmt, n_dup, n_val, n_gold,
            )
        return tasks

    # ---- Fan-out chat ----

    def _fanout_chat(self, system_fn, user_fn, n: int, label: str) -> list[Any]:
        from concurrent.futures import ThreadPoolExecutor
        per_call = max(1, self._tasks_per_call)
        n_calls = (n + per_call - 1) // per_call
        workers = max(1, min(self._max_workers, n_calls))
        chat_logs: list[dict[str, str]] = [{}] * n_calls

        def _one_call(idx):
            k = min(per_call, n - idx * per_call)
            system = system_fn(k)
            user = user_fn(k)
            raw = self._chat(system=system, user=user)
            chat_logs[idx] = {"system": system, "user": user, "raw_response": raw}
            logger.info(
                "%s[%d/%d] raw response (n=%d, len=%d chars)",
                label, idx + 1, n_calls, k, len(raw),
            )
            return _parse_json_list(raw)

        specs: list[Any] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for chunk in pool.map(_one_call, range(n_calls)):
                specs.extend(chunk)
        self._last_chat_logs = chat_logs
        return specs

    # ---- Generation ----

    def cold_start(self, n: int) -> list[dict[str, Any]]:
        """Generate n math problems without student feedback."""
        def _sys(_k):
            return _MATH_SYSTEM.format(banned_keywords=", ".join(BANNED_KEYWORDS))
        def _usr(k):
            return _MATH_USER_TEMPLATE.format(
                n=k, reference_section=self._sample_references(),
            )
        specs = self._fanout_chat(_sys, _usr, n, label="math_cold_start")
        return self._validate_specs(specs, n)

    def generate(self, memory: "GeneratorMemory", n: int) -> list[dict[str, Any]]:
        """Generate n math problems conditioned on student performance."""
        ctx = memory.snapshot_prompt_context()
        cap_summary = ctx.get("capability_summary", {})
        cap_text = "\n".join(
            f"  {tag}: {rate:.1%}"
            for tag, rate in sorted(cap_summary.items(), key=lambda x: x[1])
        ) or "(no data yet)"
        weak_tags = ", ".join(ctx.get("weak_tags", [])) or "(none)"
        strong_tags = ", ".join(ctx.get("strong_tags", [])) or "(none)"
        difficulty_hint = ctx.get("difficulty_hint", "Maintain moderate difficulty.")
        failures_text = ctx.get("recent_failures", "(none)")
        if len(failures_text) > 1500:
            failures_text = failures_text[:1500] + "\n... (truncated)"

        def _sys(k):
            return _MATH_REACT_SYSTEM.format(
                n=k,
                banned_keywords=", ".join(BANNED_KEYWORDS),
                difficulty_hint=difficulty_hint,
            )
        def _usr(k):
            return _MATH_REACT_USER.format(
                student_profile=ctx.get("student_profile") or "(no profile yet)",
                capability_summary=cap_text,
                weak_tags=weak_tags,
                strong_tags=strong_tags,
                failure_examples=failures_text,
                reference_section=self._sample_references(n=3),
                n=k,
            )
        specs = self._fanout_chat(_sys, _usr, n, label="math_generate")
        tasks = self._validate_specs(specs, n)
        if not tasks:
            logger.warning("math generate(): 0 valid; falling back to cold_start(%d)", n)
            return self.cold_start(n)
        return tasks

    def close(self) -> None:
        self._executor.close()


def _parse_json_list(text: str) -> list[Any]:
    """Best-effort extraction of a JSON list from LLM response."""
    if not text:
        return []
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    return parsed if isinstance(parsed, list) else []
