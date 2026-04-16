"""ReAct Generator — frozen LLM that designs curriculum via in-context reasoning.

  - `cold_start(n)`: seed tasks before any Student feedback exists. Prompts
    the served vLLM model to emit N `{code, inputs}` specs as a JSON list.
    We execute each spec through the `CodeExecutor` to fill in `gold_output`
    and drop anything that fails validity.
  - `generate(memory, n)`: memory-conditioned generation using ReAct prompts.
    Uses the Student's per-tag pass rates, failure examples, and natural-language
    profile to generate targeted tasks focusing on weak areas.

Talks to vLLM via the OpenAI-compatible HTTP API. Endpoint is taken from
`OPENAI_BASE_URL` (default `http://localhost:8000/v1`) and the served model
name is whatever vLLM's `--served-model-name` says (default `"generator"`).
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spec_diag.executors.code_executor import CodeExecutor

if TYPE_CHECKING:
    from spec_diag.generator.memory import GeneratorMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Banned keywords — checked via AST in CodeExecutor.check_validity
# ---------------------------------------------------------------------------

BANNED_KEYWORDS = [
    "logging", "random", "multiprocessing", "pebble", "subprocess",
    "threading", "datetime", "time", "hashlib", "hmac", "bcrypt",
    "os.sys", "os.path", "sys.exit", "os.environ", "calendar",
]

# ---------------------------------------------------------------------------
# Prompt templates — cold-start (Phase 0) and ReAct (Phase 1)
# ---------------------------------------------------------------------------

_COLD_START_SYSTEM = """\
You design Python code reasoning tasks. Every task defines a pure \
deterministic function `def f(...)` and a concrete input. The student \
will be asked to predict `repr(f(input))`.

### Code Requirements:
- Name the entry function `f` (e.g., `def f(...): ...`); nested defs inside `f` are allowed
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- Make the snippet require state tracking across multiple data \
transformations, ensuring the task requires long multi-step reasoning
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
  * These banned keywords: {banned_keywords}
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top

### Difficulty Guidelines:
Focus on algorithmic reasoning or logic complexity. Examples:
- Complex data structures: trees, heaps, stacks, queues, graphs
- Algorithms: dynamic programming, recursion, divide and conquer, \
greedy, backtracking, BFS/DFS
- Multi-step state transformations, nested loops with non-trivial logic
- String/list manipulations requiring careful index tracking

### Input Requirements:
- Provide exactly one test input for your function
- Format multiple arguments with commas between them
- Remember to add quotes around string arguments

Output STRICT JSON only. No prose, no markdown fences.\
"""

_COLD_START_USER_TEMPLATE = """\
{reference_section}\
Produce {n} distinct tasks as a JSON list. Each element must be an \
object with keys:
  "code": string. Python source defining `def f(...): ...`. Can include \
helper defs and imports at the top.
  "inputs": string. Python literal passed as positional args to f, e.g. \
"1, 2" or "[3, 1, 2]".
  "capability_tags": list[string]. 1-3 short tags describing the \
algorithmic capability tested, e.g. "recursion", "graph", "dp", \
"string", "tree", "backtracking", "greedy", "stack", "arithmetic".

Return ONLY the JSON list.\
"""


class ReActGenerator:
    """Frozen-LLM curriculum designer (inference-only)."""

    def __init__(self, model_name: str, config: dict[str, Any]) -> None:
        self.model_name = model_name
        self.config = config or {}
        self._executor = CodeExecutor(
            timeout_length=int(self.config.get("validity_timeout", 5)),
            max_workers=int(self.config.get("validity_workers", 4)),
            ast_check=True,
        )
        self._client = None  # lazy
        self._seed_snippets: list[dict[str, Any]] | None = None
        self._n_references = int(self.config.get("n_references", 6))

    # ---- OpenAI client ----

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai>=1.0 is required for ReActGenerator. "
                "pip install openai"
            ) from e
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

    # ---- Seed / reference snippets ----

    def _load_seeds(self) -> list[dict[str, Any]]:
        """Load seed snippets from JSONL file (lazy, cached)."""
        if self._seed_snippets is not None:
            return self._seed_snippets

        seed_path = self.config.get("seed_data_path")
        if not seed_path:
            self._seed_snippets = []
            return self._seed_snippets

        p = Path(seed_path)
        if not p.exists():
            logger.warning("seed_data_path %s not found; no references", p)
            self._seed_snippets = []
            return self._seed_snippets

        snippets = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "snippet" in entry and "input" in entry and "output" in entry:
                        snippets.append(entry)
                except json.JSONDecodeError:
                    continue
        logger.info("loaded %d seed snippets from %s", len(snippets), p)
        self._seed_snippets = snippets
        return self._seed_snippets

    def _sample_references(self, n: int | None = None) -> str:
        """Sample reference snippets and format them for prompt injection."""
        seeds = self._load_seeds()
        if not seeds:
            return ""

        k = min(n or self._n_references, len(seeds))
        chosen = random.sample(seeds, k)

        parts = ["\n### Reference Code Snippets (for style and difficulty guidance):\n"]
        for i, snip in enumerate(chosen):
            parts.append(
                f"<snippet_{i}>\n"
                f"```python\n{snip['snippet']}\n```\n"
                f"```input\n{snip['input']}\n```\n"
                f"```output\n{snip['output']}\n```\n"
                f"</snippet_{i}>\n"
            )
        parts.append(
            "Design tasks at a similar or higher difficulty level than "
            "these references. Your tasks must be sufficiently different "
            "from the provided snippets.\n\n"
        )
        return "".join(parts)

    # ---- Validation helper ----

    def _validate_specs(
        self, specs: list[Any], n: int,
    ) -> list[dict[str, Any]]:
        """Validate raw LLM specs → filter + compute gold_output."""
        tasks: list[dict[str, Any]] = []
        n_format_bad = 0
        n_validity_fail = 0
        n_gold_fail = 0
        for spec in specs:
            if not isinstance(spec, dict):
                n_format_bad += 1
                continue
            code = spec.get("code")
            inputs = spec.get("inputs")
            if not isinstance(code, str) or not isinstance(inputs, str):
                n_format_bad += 1
                continue
            draft: dict[str, Any] = {
                "domain": "code",
                "code": code,
                "inputs": inputs,
                "imports": spec.get("imports") or [],
                "capability_tags": spec.get("capability_tags") or [],
            }
            if not self._executor.check_validity(draft):
                n_validity_fail += 1
                continue
            gold = self._executor.compute_gold_output(draft)
            if gold is None:
                n_gold_fail += 1
                continue
            draft["gold_output"] = gold
            tasks.append(draft)
            if len(tasks) >= n:
                break
        total = len(specs)
        if total > 0:
            logger.info(
                "validate_specs: %d/%d passed (format_bad=%d, "
                "validity_fail=%d, gold_fail=%d)",
                len(tasks), total, n_format_bad, n_validity_fail, n_gold_fail,
            )
        return tasks

    # ---- Task generation ----

    def cold_start(self, n: int) -> list[dict[str, Any]]:
        """Generate `n` seed tasks with gold outputs computed locally."""
        system = _COLD_START_SYSTEM.format(
            banned_keywords=", ".join(BANNED_KEYWORDS),
        )
        user = _COLD_START_USER_TEMPLATE.format(
            n=n,
            reference_section=self._sample_references(),
        )
        raw = self._chat(system=system, user=user)
        specs = _parse_json_list(raw)
        return self._validate_specs(specs, n)

    def generate(self, memory: "GeneratorMemory", n: int) -> list[dict[str, Any]]:
        """Phase 1: generate tasks conditioned on Student performance memory."""
        from spec_diag.generator.prompts import REACT_SYSTEM, REACT_USER_TEMPLATE

        ctx = memory.snapshot_prompt_context()

        # Format capability summary as a table
        cap_summary = ctx.get("capability_summary", {})
        cap_text = "\n".join(
            f"  {tag}: {rate:.1%}"
            for tag, rate in sorted(cap_summary.items(), key=lambda x: x[1])
        ) or "(no data yet)"

        weak_tags = ", ".join(ctx.get("weak_tags", [])) or "(none identified)"
        strong_tags = ", ".join(ctx.get("strong_tags", [])) or "(none identified)"

        system = REACT_SYSTEM.format(
            n=n,
            banned_keywords=", ".join(BANNED_KEYWORDS),
        )
        user = REACT_USER_TEMPLATE.format(
            student_profile=ctx.get("student_profile") or "(no profile yet)",
            capability_summary=cap_text,
            weak_tags=weak_tags,
            strong_tags=strong_tags,
            failure_examples=ctx.get("recent_failures", "(none)"),
            reference_section=self._sample_references(),
            n=n,
        )

        raw = self._chat(system=system, user=user)
        specs = _parse_json_list(raw)
        tasks = self._validate_specs(specs, n)

        # Fallback: if memory-conditioned generation yielded nothing
        if not tasks:
            logger.warning(
                "generate(): 0 valid tasks from ReAct prompt; "
                "falling back to cold_start(%d)", n,
            )
            return self.cold_start(n)

        return tasks

    def close(self) -> None:
        self._executor.close()


# ---- helpers ----


def _parse_json_list(text: str) -> list[Any]:
    """Best-effort extraction of a JSON list from a model response."""
    if not text:
        return []
    text = text.strip()
    # strip Qwen3 <think>...</think> blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # strip ```json ... ``` fences if the model ignored instructions
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # try direct parse first
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # grab the outermost [...] block
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(parsed, list):
        return parsed
    return []
