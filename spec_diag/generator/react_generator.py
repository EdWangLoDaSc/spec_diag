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
deterministic function `def f(...)` and a concrete input.

There are FOUR task types you can create:
- "code_o": Student is given code + input, must predict the output. \
The function must be COMPLETE and runnable.
- "code_i": Student is given code + output, must deduce a valid input \
that produces that output. The function must be COMPLETE, FULLY \
IMPLEMENTED, and RUNNABLE — never use `pass` or leave the body empty. \
The difficulty lies in reverse-engineering the input from the output, \
which requires deep understanding of the algorithm.
- "code_e": Student is given code + input, must predict the error type. \
The code MUST ACTUALLY RAISE AN ERROR (e.g., TypeError, ValueError, \
IndexError, KeyError, ZeroDivisionError) when run with the given input. \
Do NOT generate code that runs successfully — the whole point is that \
the student must trace through the code to figure out which error occurs.
- "code_f": Student is given input/output pairs + a hint message, must \
deduce and write the function `def f(...)` that produces those outputs. \
Provide 4-6 diverse inputs via "inputs_list" and a short descriptive message. \
The function itself is the "hidden answer" — the student must reverse-engineer it.

### Code Requirements:
- Name the entry function `f` (e.g., `def f(...): ...`); nested defs inside `f` are allowed
- ALL functions must be COMPLETE and FULLY IMPLEMENTED — never use \
`pass`, `...`, `# TODO`, or stub implementations
- For code_o, code_i, code_f: the function must return a value successfully
- For code_e: the function must RAISE a specific error on the given input
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
Produce {n} distinct tasks as a JSON list. Mix task types: roughly \
40% code_o, 20% code_i, 20% code_e, 20% code_f.

For code_o, code_i, code_e, each element must have:
  "task_type": one of "code_o", "code_i", "code_e".
  "code": string. Python source defining `def f(...): ...`.
  "inputs": string. Python literal passed as positional args to f.
  "capability_tags": list[string]. 1-3 tags like "recursion", "graph", \
"dp", "string", "tree", "backtracking", "greedy", "stack".

For code_f, each element must have:
  "task_type": "code_f".
  "code": string. The hidden gold function `def f(...)`.
  "inputs_list": list[string]. 3-5 diverse inputs as Python literals.
  "message": string. A short hint describing what f does (without \
revealing the implementation).
  "capability_tags": list[string]. 1-3 tags.

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
        # How many tasks to ask for in a single LLM call. Small values keep each
        # response well under max_tokens; we fire multiple calls concurrently to
        # reach the requested batch size.
        self._tasks_per_call = int(self.config.get("tasks_per_call", 4))
        self._max_generation_workers = int(
            self.config.get("max_generation_workers", 8)
        )

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
        """Validate raw LLM specs → filter + dedup + compute gold answer.

        Supports task_type: "code_o" (default), "code_i", "code_e".
        - code_o / code_i: compute gold_output via execution
        - code_e: compute error_type via execution (may be "NoError")
        """
        tasks: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        n_format_bad = 0
        n_duplicate = 0
        n_validity_fail = 0
        n_gold_fail = 0
        for spec in specs:
            if not isinstance(spec, dict):
                n_format_bad += 1
                continue
            code = spec.get("code")
            task_type = spec.get("task_type", "code_o")
            if task_type not in ("code_o", "code_i", "code_e", "code_f"):
                task_type = "code_o"

            # code_f has different required fields
            if task_type == "code_f":
                inputs_list = spec.get("inputs_list")
                message = spec.get("message", "")
                if (not isinstance(code, str) or not isinstance(inputs_list, list)
                        or len(inputs_list) < 2):
                    n_format_bad += 1
                    continue
                key = (code.strip(), str(inputs_list))
                if key in seen:
                    n_duplicate += 1
                    continue
                seen.add(key)
                # Use first input for validity check
                draft: dict[str, Any] = {
                    "domain": "code",
                    "task_type": "code_f",
                    "code": code,
                    "inputs": inputs_list[0],
                    "imports": spec.get("imports") or [],
                    "capability_tags": spec.get("capability_tags") or [],
                    "message": message,
                }
                if not self._executor.check_validity(draft):
                    n_validity_fail += 1
                    continue
                io_pairs = self._executor.compute_io_pairs(
                    draft, inputs_list,
                )
                if io_pairs is None or len(io_pairs) < 2:
                    n_gold_fail += 1
                    continue
                draft["io_pairs"] = io_pairs
                draft.pop("inputs", None)
                tasks.append(draft)
                if len(tasks) >= n:
                    break
                continue

            # code_o / code_i / code_e
            inputs = spec.get("inputs")
            if not isinstance(code, str) or not isinstance(inputs, str):
                n_format_bad += 1
                continue
            # Deduplicate by (code, inputs)
            key = (code.strip(), inputs.strip())
            if key in seen:
                n_duplicate += 1
                continue
            seen.add(key)
            # Reject stub/incomplete functions
            code_lower = code.lower()
            if any(marker in code_lower for marker in
                   ("# todo", "# implement", "# your code", "notimplementederror")):
                n_validity_fail += 1
                continue
            # Reject if function body is just "pass" or "return -1" placeholder
            lines = [l.strip() for l in code.strip().split("\n") if l.strip()]
            body_lines = [l for l in lines if not l.startswith(("def ", "from ", "import ", "#", "@"))]
            if body_lines and body_lines[-1] in ("pass", "return -1", "return None", "..."):
                if task_type in ("code_i", "code_o"):
                    n_validity_fail += 1
                    continue

            draft = {
                "domain": "code",
                "task_type": task_type,
                "code": code,
                "inputs": inputs,
                "imports": spec.get("imports") or [],
                "capability_tags": spec.get("capability_tags") or [],
            }
            if not self._executor.check_validity(draft):
                n_validity_fail += 1
                continue

            if task_type == "code_e":
                error_type = self._executor.compute_error_type(draft)
                if error_type is None:
                    n_gold_fail += 1
                    continue
                # Reject code_e tasks that don't actually error
                if error_type == "NoError":
                    n_gold_fail += 1
                    continue
                draft["error_type"] = error_type
            else:
                # code_o and code_i both need gold_output
                gold = self._executor.compute_gold_output(draft)
                if gold is None:
                    n_gold_fail += 1
                    continue
                # Reject code_i tasks with trivial None output
                if task_type == "code_i" and gold in ("None", ""):
                    n_gold_fail += 1
                    continue
                draft["gold_output"] = gold

            tasks.append(draft)
            if len(tasks) >= n:
                break
        total = len(specs)
        if total > 0:
            type_counts = {}
            for t in tasks:
                tt = t.get("task_type", "code_o")
                type_counts[tt] = type_counts.get(tt, 0) + 1
            logger.info(
                "validate_specs: %d/%d passed (format_bad=%d, dup=%d, "
                "validity_fail=%d, gold_fail=%d, types=%s)",
                len(tasks), total, n_format_bad, n_duplicate,
                n_validity_fail, n_gold_fail, type_counts,
            )
        return tasks

    # ---- Task generation ----

    def _fanout_chat(
        self, system_fn, user_fn, n: int, label: str,
    ) -> list[Any]:
        """Issue ceil(n / tasks_per_call) concurrent _chat calls.

        system_fn / user_fn are callables that each produce a fresh prompt for
        a single call (so references can be re-sampled per call). They receive
        the per-call `n` as argument.

        Side-effect: populates ``self._last_chat_logs`` with per-call dicts
        ``{system, user, raw_response}`` for downstream saving.
        """
        from concurrent.futures import ThreadPoolExecutor

        per_call = max(1, self._tasks_per_call)
        n_calls = (n + per_call - 1) // per_call
        workers = max(1, min(self._max_generation_workers, n_calls))
        chat_logs: list[dict[str, str]] = [{}] * n_calls

        def _one_call(idx: int) -> list[Any]:
            k = min(per_call, n - idx * per_call)
            system = system_fn(k)
            user = user_fn(k)
            raw = self._chat(system=system, user=user)
            chat_logs[idx] = {
                "system": system,
                "user": user,
                "raw_response": raw,
            }
            logger.info(
                "%s[%d/%d] raw response (n=%d, len=%d chars):\n%s\n---END RAW---",
                label, idx + 1, n_calls, k, len(raw), raw,
            )
            return _parse_json_list(raw)

        specs: list[Any] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for chunk in pool.map(_one_call, range(n_calls)):
                specs.extend(chunk)
        self._last_chat_logs = chat_logs
        return specs

    def cold_start(self, n: int) -> list[dict[str, Any]]:
        """Generate `n` seed tasks with gold outputs computed locally."""
        def _system(_k: int) -> str:
            return _COLD_START_SYSTEM.format(
                banned_keywords=", ".join(BANNED_KEYWORDS),
            )

        def _user(k: int) -> str:
            return _COLD_START_USER_TEMPLATE.format(
                n=k,
                reference_section=self._sample_references(),
            )

        specs = self._fanout_chat(_system, _user, n, label="cold_start")
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

        def _system(k: int) -> str:
            return REACT_SYSTEM.format(
                n=k,
                banned_keywords=", ".join(BANNED_KEYWORDS),
            )

        # generate() prompt is longer than cold_start (adds memory context),
        # so use fewer references (3 vs 6) and truncate failures to stay
        # within context limit.
        failures_text = ctx.get("recent_failures", "(none)")
        if len(failures_text) > 1500:
            failures_text = failures_text[:1500] + "\n... (truncated)"

        difficulty_hint = ctx.get("difficulty_hint", "Maintain moderate difficulty.")

        def _user(k: int) -> str:
            return REACT_USER_TEMPLATE.format(
                student_profile=ctx.get("student_profile") or "(no profile yet)",
                capability_summary=cap_text,
                weak_tags=weak_tags,
                strong_tags=strong_tags,
                failure_examples=failures_text,
                reference_section=self._sample_references(n=3),
                difficulty_hint=difficulty_hint,
                n=k,
            )

        specs = self._fanout_chat(_system, _user, n, label="generate")
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
