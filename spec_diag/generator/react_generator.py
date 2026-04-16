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
import os
import re
from typing import TYPE_CHECKING, Any

from spec_diag.executors.code_executor import CodeExecutor

if TYPE_CHECKING:
    from spec_diag.generator.memory import GeneratorMemory


_COLD_START_SYSTEM = (
    "You design Python code reasoning tasks. Every task defines a pure "
    "deterministic function `def f(...)` and a concrete input. The student "
    "will be asked to predict `repr(f(input))`. Follow these rules:\n"
    "1. `f` must be a pure function of its arguments (no randomness, no IO, "
    "no network, no filesystem, no time).\n"
    "2. `f` must terminate in well under one second on the given input.\n"
    "3. The input must be a valid Python literal passable as positional args.\n"
    "4. Prefer simple algorithmic problems (string, list, arithmetic, dict).\n"
    "5. Output STRICT JSON only. No prose, no markdown fences."
)

_COLD_START_USER_TEMPLATE = (
    "Produce {n} distinct tasks as a JSON list. Each element must be an "
    "object with keys:\n"
    '  "code": string. Python source defining `def f(...): ...`. Can include '
    "helper defs and imports at the top.\n"
    '  "inputs": string. Python literal passed as positional args to f, e.g. '
    '"1, 2" or "[3, 1, 2]".\n'
    '  "capability_tags": list[string]. 1-3 short tags like '
    '"sort", "string", "arithmetic".\n'
    "Return ONLY the JSON list."
)


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

    # ---- Task generation ----

    def cold_start(self, n: int) -> list[dict[str, Any]]:
        """Generate `n` seed tasks with gold outputs computed locally."""
        raw = self._chat(
            system=_COLD_START_SYSTEM,
            user=_COLD_START_USER_TEMPLATE.format(n=n),
        )
        specs = _parse_json_list(raw)
        tasks: list[dict[str, Any]] = []
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            code = spec.get("code")
            inputs = spec.get("inputs")
            if not isinstance(code, str) or not isinstance(inputs, str):
                continue
            draft: dict[str, Any] = {
                "domain": "code",
                "code": code,
                "inputs": inputs,
                "imports": spec.get("imports") or [],
                "capability_tags": spec.get("capability_tags") or [],
            }
            if not self._executor.check_validity(draft):
                continue
            gold = self._executor.compute_gold_output(draft)
            if gold is None:
                continue
            draft["gold_output"] = gold
            tasks.append(draft)
            if len(tasks) >= n:
                break
        return tasks

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

        system = REACT_SYSTEM.format(n=n)
        user = REACT_USER_TEMPLATE.format(
            student_profile=ctx.get("student_profile") or "(no profile yet)",
            capability_summary=cap_text,
            weak_tags=weak_tags,
            strong_tags=strong_tags,
            failure_examples=ctx.get("recent_failures", "(none)"),
            n=n,
        )

        raw = self._chat(system=system, user=user)
        specs = _parse_json_list(raw)

        # Validate + compute gold_output (same logic as cold_start)
        tasks: list[dict[str, Any]] = []
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            code = spec.get("code")
            inputs = spec.get("inputs")
            if not isinstance(code, str) or not isinstance(inputs, str):
                continue
            draft: dict[str, Any] = {
                "domain": "code",
                "code": code,
                "inputs": inputs,
                "imports": spec.get("imports") or [],
                "capability_tags": spec.get("capability_tags") or [],
            }
            if not self._executor.check_validity(draft):
                continue
            gold = self._executor.compute_gold_output(draft)
            if gold is None:
                continue
            draft["gold_output"] = gold
            tasks.append(draft)
            if len(tasks) >= n:
                break

        # Fallback: if memory-conditioned generation yielded nothing
        if not tasks:
            import logging
            logging.getLogger(__name__).warning(
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
