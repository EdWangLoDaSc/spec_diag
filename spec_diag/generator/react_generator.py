"""ReAct Generator — frozen LLM that designs curriculum via in-context reasoning.

Phase 0 scope:
  - `cold_start(n)`: seed tasks before any Student feedback exists. Prompts
    the served vLLM model to emit N `{code, inputs}` specs as a JSON list.
    We execute each spec through the `CodeExecutor` to fill in `gold_output`
    and drop anything that fails validity.
  - `generate(memory, n)`: not implemented yet (Phase 1). See experiment_plan.md §2.2.

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
        """Phase 1: ReAct round conditioned on memory. Not yet implemented."""
        raise NotImplementedError(
            "ReActGenerator.generate is Phase 1 (memory-conditioned). "
            "Use cold_start for Phase 0 smoke tests."
        )

    def close(self) -> None:
        self._executor.close()


# ---- helpers ----


def _parse_json_list(text: str) -> list[Any]:
    """Best-effort extraction of a JSON list from a model response."""
    if not text:
        return []
    text = text.strip()
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
