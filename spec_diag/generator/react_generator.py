"""ReAct Generator — frozen LLM that designs curriculum via in-context reasoning.

Phase 1. Stub only — the Generator observes Student performance, reflects on
capability gaps, and outputs a batch of new tasks. No training involved.

See `idea-iter/experiment_plan.md` §2.2 for the full design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spec_diag.generator.memory import GeneratorMemory


class ReActGenerator:
    """Frozen-LLM curriculum designer.

    Inference-only. Executes an Observe → Think → Act chain per round using
    the current `GeneratorMemory` as context.
    """

    def __init__(self, model_name: str, config: dict[str, Any]) -> None:
        self.model_name = model_name
        self.config = config

    def cold_start(self, n: int) -> list[dict[str, Any]]:
        """Generate seed tasks before any Student feedback exists."""
        raise NotImplementedError("Phase 1")

    def generate(self, memory: "GeneratorMemory", n: int) -> list[dict[str, Any]]:
        """Run one ReAct round conditioned on memory, return n new tasks."""
        raise NotImplementedError("Phase 1")
