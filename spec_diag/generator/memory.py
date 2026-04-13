"""GeneratorMemory — persistent context across ReAct rounds.

Phase 1. Stub.

Holds:
  - task_history:          all tasks ever emitted + Student pass/fail outcomes
  - capability_trajectory: per-capability-tag pass-rate curves over time
  - recent_failures:       typical error samples (injected into the next prompt)
  - student_profile:       natural-language diagnosis, refreshed every K rounds
  - exemplar_pool:         high-quality ReAct chains for few-shot prompting (Phase 2)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GeneratorMemory:
    task_history: list[dict[str, Any]] = field(default_factory=list)
    capability_trajectory: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    recent_failures: list[dict[str, Any]] = field(default_factory=list)
    student_profile: str = ""
    exemplar_pool: list[dict[str, Any]] = field(default_factory=list)

    def update(self, performance_report: dict[str, Any]) -> None:
        """Merge a new Student performance report into memory."""
        raise NotImplementedError("Phase 1")

    def snapshot_prompt_context(self) -> dict[str, Any]:
        """Return the working-memory dict injected into the next ReAct prompt."""
        raise NotImplementedError("Phase 1")
