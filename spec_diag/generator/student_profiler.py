"""Student profiler — turns raw training metrics into a natural-language diagnosis.

Phase 1. Stub.

The profiler is itself an LLM call: it reads recent performance stats
(pass rates, trajectory deltas, failure samples) and writes a short prose
diagnosis of the Student's current strengths and weaknesses. This prose is
then injected into the ReAct Generator's prompt as `student_profile`.
"""

from __future__ import annotations

from typing import Any


def build_student_profile(performance_report: dict[str, Any], model_name: str) -> str:
    """Call an LLM to produce a natural-language Student diagnosis."""
    raise NotImplementedError("Phase 1")
