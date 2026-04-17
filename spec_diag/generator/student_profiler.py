"""Student profiler — turns raw training metrics into a natural-language diagnosis.

The profiler calls the Generator LLM to produce a short prose diagnosis of
the Student's current strengths and weaknesses. This prose is injected into
the ReAct Generator's prompt as ``student_profile``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_PROFILER_SYSTEM = (
    "You are a diagnostic expert analyzing a student AI model's performance "
    "on Python code reasoning tasks. Given the student's per-capability pass "
    "rates and example failures, write a concise 2-3 sentence diagnosis of "
    "the student's current strengths and weaknesses. Be specific about which "
    "capabilities are struggling and hypothesize why."
)

_PROFILER_USER_TEMPLATE = (
    "Student model: {model_name}\n\n"
    "Per-capability pass rates (recent window):\n{pass_rate_table}\n\n"
    "Trajectory deltas (improvement/decline per tag):\n{trajectory_deltas}\n\n"
    "Example failures:\n{failure_samples}\n\n"
    "Write a 2-3 sentence diagnosis."
)


def build_student_profile(
    performance_report: dict[str, Any],
    model_name: str,
) -> str:
    """Call the Generator LLM to produce a natural-language Student diagnosis."""
    from openai import OpenAI

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    generator_model = os.environ.get("SPEC_DIAG_MODEL", "generator")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Pass rate table
    per_tag = performance_report.get("per_tag_pass_rates", {})
    pass_rate_table = "\n".join(
        f"  {tag}: {rate:.1%}"
        for tag, rate in sorted(per_tag.items(), key=lambda x: x[1])
    ) or "(no data)"

    # Trajectory deltas
    trajectory = performance_report.get("capability_trajectory", {})
    deltas = []
    for tag, traj in trajectory.items():
        if len(traj) >= 2:
            delta = traj[-1] - traj[-2]
            sign = "+" if delta >= 0 else ""
            deltas.append(f"  {tag}: {sign}{delta:.1%}")
    trajectory_deltas = "\n".join(deltas) or "(insufficient data)"

    # Failure samples
    failures = performance_report.get("failures", [])[:6]
    failure_text = ""
    for i, f in enumerate(failures, 1):
        task = f.get("task", {})
        failure_text += (
            f"\n--- Failure {i} ---\n"
            f"Tags: {f.get('tags', [])}\n"
            f"Code: {task.get('code', '?')[:300]}\n"
            f"Expected: {task.get('gold_output', '?')}\n"
            f"Student: {f.get('response', '?')[:200]}\n"
        )
    failure_text = failure_text or "(no failures recorded)"

    user_msg = _PROFILER_USER_TEMPLATE.format(
        model_name=model_name,
        pass_rate_table=pass_rate_table,
        trajectory_deltas=trajectory_deltas,
        failure_samples=failure_text,
    )

    try:
        resp = client.chat.completions.create(
            model=generator_model,
            messages=[
                {"role": "system", "content": _PROFILER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Strip Qwen3 <think> blocks if present
        import re
        profile = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        logger.info(
            "student_profiler: generated profile (%d chars, raw=%d)",
            len(profile), len(raw),
        )
        return profile
    except Exception:
        logger.exception("student_profiler: LLM call failed")
        return ""
