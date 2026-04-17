"""GeneratorMemory — persistent context across ReAct rounds.

Holds:
  - task_history:          per-round summary of pass rates
  - capability_trajectory: per-capability-tag pass-rate curves over time
  - recent_failures:       example failures (injected into the next prompt)
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

    # ---- limits ----
    _max_history: int = 100
    _max_failures: int = 30

    def update(self, performance_report: dict[str, Any]) -> None:
        """Merge a new performance report (from RewardTracker) into memory."""
        per_tag = performance_report.get("per_tag_pass_rates", {})

        # 1. Update capability trajectory
        for tag, rate in per_tag.items():
            self.capability_trajectory[tag].append(rate)

        # 2. Update recent failures (deduplicate by code+inputs)
        existing_keys = {
            (f.get("task", {}).get("code", ""), f.get("task", {}).get("inputs", ""))
            for f in self.recent_failures
        }
        for f in performance_report.get("failures", []):
            key = (f.get("task", {}).get("code", ""), f.get("task", {}).get("inputs", ""))
            if key not in existing_keys:
                self.recent_failures.append(f)
                existing_keys.add(key)
        if len(self.recent_failures) > self._max_failures:
            self.recent_failures = self.recent_failures[-self._max_failures:]

        # 3. Append summary to task history
        self.task_history.append({
            "per_tag_pass_rates": per_tag,
            "total_tasks": performance_report.get("total_tasks", 0),
        })
        if len(self.task_history) > self._max_history:
            self.task_history = self.task_history[-self._max_history:]

    def snapshot_prompt_context(self) -> dict[str, Any]:
        """Return the working-memory dict injected into the next ReAct prompt.

        Only includes tags with enough samples (per_tag_counts >= 3) to
        avoid noisy 0% / 100% tags from single observations. Caps the
        summary at 30 tags to keep the prompt manageable.
        """
        weak_tags: list[str] = []
        strong_tags: list[str] = []
        capability_summary: dict[str, float] = {}

        # Filter: only tags with enough data points across all updates
        per_tag_total_samples: dict[str, int] = {}
        for entry in self.task_history:
            for tag in entry.get("per_tag_pass_rates", {}):
                per_tag_total_samples[tag] = per_tag_total_samples.get(tag, 0) + 1

        for tag, trajectory in self.capability_trajectory.items():
            if not trajectory:
                continue
            # Skip tags with fewer than 3 observations
            if per_tag_total_samples.get(tag, 0) < 3:
                continue
            latest = trajectory[-1]
            capability_summary[tag] = latest
            if latest < 0.5:
                weak_tags.append(tag)
            elif latest > 0.8:
                strong_tags.append(tag)

        # Cap at 30 tags sorted by pass rate (show weakest first)
        if len(capability_summary) > 30:
            sorted_tags = sorted(capability_summary.items(), key=lambda x: x[1])
            capability_summary = dict(sorted_tags[:30])

        # Format recent failures as readable text
        formatted: list[str] = []
        for f in self.recent_failures[-9:]:
            task = f.get("task", {})
            formatted.append(
                f"Tags: {f.get('tags', [])}\n"
                f"Code:\n{task.get('code', '?')}\n"
                f"Input: f({task.get('inputs', '?')})\n"
                f"Expected: {task.get('gold_output', '?')}\n"
                f"Student answered: {f.get('response', '?')[:200]}\n"
                f"Score: {f.get('score', 0.0)}"
            )

        return {
            "student_profile": self.student_profile,
            "capability_trajectory": {
                tag: traj[-5:] for tag, traj in self.capability_trajectory.items()
            },
            "recent_failures": "\n---\n".join(formatted) if formatted else "(none)",
            "weak_tags": weak_tags,
            "strong_tags": strong_tags,
            "capability_summary": capability_summary,
        }
