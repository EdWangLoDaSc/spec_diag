"""RewardTracker — named Ray actor for cross-process reward feedback.

Phase 1.  Created by SpecDiagTaskRunner, discovered by compute_score
via ``ray.get_actor(REWARD_TRACKER_NAME)``.

The tracker collects ``(tags, score, task_summary, response_snippet)``
tuples reported by ``compute_score`` and serves aggregated performance
reports to the feeder thread so it can update ``GeneratorMemory``.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

import ray

REWARD_TRACKER_NAME = "spec_diag_reward_tracker"


class RewardTrackerImpl:
    """Thread-safe reward tracker.  Pure Python — testable without Ray."""

    def __init__(
        self, max_failures_per_tag: int = 10, max_scores_per_tag: int = 2000,
        n_rollouts: int = 8,
    ) -> None:
        self._lock = threading.Lock()
        self._max_failures = int(max_failures_per_tag)
        self._max_scores = int(max_scores_per_tag)
        # per-tag accumulators (capped ring buffers)
        self._scores: dict[str, list[float]] = defaultdict(list)
        self._steps: dict[str, list[int]] = defaultdict(list)
        self._failures: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._current_step: int = 0
        self._total_recorded: int = 0
        self._n_rollouts: int = int(n_rollouts)
        # Per-task eviction tracking: key → consecutive streak count
        # key = (code[:100], inputs[:100]) or (code[:100], problem[:100])
        self._task_perfect_streak: dict[tuple[str, str], int] = {}  # all-correct streak
        self._task_zero_streak: dict[tuple[str, str], int] = {}     # all-wrong streak
        self._evicted_keys: set[tuple[str, str]] = set()

    # ---- write path (called from compute_score) ----

    def set_current_step(self, step: int) -> None:
        with self._lock:
            self._current_step = int(step)

    def record(
        self,
        tags: list[str],
        score: float,
        task: dict[str, Any] | None = None,
        response: str | None = None,
    ) -> None:
        with self._lock:
            step = self._current_step
            effective_tags = [_normalize_tag(t) for t in tags] if tags else ["_untagged"]
            for tag in effective_tags:
                scores = self._scores[tag]
                steps = self._steps[tag]
                scores.append(score)
                steps.append(step)
                # Cap per-tag score history to avoid unbounded growth
                if len(scores) > self._max_scores:
                    trim = len(scores) - self._max_scores
                    del scores[:trim]
                    del steps[:trim]
                if score < 1.0 and task is not None:
                    failures = self._failures[tag]
                    failures.append({
                        "tags": list(tags),
                        "task": _summarise_task(task),
                        "response": (response or "")[:300],
                        "score": score,
                        "step": step,
                    })
                    if len(failures) > self._max_failures:
                        del failures[: len(failures) - self._max_failures]
            # Per-task eviction tracking (mastered + impossible)
            if task is not None:
                task_key = _task_key(task)
                if score >= 1.0:
                    self._task_perfect_streak[task_key] = (
                        self._task_perfect_streak.get(task_key, 0) + 1
                    )
                    self._task_zero_streak[task_key] = 0
                elif score <= 0.0:
                    self._task_zero_streak[task_key] = (
                        self._task_zero_streak.get(task_key, 0) + 1
                    )
                    self._task_perfect_streak[task_key] = 0
                else:
                    self._task_perfect_streak[task_key] = 0
                    self._task_zero_streak[task_key] = 0
            self._total_recorded += 1

    # ---- read path (called from feeder thread) ----

    def get_report(self, since_step: int = 0) -> dict[str, Any]:
        with self._lock:
            per_tag_pass_rates: dict[str, float] = {}
            per_tag_counts: dict[str, int] = {}
            all_failures: list[dict[str, Any]] = []

            for tag in self._scores:
                recent = [
                    s for s, st in zip(self._scores[tag], self._steps[tag])
                    if st >= since_step
                ]
                if recent:
                    per_tag_pass_rates[tag] = sum(recent) / len(recent)
                    per_tag_counts[tag] = len(recent)

                all_failures.extend(
                    f for f in self._failures[tag] if f["step"] >= since_step
                )

            # sort failures by step descending so most recent come first
            all_failures.sort(key=lambda f: f["step"], reverse=True)

            return {
                "per_tag_pass_rates": per_tag_pass_rates,
                "per_tag_counts": per_tag_counts,
                "failures": all_failures,
                "total_tasks": self._total_recorded,
                "since_step": since_step,
            }

    def reset(self) -> None:
        with self._lock:
            self._scores.clear()
            self._steps.clear()
            self._failures.clear()
            self._total_recorded = 0
            self._current_step = 0

    def get_mastered_keys(self, threshold: int = 2) -> list[tuple[str, str]]:
        """Return task keys scored 1.0 for `threshold` consecutive batches.

        Each batch produces n_rollouts record() calls per task.
        threshold=2 means 2 × n_rollouts consecutive perfect scores.
        """
        with self._lock:
            min_streak = threshold * self._n_rollouts
            mastered = [
                k for k, count in self._task_perfect_streak.items()
                if count >= min_streak
            ]
            self._evicted_keys.update(mastered)
            return mastered

    def get_impossible_keys(self, threshold: int = 2) -> list[tuple[str, str]]:
        """Return task keys scored 0.0 for `threshold` consecutive batches.

        These tasks are too hard — the student never gets any rollout
        correct, so GRPO gets no useful gradient signal.
        """
        with self._lock:
            min_streak = threshold * self._n_rollouts
            impossible = [
                k for k, count in self._task_zero_streak.items()
                if count >= min_streak
            ]
            self._evicted_keys.update(impossible)
            return impossible

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_recorded": self._total_recorded,
                "current_step": self._current_step,
                "n_tags": len(self._scores),
                "tags": list(self._scores.keys()),
                "n_evicted": len(self._evicted_keys),
                "n_tracked_tasks": len(self._task_perfect_streak),
            }


@ray.remote
class RewardTracker(RewardTrackerImpl):
    """Named Ray actor wrapper.

    Create with::

        RewardTracker.options(
            name=REWARD_TRACKER_NAME,
        ).remote(max_failures_per_tag=12)
    """
    pass


# ---- helpers ----

def _task_key(task: dict[str, Any]) -> tuple[str, str]:
    """Compute a stable key for a task dict (for mastery tracking)."""
    code = task.get("code", "")[:100]
    inputs = task.get("inputs", task.get("problem", ""))[:100]
    return (code, inputs)


def _normalize_tag(tag: str) -> str:
    """Normalize capability tags to reduce fragmentation.

    'error handling', 'error_handling', 'error-handling' → 'error_handling'
    """
    return tag.strip().lower().replace(" ", "_").replace("-", "_")


def _summarise_task(task: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields needed for failure reporting (avoid bloat)."""
    return {
        "code": task.get("code", ""),
        "inputs": task.get("inputs", ""),
        "gold_output": task.get("gold_output", ""),
        "capability_tags": task.get("capability_tags", []),
    }
