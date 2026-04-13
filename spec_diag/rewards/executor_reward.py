"""Executor-backed reward manager plug-in for verl.

Phase 0. Stub.

Wraps a `spec_diag.executors.Executor` as a verl-compatible reward function.
Integration point: verl's `verl/workers/reward_manager/registry.py` + abstract
base in `verl/workers/reward_manager/abstract.py`.

At Phase 0, we simply run `executor.eval_student(task, response)` and return
the scalar reward; no reward shaping beyond binary correct/incorrect.
"""

from __future__ import annotations

from typing import Any

from spec_diag.executors.base import Executor


class ExecutorRewardManager:
    """verl reward_manager adapter around an `Executor`."""

    def __init__(self, executor: Executor, config: dict[str, Any] | None = None) -> None:
        self.executor = executor
        self.config = config or {}

    def __call__(self, batch: Any) -> Any:
        """Compute rewards for a verl rollout batch. Returns verl-compatible output."""
        raise NotImplementedError("Phase 0")
