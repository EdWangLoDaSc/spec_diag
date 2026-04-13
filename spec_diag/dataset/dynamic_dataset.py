"""DynamicDataset — Ray Actor that holds an evolving task buffer.

Phase 0. Stub.

Modeled after AZR's `DatasetManager`
(see `Absolute-Zero-Reasoner/absolute_zero_reasoner/trainer/ppo/azr_ray_trainer.py:246`).

Contract:
  - Generator calls `add_batch()` every round with new tasks + the current step
  - Trainer calls `sample_batch()` to pull training batches
  - Each buffered task carries a `step` stamp for recency-weighted sampling
  - Thread-safe: `threading.Lock` per buffer
"""

from __future__ import annotations

import threading
from copy import deepcopy
from typing import Any

import ray


@ray.remote
class DynamicDataset:
    """Ray Actor holding the evolving task buffer."""

    def __init__(self, max_size: int = 10_000) -> None:
        self._tasks: list[dict[str, Any]] = []
        self._steps: list[int] = []
        self._lock = threading.Lock()
        self._max_size = max_size

    def add_batch(self, tasks: list[dict[str, Any]], step: int) -> int:
        """Append new tasks with their generation step. Returns new buffer size."""
        raise NotImplementedError("Phase 0")

    def sample_batch(
        self, n: int, strategy: str = "uniform"
    ) -> list[dict[str, Any]]:
        """Sample n tasks. Strategies: 'uniform', 'recency_weighted', 'mixed'."""
        raise NotImplementedError("Phase 0")

    def get_recent(self, window: int) -> list[dict[str, Any]]:
        """Return tasks added in the last `window` steps."""
        raise NotImplementedError("Phase 0")

    def truncate(self, max_size: int) -> tuple[int, int]:
        """FIFO prune. Returns (size_before, size_after)."""
        raise NotImplementedError("Phase 0")

    def stats(self) -> dict[str, Any]:
        """Return buffer stats (size, step range, coverage summary)."""
        raise NotImplementedError("Phase 0")

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a deep copy of the full buffer (for testing/inspection)."""
        with self._lock:
            return deepcopy(self._tasks)
