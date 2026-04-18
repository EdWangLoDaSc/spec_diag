"""DynamicDataset — Ray Actor that holds an evolving task buffer.

Modeled after AZR's `DatasetManager`
(see `Absolute-Zero-Reasoner/absolute_zero_reasoner/trainer/ppo/azr_ray_trainer.py:246`).

Contract:
  - Generator calls `add_batch()` every round with new tasks + the current step
  - Trainer calls `sample_batch()` to pull training batches
  - Each buffered task carries a `step` stamp for recency-weighted sampling
  - Thread-safe: `threading.Lock` per buffer
"""

from __future__ import annotations

import random
import threading
from copy import deepcopy
from typing import Any

import ray


class DynamicDatasetImpl:
    """Thread-safe task buffer. Pure-Python so we can unit test without Ray."""

    def __init__(self, max_size: int = 10_000, seed: int | None = None) -> None:
        self._tasks: list[dict[str, Any]] = []
        self._steps: list[int] = []
        self._lock = threading.Lock()
        self._max_size = int(max_size)
        self._rng = random.Random(seed)

    def add_batch(self, tasks: list[dict[str, Any]], step: int) -> int:
        with self._lock:
            for t in tasks:
                self._tasks.append(deepcopy(t))
                self._steps.append(int(step))
            if len(self._tasks) > self._max_size:
                overflow = len(self._tasks) - self._max_size
                del self._tasks[:overflow]
                del self._steps[:overflow]
            return len(self._tasks)

    def sample_batch(
        self, n: int, strategy: str = "uniform"
    ) -> list[dict[str, Any]]:
        with self._lock:
            size = len(self._tasks)
            if size == 0 or n <= 0:
                return []
            n = min(n, size)
            if strategy == "uniform":
                idxs = self._rng.sample(range(size), n)
            elif strategy == "recency_weighted":
                max_step = max(self._steps)
                # weight = 1 + (step - min_step); more recent → larger weight
                min_step = min(self._steps)
                weights = [1.0 + (s - min_step) for s in self._steps]
                idxs = self._weighted_sample(weights, n)
                _ = max_step  # noqa: F841
            elif strategy == "mixed":
                half = n // 2
                idxs_u = self._rng.sample(range(size), min(half, size))
                remaining = n - len(idxs_u)
                if remaining > 0:
                    min_step = min(self._steps)
                    weights = [1.0 + (s - min_step) for s in self._steps]
                    idxs_r = self._weighted_sample(weights, remaining)
                    idxs = idxs_u + idxs_r
                else:
                    idxs = idxs_u
            else:
                raise ValueError(f"unknown sampling strategy: {strategy}")
            return [deepcopy(self._tasks[i]) for i in idxs]

    def _weighted_sample(self, weights: list[float], n: int) -> list[int]:
        # sample w/o replacement by repeatedly drawing with weights
        pool = list(range(len(weights)))
        w = list(weights)
        out: list[int] = []
        for _ in range(min(n, len(pool))):
            total = sum(w)
            if total <= 0:
                idx = self._rng.randrange(len(pool))
            else:
                r = self._rng.random() * total
                acc = 0.0
                idx = 0
                for i, wi in enumerate(w):
                    acc += wi
                    if acc >= r:
                        idx = i
                        break
            out.append(pool[idx])
            del pool[idx]
            del w[idx]
        return out

    def get_recent(self, window: int) -> list[dict[str, Any]]:
        with self._lock:
            if window <= 0 or not self._steps:
                return []
            max_step = max(self._steps)
            cutoff = max_step - int(window) + 1
            return [
                deepcopy(t)
                for t, s in zip(self._tasks, self._steps)
                if s >= cutoff
            ]

    def truncate(self, max_size: int) -> tuple[int, int]:
        with self._lock:
            before = len(self._tasks)
            if before > max_size:
                overflow = before - int(max_size)
                del self._tasks[:overflow]
                del self._steps[:overflow]
            self._max_size = int(max_size)
            return before, len(self._tasks)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            size = len(self._tasks)
            if size == 0:
                return {"size": 0, "min_step": None, "max_step": None, "tag_counts": {}}
            tag_counts: dict[str, int] = {}
            for t in self._tasks:
                for tag in t.get("capability_tags", []) or []:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            return {
                "size": size,
                "min_step": min(self._steps),
                "max_step": max(self._steps),
                "tag_counts": tag_counts,
            }

    def evict_mastered(self, keys: list[tuple[str, str]]) -> int:
        """Remove tasks whose (code[:100], inputs[:100]) key is in `keys`.

        Returns the number of tasks evicted.
        """
        if not keys:
            return 0
        key_set = set(keys)
        with self._lock:
            before = len(self._tasks)
            keep = []
            keep_steps = []
            for t, s in zip(self._tasks, self._steps):
                code = t.get("code", "")[:100]
                inputs = t.get("inputs", t.get("problem", ""))[:100]
                if (code, inputs) not in key_set:
                    keep.append(t)
                    keep_steps.append(s)
            self._tasks = keep
            self._steps = keep_steps
            return before - len(self._tasks)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._tasks)

    def save(self, path: str) -> None:
        """Persist buffer to JSON file (thread-safe)."""
        import json
        import os
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            data = {
                "tasks": self._tasks,
                "steps": self._steps,
                "max_size": self._max_size,
            }

        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> int:
        """Load buffer from JSON file (thread-safe). Returns number of tasks loaded."""
        import json
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            return 0

        with open(p, encoding="utf-8") as f:
            data = json.load(f)

        with self._lock:
            self._tasks = data.get("tasks", [])
            self._steps = data.get("steps", [])
            # Preserve current max_size unless file has a larger one
            file_max_size = data.get("max_size", self._max_size)
            self._max_size = max(self._max_size, file_max_size)

        return len(self._tasks)


@ray.remote
class DynamicDataset(DynamicDatasetImpl):
    """Ray Actor wrapper around DynamicDatasetImpl."""

    pass
