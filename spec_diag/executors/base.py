"""Abstract Executor interface.

An Executor is a domain-specific environment that (a) validates whether a
generated task is well-formed and solvable, and (b) grades a Student response
against ground truth.

Concrete implementations:
  - `PythonExecutor`    — Phase 0 (code domain)
  - `SQLExecutor`       — Phase 2 (SQL domain)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Executor(ABC):
    """Minimal Executor interface used by DynamicDataset + reward_manager."""

    @abstractmethod
    def check_validity(self, task: dict[str, Any]) -> bool:
        """Return True if the task is well-formed and solvable in this env."""

    @abstractmethod
    def eval_student(self, task: dict[str, Any], response: str) -> float:
        """Grade a Student response. Return a scalar reward in [0, 1]."""
