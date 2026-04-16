"""CodeExecutor — concrete Executor for the Python code domain.

Wraps `PythonExecutor` (copied from AZR) and exposes the minimal
`Executor` interface used by DynamicDataset + reward manager.

Task schema (the dict passed around in this project for code tasks):

    {
        "domain": "code",
        "code":   "<ground-truth function body defining `def f(...)`>",
        "inputs": "<python literal args string, e.g. '1, 2' or '[3,4]'>",
        "gold_output": "<repr of f(inputs), str>",
        "imports": ["import math", ...],   # optional
        "capability_tags": ["sort", ...],  # optional, for generator memory
    }

A Student's response is expected to be a python literal answering either
the *input* or *output* prediction task. For Phase 0 smoke test we grade
**output prediction** (given code+inputs, predict repr of f(inputs)).
"""

from __future__ import annotations

from typing import Any

from spec_diag.executors.base import Executor
from spec_diag.executors.python_executor import PythonExecutor


class CodeExecutor(Executor):
    """Python code domain executor."""

    def __init__(
        self,
        timeout_length: int = 10,
        max_workers: int = 4,
        ast_check: bool = True,
    ) -> None:
        self._pyexec = PythonExecutor(
            timeout_length=timeout_length,
            max_workers=max_workers,
            ast_check=ast_check,
        )

    # Default banned keywords (same as AZR)
    BANNED_KEYWORDS = [
        "logging", "random", "multiprocessing", "pebble", "subprocess",
        "threading", "datetime", "time", "hashlib", "hmac", "bcrypt",
        "os.sys", "os.path", "sys.exit", "os.environ", "calendar",
    ]

    def check_validity(self, task: dict[str, Any]) -> bool:
        """Task is valid iff code parses, runs on inputs, is deterministic,
        and contains no banned keywords."""
        code = task.get("code")
        inputs = task.get("inputs")
        if not isinstance(code, str) or not isinstance(inputs, str):
            return False
        imports = task.get("imports") or []
        ok, _ = self._pyexec.check_all(
            code=code,
            inputs=inputs,
            imports=imports,
            banned_keywords=self.BANNED_KEYWORDS,
            check_determinism=True,
            check_error=False,
        )
        return bool(ok)

    def eval_student(self, task: dict[str, Any], response: str) -> float:
        """Grade Student output-prediction response against task['gold_output']."""
        code = task.get("code")
        gold = task.get("gold_output")
        if not isinstance(code, str) or not isinstance(gold, str):
            return 0.0
        imports = task.get("imports") or []
        agent_output = (response or "").strip()
        if not agent_output:
            return 0.0
        return float(
            self._pyexec.eval_output_prediction(
                code=code,
                gold_output=gold,
                agent_output=agent_output,
                imports=imports,
            )
            or 0.0
        )

    def compute_gold_output(self, task: dict[str, Any]) -> str | None:
        """Execute the task's code on its inputs and return repr string.

        Used by the Generator to fill in `gold_output` when it only emitted
        `(code, inputs)`. Returns None if the code errors.
        """
        code = task.get("code")
        inputs = task.get("inputs")
        if not isinstance(code, str) or not isinstance(inputs, str):
            return None
        imports = task.get("imports") or []
        output, status = self._pyexec.run_code(
            code=code, inputs=inputs, imports=imports
        )
        if "error" in status.lower() or not output:
            return None
        return output

    def close(self) -> None:
        self._pyexec.cleanup()
