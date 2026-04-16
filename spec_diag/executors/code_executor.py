"""CodeExecutor — concrete Executor for the Python code domain.

Wraps `PythonExecutor` (copied from AZR) and exposes the minimal
`Executor` interface used by DynamicDataset + reward manager.

Task schema (the dict passed around in this project for code tasks):

    {
        "domain": "code",
        "task_type": "code_o" | "code_i" | "code_e",
        "code":   "<ground-truth function body defining `def f(...)`>",
        "inputs": "<python literal args string, e.g. '1, 2' or '[3,4]'>",
        "gold_output": "<repr of f(inputs), str>",      # code_o / code_i
        "error_type": "ValueError" | "NoError" | ...,    # code_e only
        "imports": ["import math", ...],   # optional
        "capability_tags": ["sort", ...],  # optional, for generator memory
    }
"""

from __future__ import annotations

from typing import Any

from spec_diag.executors.base import Executor
from spec_diag.executors.parsers import parse_error
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

    # ---- validation ----

    def check_validity(self, task: dict[str, Any]) -> bool:
        """Task is valid iff code parses, runs on inputs, is deterministic,
        and contains no banned keywords."""
        code = task.get("code")
        inputs = task.get("inputs")
        if not isinstance(code, str) or not isinstance(inputs, str):
            return False
        imports = task.get("imports") or []
        task_type = task.get("task_type", "code_o")

        if task_type == "code_e":
            # For error tasks: code must parse and execute (may error),
            # but must be deterministic (same error or same output each run)
            ok, result = self._pyexec.check_all(
                code=code,
                inputs=inputs,
                imports=imports,
                banned_keywords=self.BANNED_KEYWORDS,
                check_determinism=True,
                check_error=True,
            )
            return bool(ok)
        else:
            # code_o / code_i: code must run successfully + be deterministic
            ok, _ = self._pyexec.check_all(
                code=code,
                inputs=inputs,
                imports=imports,
                banned_keywords=self.BANNED_KEYWORDS,
                check_determinism=True,
                check_error=False,
            )
            return bool(ok)

    # ---- gold answer computation ----

    def compute_gold_output(self, task: dict[str, Any]) -> str | None:
        """Execute the task's code on its inputs and return repr string.
        Returns None if the code errors."""
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

    def compute_error_type(self, task: dict[str, Any]) -> str | None:
        """Execute code and return error type string, or 'NoError'.
        Returns None if the code can't be parsed or is non-deterministic."""
        code = task.get("code")
        inputs = task.get("inputs")
        if not isinstance(code, str) or not isinstance(inputs, str):
            return None
        imports = task.get("imports") or []
        ok, result = self._pyexec.check_all(
            code=code,
            inputs=inputs,
            imports=imports,
            banned_keywords=self.BANNED_KEYWORDS,
            check_determinism=True,
            check_error=True,
        )
        if not ok:
            return None
        # result is "NoError" or an error type like "ValueError"
        return result

    # ---- student evaluation ----

    def eval_student(self, task: dict[str, Any], response: str) -> float:
        """Grade Student response based on task_type."""
        task_type = task.get("task_type", "code_o")
        if task_type == "code_o":
            return self._eval_output(task, response)
        elif task_type == "code_i":
            return self._eval_input(task, response)
        elif task_type == "code_e":
            return self._eval_error(task, response)
        return 0.0

    def _eval_output(self, task: dict[str, Any], response: str) -> float:
        """code_o: given code+input, student predicts output."""
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

    def _eval_input(self, task: dict[str, Any], response: str) -> float:
        """code_i: given code+output, student predicts a valid input."""
        code = task.get("code")
        gold_output = task.get("gold_output")
        if not isinstance(code, str) or not isinstance(gold_output, str):
            return 0.0
        imports = task.get("imports") or []
        agent_input = (response or "").strip()
        if not agent_input:
            return 0.0
        return float(
            self._pyexec.eval_input_prediction(
                code=code,
                gold_output=gold_output,
                agent_input=agent_input,
                imports=imports,
            )
            or 0.0
        )

    def _eval_error(self, task: dict[str, Any], response: str) -> float:
        """code_e: given code+input, student predicts error type."""
        gold_error = task.get("error_type")
        if not isinstance(gold_error, str):
            return 0.0
        # Extract first token from response (e.g., "ValueError: blah" → "ValueError")
        agent_error = (response or "").strip().split()[0].split(":")[0] if response else ""
        if agent_error.lower() == gold_error.lower():
            return 1.0
        return 0.0

    def close(self) -> None:
        self._pyexec.cleanup()
