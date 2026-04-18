"""MathExecutor — executor for math reasoning tasks.

Math tasks have a natural-language problem statement and a hidden Python
verification function ``def f(): return <answer>``. The Student sees only
the problem; grading runs the verification code to compute the gold answer,
then compares with the Student's response.

Task schema:

    {
        "domain": "math",
        "task_type": "math_o",
        "problem": "What is the sum of the first 10 prime numbers?",
        "code": "def f(): return 2+3+5+7+11+13+17+19+23+29",
        "gold_answer": "129",
        "capability_tags": ["number_theory"],
    }
"""

from __future__ import annotations

import re
from typing import Any

from spec_diag.executors.python_executor import PythonExecutor


def _normalize_math_answer(s: str) -> str:
    """Normalize a math answer for comparison.

    Strips whitespace, $, \\boxed{}, commas in numbers, trailing periods.
    """
    s = s.strip()
    # Strip LaTeX wrappers
    s = re.sub(r"\\boxed\{(.*?)\}", r"\1", s)
    s = s.replace("$", "").strip()
    # Remove commas in numbers (e.g., "1,000" → "1000")
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    # Remove trailing period
    s = s.rstrip(".")
    return s.strip()


class MathExecutor:
    """Math domain executor. Uses PythonExecutor to verify answers."""

    BANNED_KEYWORDS = [
        "logging", "random", "multiprocessing", "pebble", "subprocess",
        "threading", "datetime", "time", "hashlib", "hmac", "bcrypt",
        "os.sys", "os.path", "sys.exit", "os.environ", "calendar",
    ]

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

    def check_validity(self, task: dict[str, Any]) -> bool:
        """Math task is valid iff verification code runs and returns a value."""
        code = task.get("code")
        if not isinstance(code, str) or "def f" not in code:
            return False
        ok, _ = self._pyexec.check_all(
            code=code,
            inputs="",
            imports=[],
            banned_keywords=self.BANNED_KEYWORDS,
            check_determinism=True,
            check_error=False,
        )
        return bool(ok)

    def compute_gold_answer(self, task: dict[str, Any]) -> str | None:
        """Run verification code and return the answer as a string."""
        code = task.get("code")
        if not isinstance(code, str):
            return None
        output, status = self._pyexec.run_code(
            code=code, inputs="", imports=[],
        )
        if "error" in status.lower() or not output:
            return None
        return _normalize_math_answer(output)

    def eval_student(self, task: dict[str, Any], response: str) -> float:
        """Grade Student's math answer against gold.

        Extracts \\boxed{} from CoT response, then normalizes and compares.
        """
        gold = task.get("gold_answer")
        if not isinstance(gold, str):
            return 0.0
        from spec_diag.rewards.math_grading import extract_boxed_answer, grade_math_answer
        # grade_math_answer handles boxed extraction + normalization + comparison
        return grade_math_answer(response or "", gold)

    def close(self) -> None:
        self._pyexec.cleanup()
