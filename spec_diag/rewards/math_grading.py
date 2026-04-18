"""Math answer grading — extract \\boxed{} and compare symbolically.

Adapted from the MATH dataset evaluation conventions.
Handles LaTeX normalization, fraction/decimal comparison, etc.
"""

from __future__ import annotations

import re


def extract_boxed_answer(text: str) -> str:
    """Extract the last \\boxed{...} content from a response.

    Handles nested braces correctly.
    """
    if not text:
        return ""
    # Find all \boxed{ occurrences, take the last one
    idx = text.rfind("\\boxed{")
    if idx == -1:
        # Fallback: try to find the last number or expression
        return _fallback_extract(text)
    # Match balanced braces
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    # Unbalanced — return what we have
    return text[start:].strip()


def _fallback_extract(text: str) -> str:
    """If no \\boxed{}, try to extract the last line's answer."""
    lines = text.strip().split("\n")
    last = lines[-1].strip() if lines else ""
    # Strip common prefixes
    for prefix in ("Answer:", "The answer is", "= ", "answer:", "ANSWER:"):
        if last.lower().startswith(prefix.lower()):
            last = last[len(prefix):].strip()
    return last.rstrip(".").strip()


def normalize_math_answer(s: str) -> str:
    """Normalize a math answer string for comparison."""
    s = s.strip()
    # Remove LaTeX formatting — order matters: do frac BEFORE stripping braces
    # dfrac/frac → fraction (must be before generic brace removal)
    s = re.sub(r"\\dfrac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    # Remove \text{...}, \mathrm{...} etc. (keep content)
    s = re.sub(r"\\(mathrm|text|textbf|mathbf)\{([^}]*)\}", r"\2", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\ ", "")
    # Remove $
    s = s.replace("$", "")
    # \pi → pi, \sqrt → sqrt
    s = s.replace("\\pi", "pi").replace("\\sqrt", "sqrt")
    # Remove commas in numbers
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    # Remove trailing period
    s = s.rstrip(".")
    # Normalize whitespace
    s = " ".join(s.split())
    return s.strip()


def grade_math_answer(student_response: str, gold_answer: str) -> float:
    """Grade a student's math response against the gold answer.

    1. Extract \\boxed{} from student response
    2. Normalize both answers
    3. Compare (exact string, then numeric)

    Returns 1.0 if correct, 0.0 otherwise.
    """
    student_raw = extract_boxed_answer(student_response)
    student_norm = normalize_math_answer(student_raw)
    gold_norm = normalize_math_answer(gold_answer)

    # Exact string match after normalization
    if student_norm == gold_norm:
        return 1.0

    # Try numeric comparison
    try:
        # Replace pi with its value for numeric comparison
        s_eval = student_norm.replace("pi", "3.14159265358979")
        g_eval = gold_norm.replace("pi", "3.14159265358979")
        s_val = float(eval(s_eval))
        g_val = float(eval(g_eval))
        if abs(s_val - g_val) < 1e-4:
            return 1.0
    except Exception:
        pass

    # Try after removing spaces and comparing
    if student_norm.replace(" ", "") == gold_norm.replace(" ", ""):
        return 1.0

    return 0.0
