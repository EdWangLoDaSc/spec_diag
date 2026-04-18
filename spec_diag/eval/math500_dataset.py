"""MATH500 validation dataset for periodic math eval during training.

500 competition-level math problems from HuggingFaceH4/MATH-500.
Answers are LaTeX strings; grading uses symbolic comparison.

Data source: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
Requires: data/math500.jsonl (pre-downloaded)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MATH500_PROMPT = (
    "Solve the following math problem step by step. Show your reasoning, "
    "then put your final answer inside \\boxed{{}}.\n\n"
    "Problem: {problem}\n\n"
    "Solution:"
)


class MATH500Dataset:
    """Map-style dataset of MATH500 problems for verl validation."""

    def __init__(self, data_path: str | Path | None = None) -> None:
        if data_path is None:
            data_path = Path(__file__).resolve().parent.parent.parent / "data" / "math500.jsonl"
        else:
            data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"MATH500 data not found at {data_path}")

        self._problems: list[dict[str, Any]] = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._problems.append(json.loads(line))

        logger.info("MATH500Dataset: loaded %d problems", len(self._problems))

    def __len__(self) -> int:
        return len(self._problems)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        problem = self._problems[index]
        content = _MATH500_PROMPT.format(problem=problem["problem"])
        messages = [{"role": "user", "content": content}]

        return {
            "prompt": messages,
            "raw_prompt": messages,
            "data_source": "math500",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "answer": problem["answer"],
                    "subject": problem.get("subject", ""),
                    "unique_id": problem.get("unique_id", f"math500_{index}"),
                },
            },
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "extra_info": {"index": index},
            "index": index,
            "tools_kwargs": {},
            "interaction_kwargs": {},
        }
