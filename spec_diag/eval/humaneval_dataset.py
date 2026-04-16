"""HumanEval validation dataset for periodic eval during training.

Uses evalplus to load HumanEval problems and formats them as a
verl-compatible map-style dataset. Plugs into RayPPOTrainer's
val_dataset slot so ``_validate()`` runs HumanEval automatically
every ``test_freq`` steps.

Requires: ``pip install evalplus``
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _format_prompt(problem: dict[str, Any]) -> str:
    """Format a HumanEval problem as a completion prompt."""
    # evalplus problem has: task_id, prompt, entry_point, canonical_solution, test, ...
    # The prompt already contains the function signature + docstring
    return (
        "Complete the following Python function. Output ONLY the function "
        "body (no explanation, no markdown fences).\n\n"
        f"{problem['prompt']}"
    )


class HumanEvalDataset:
    """Map-style dataset of HumanEval problems for verl validation.

    ``__getitem__`` returns a dict matching verl's RLHFDataset schema:
    ``prompt`` (chat messages), ``data_source``, ``reward_model``, etc.

    Usage::

        val_dataset = HumanEvalDataset()
        # pass to RayPPOTrainer(val_dataset=val_dataset, ...)
    """

    def __init__(self) -> None:
        from evalplus.data import get_human_eval_plus
        self._problems = list(get_human_eval_plus().values())
        logger.info("HumanEvalDataset: loaded %d problems", len(self._problems))

    def __len__(self) -> int:
        return len(self._problems)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        problem = self._problems[index]
        content = _format_prompt(problem)
        messages = [{"role": "user", "content": content}]

        return {
            "prompt": messages,
            "raw_prompt": messages,
            "data_source": "humaneval",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "task_id": problem["task_id"],
                    "prompt": problem["prompt"],
                    "entry_point": problem["entry_point"],
                    "test": problem.get("test", ""),
                    "plus_input": problem.get("plus_input"),
                    "plus": problem.get("plus"),
                },
            },
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "extra_info": {"index": index},
            "index": index,
            "tools_kwargs": {},
            "interaction_kwargs": {},
        }
