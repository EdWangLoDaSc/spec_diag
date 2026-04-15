"""Executor-backed reward manager plug-in for verl.

Implements verl's `AbstractRewardManager` interface. On every call it:
  1. Decodes each rollout's response to a string.
  2. Looks up the task dict from `non_tensor_batch["spec_diag_task"]`
     (injected by `DynamicGRPOTrainer` when it builds the batch).
  3. Calls `executor.eval_student(task, response)` → scalar in [0, 1].
  4. Writes the scalar at the final valid-response position, matching the
     convention used by `verl.workers.reward_manager.naive.NaiveRewardManager`.

Register as `"spec_diag_executor"` so verl configs can pick it up via
`reward_model.reward_manager=spec_diag_executor`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

from spec_diag.executors.base import Executor


def _register_with_verl():
    """Lazy registration: only try when verl is importable."""
    try:
        import torch  # noqa: F401
        from verl import DataProto  # noqa: F401
        from verl.workers.reward_manager import register
        from verl.workers.reward_manager.abstract import AbstractRewardManager
    except Exception:
        return None, None
    return register, AbstractRewardManager


_register, _AbstractRewardManager = _register_with_verl()


if _AbstractRewardManager is None:
    # verl not installed — expose a plain adapter so unit tests still import.
    class ExecutorRewardManager:  # type: ignore[no-redef]
        """Standalone executor reward manager (no verl dep).

        Used by the Phase 0 smoke test in `spec_diag.main`; the real verl-
        integrated version below activates whenever verl is importable.
        """

        def __init__(
            self,
            executor: Executor,
            tokenizer: Any = None,
            num_examine: int = 0,
            compute_score: Any = None,
            reward_fn_key: str = "data_source",
            **kwargs: Any,
        ) -> None:
            self.executor = executor
            self.tokenizer = tokenizer
            self.num_examine = num_examine
            self.reward_fn_key = reward_fn_key

        def score_one(self, task: dict[str, Any], response: str) -> float:
            return float(self.executor.eval_student(task, response))

else:

    @_register("spec_diag_executor")
    class ExecutorRewardManager(_AbstractRewardManager):  # type: ignore[misc]
        """verl reward_manager adapter around a `spec_diag.executors.Executor`."""

        def __init__(
            self,
            executor: Executor | None = None,
            tokenizer: Any = None,
            num_examine: int = 0,
            compute_score: Any = None,
            reward_fn_key: str = "data_source",
            **kwargs: Any,
        ) -> None:
            from spec_diag.executors.code_executor import CodeExecutor

            self.executor: Executor = executor or CodeExecutor()
            self.tokenizer = tokenizer
            self.num_examine = int(num_examine)
            self.reward_fn_key = reward_fn_key
            self._compute_score = compute_score  # unused; we have an executor

        def score_one(self, task: dict[str, Any], response: str) -> float:
            return float(self.executor.eval_student(task, response))

        def __call__(self, data, return_dict: bool = False):
            import torch

            cached = self._extract_reward_from_rm_scores(data, return_dict)
            if cached is not None:
                return cached

            reward_tensor = torch.zeros_like(
                data.batch["responses"], dtype=torch.float32
            )
            reward_extra_info: dict[str, list[Any]] = defaultdict(list)
            printed = 0

            for i in range(len(data)):
                item = data[i]
                prompt_ids = item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                response_ids = item.batch["responses"]
                valid_response_length = int(
                    item.batch["attention_mask"][prompt_length:].sum().item()
                )
                if valid_response_length <= 0:
                    continue
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=True
                ) if self.tokenizer is not None else ""

                task = item.non_tensor_batch.get("spec_diag_task")
                if not isinstance(task, dict):
                    # Fallback: try the verl-native ground_truth schema.
                    logger.warning(
                        "ExecutorRewardManager: sample %d missing "
                        "'spec_diag_task' in non_tensor_batch; falling back "
                        "to reward_model.ground_truth. If this happens every "
                        "step verl's collate may be dropping the field.",
                        i,
                    )
                    rm = item.non_tensor_batch.get("reward_model") or {}
                    task = rm.get("ground_truth") if isinstance(rm, dict) else None
                if not isinstance(task, dict):
                    logger.error(
                        "ExecutorRewardManager: sample %d has no task "
                        "(neither spec_diag_task nor reward_model.ground_truth"
                        "); scoring 0.0. This will silently break training "
                        "signal — investigate dataset schema.",
                        i,
                    )
                    reward = 0.0
                else:
                    reward = self.score_one(task, response_str)

                reward_tensor[i, valid_response_length - 1] = reward
                reward_extra_info["spec_diag_reward"].append(reward)

                if printed < self.num_examine:
                    printed += 1
                    print("[spec_diag][response]", response_str)
                    print("[spec_diag][reward]", reward)

            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            return reward_tensor
