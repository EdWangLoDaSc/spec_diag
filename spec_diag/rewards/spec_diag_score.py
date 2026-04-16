"""Custom reward adapter for spec_diag_code data source.

Used via verl config:
  reward.custom_reward_function.path=pkg://spec_diag.rewards.spec_diag_score
  reward.custom_reward_function.name=compute_score
"""

from __future__ import annotations

import threading
from typing import Any

# Module-level singleton — avoids creating/destroying a ProcessPool on
# every reward call (256× per training step with GRPO n=8, bs=32).
_executor_lock = threading.Lock()
_executor = None


def _get_executor():
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                from spec_diag.executors.code_executor import CodeExecutor
                _executor = CodeExecutor(timeout_length=5, max_workers=2, ast_check=True)
    return _executor


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Score one rollout response.

    For `data_source="spec_diag_code"`, `ground_truth` is the task dict injected
    by `_task_to_sample` in `dynamic_grpo_trainer.py`.
    """
    if data_source == "humaneval":
        return _score_humaneval(ground_truth, solution_str)

    if data_source != "spec_diag_code":
        # Fallback to verl's default behavior for other data sources.
        from verl.utils.reward_score import default_compute_score

        return float(default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs))

    if not isinstance(ground_truth, dict):
        return 0.0

    executor = _get_executor()
    score = float(executor.eval_student(ground_truth, solution_str))

    # Phase 1: report to RewardTracker (fire-and-forget)
    _try_report(
        tags=ground_truth.get("capability_tags") or [],
        score=score,
        task=ground_truth,
        response=solution_str,
    )
    return score


# ---- HumanEval scoring via evalplus ----


def _score_humaneval(ground_truth: Any, solution_str: str) -> float:
    """Score a HumanEval solution using evalplus's check_correctness."""
    if not isinstance(ground_truth, dict):
        return 0.0
    try:
        from evalplus.eval._special_oracle import SPECIAL_ORACLE
        from evalplus.evaluate import check_correctness

        task_id = ground_truth.get("task_id", "")
        prompt = ground_truth.get("prompt", "")
        entry_point = ground_truth.get("entry_point", "")

        # Reconstruct full solution: prompt + student's completion
        solution = prompt + (solution_str or "")

        # Build the evalplus problem dict
        problem = {
            "task_id": task_id,
            "prompt": prompt,
            "entry_point": entry_point,
            "test": ground_truth.get("test", ""),
            "base_input": ground_truth.get("plus_input"),
            "plus_input": ground_truth.get("plus_input"),
            "atol": ground_truth.get("atol", 0),
        }

        result = check_correctness(
            dataset="humaneval",
            completion_id=(task_id, 0),
            problem=problem,
            solution=solution,
            expected_output=None,
            base_only=True,  # use base HumanEval tests (not plus)
            fast_check=True,
            identifier=task_id,
            min_time_limit=1.0,
            gt_time_limit_factor=4.0,
        )
        # result is a dict with "base_status" or "result"
        passed = result.get("base", [False])[0] if isinstance(result.get("base"), list) else False
        if not passed:
            # Try alternative result format
            passed = result.get("result", "") == "passed"
        return 1.0 if passed else 0.0
    except Exception:
        import logging
        logging.getLogger(__name__).debug(
            "humaneval scoring failed for %s", ground_truth.get("task_id", "?"),
            exc_info=True,
        )
        return 0.0


# ---- Phase 1: reward tracking ----

_tracker_handle = None
_tracker_miss_count = 0
_TRACKER_MAX_RETRIES = 20  # stop trying after N consecutive failures


def _try_report(
    tags: list[str],
    score: float,
    task: dict[str, Any] | None,
    response: str | None,
) -> None:
    """Report score to the named RewardTracker actor.  Never blocks or raises."""
    global _tracker_handle, _tracker_miss_count
    if _tracker_miss_count >= _TRACKER_MAX_RETRIES:
        return
    try:
        if _tracker_handle is None:
            import ray
            from spec_diag.rewards.reward_tracker import REWARD_TRACKER_NAME
            _tracker_handle = ray.get_actor(REWARD_TRACKER_NAME)
            _tracker_miss_count = 0  # found it — reset counter
        _tracker_handle.record.remote(tags, score, task, response)
    except Exception:
        _tracker_handle = None  # force re-lookup next time
        _tracker_miss_count += 1

