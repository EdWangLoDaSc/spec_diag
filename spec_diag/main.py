"""spec_diag orchestrator entry point.

Phase 0 scaffold. Currently just confirms imports resolve; the real loop
described in `idea-iter/experiment_plan.md` §3.3 will be wired up in Phase 0.
"""

from __future__ import annotations


def main() -> None:
    # Import placeholders — just to confirm the package tree is wired up.
    from spec_diag.dataset.dynamic_dataset import DynamicDataset  # noqa: F401
    from spec_diag.executors.base import Executor  # noqa: F401
    from spec_diag.generator.react_generator import ReActGenerator  # noqa: F401
    from spec_diag.rewards.executor_reward import ExecutorRewardManager  # noqa: F401
    from spec_diag.trainer.dynamic_grpo_trainer import DynamicGRPOTrainer  # noqa: F401

    print("TODO: Phase 0 — implement DynamicDataset + PythonExecutor + GRPO smoke test")


if __name__ == "__main__":
    main()
