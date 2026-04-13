"""RayPPOTrainer subclass that pulls training data from a DynamicDataset.

Phase 0. Stub.

Integration point: `verl/verl/trainer/ppo/ray_trainer.py` (RayPPOTrainer).
We override only the dataloader-construction path. Everything else —
rollout, advantage estimation (GRPO), PPO update, checkpoints — is
unchanged from verl upstream.
"""

from __future__ import annotations

from typing import Any

# NOTE: import deferred to avoid hard dep during scaffolding/import tests.
#   from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class DynamicGRPOTrainer:
    """Placeholder. Phase 0 will make this a real RayPPOTrainer subclass."""

    def __init__(self, config: Any, dynamic_dataset_handle: Any) -> None:
        self.config = config
        self.dynamic_dataset = dynamic_dataset_handle

    def fit(self) -> None:
        raise NotImplementedError("Phase 0")
