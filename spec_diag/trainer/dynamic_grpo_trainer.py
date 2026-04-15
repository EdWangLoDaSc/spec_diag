"""RayPPOTrainer subclass that pulls training data from a DynamicDataset Ray Actor.

Integration point: `verl/verl/trainer/ppo/ray_trainer.py` (`RayPPOTrainer`).
We override only the dataloader-construction path. Everything else —
rollout, advantage estimation (GRPO), PPO update, checkpoints — is
unchanged from verl upstream.

Status: **scaffold**. The class wires `_create_dataloader` to an
IterableDataset backed by a Ray Actor handle to `DynamicDataset`.
To actually drive a training run you still need:
  1. A verl PPO config (`spec_diag/configs/spec_diag_grpo.yaml` inherits it).
  2. The usual `ResourcePoolManager`, role worker mapping, tokenizer, etc.
     — same bootstrap as `verl/trainer/main_ppo.py`.
  3. A `DynamicDataset` actor whose buffer is kept fed by a generator loop
     (see `spec_diag.main` for the Phase 0 feeding pattern).

Imports from `verl` are deferred so that scaffolding tests don't need verl
installed.
"""

from __future__ import annotations

from typing import Any, Iterator


def _make_dynamic_torch_dataset(
    dataset_handle: Any,
    tokenizer: Any,
    sample_strategy: str = "mixed",
    batch_size: int = 32,
) -> Any:
    """Build a torch `IterableDataset` that pulls tasks from the Ray actor.

    Each yielded sample mirrors verl's RL dataset schema:
      {
        "prompt":      str (tokenizer-ready chat-format or plain text),
        "data_source": str ("spec_diag_code"),
        "reward_model": {"ground_truth": <task dict>},
        "spec_diag_task": <task dict>,   # also top-level for our reward mgr
      }
    """
    import ray
    from torch.utils.data import IterableDataset

    class _DynamicTorchDataset(IterableDataset):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self._handle = dataset_handle
            self._strategy = sample_strategy
            self._batch = int(batch_size)

        def __iter__(self) -> Iterator[dict[str, Any]]:
            while True:
                tasks = ray.get(
                    self._handle.sample_batch.remote(self._batch, self._strategy)
                )
                if not tasks:
                    # Empty buffer — caller must feed it. Yield nothing and
                    # let the trainer back off at the dataloader level.
                    return
                for task in tasks:
                    yield _task_to_sample(task)

    return _DynamicTorchDataset()


def _task_to_sample(task: dict[str, Any]) -> dict[str, Any]:
    """Format a single task dict into a verl-compatible rollout sample."""
    prompt = (
        "You are given a Python function and an input. Predict `repr(f(input))` "
        "exactly. Respond with only the predicted repr string, no prose.\n\n"
        f"```python\n{task.get('code', '')}\n```\n"
        f"Input: `f({task.get('inputs', '')})`\n"
        "Answer:"
    )
    return {
        "prompt": prompt,
        "data_source": "spec_diag_code",
        "reward_model": {"ground_truth": task},
        "spec_diag_task": task,
    }


class DynamicGRPOTrainer:
    """Thin wrapper that produces a RayPPOTrainer bound to a DynamicDataset.

    Usage sketch::

        from spec_diag.trainer.dynamic_grpo_trainer import DynamicGRPOTrainer
        trainer = DynamicGRPOTrainer(config, dynamic_dataset_handle, ...)
        trainer.fit()   # delegates to RayPPOTrainer.fit()

    Not a `RayPPOTrainer` subclass to keep verl an optional dependency at
    import time; we construct one internally in `build`.
    """

    def __init__(
        self,
        config: Any,
        dynamic_dataset_handle: Any,
        tokenizer: Any = None,
        processor: Any = None,
        role_worker_mapping: Any = None,
        resource_pool_manager: Any = None,
        ray_worker_group_cls: Any = None,
        device_name: str | None = None,
        sample_strategy: str = "mixed",
    ) -> None:
        self.config = config
        self.dynamic_dataset_handle = dynamic_dataset_handle
        self.tokenizer = tokenizer
        self.processor = processor
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.sample_strategy = sample_strategy
        self._inner = None

    def build(self):
        """Construct and return the underlying verl RayPPOTrainer."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        train_dataset = _make_dynamic_torch_dataset(
            dataset_handle=self.dynamic_dataset_handle,
            tokenizer=self.tokenizer,
            sample_strategy=self.sample_strategy,
            batch_size=int(self.config.data.train_batch_size),
        )

        self._inner = RayPPOTrainer(
            config=self.config,
            tokenizer=self.tokenizer,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=self.resource_pool_manager,
            ray_worker_group_cls=self.ray_worker_group_cls or RayWorkerGroup,
            processor=self.processor,
            train_dataset=train_dataset,
            val_dataset=None,
            collate_fn=None,
            train_sampler=None,
            device_name=self.device_name,
        )
        return self._inner

    def fit(self) -> None:
        if self._inner is None:
            self.build()
        assert self._inner is not None
        self._inner.fit()
