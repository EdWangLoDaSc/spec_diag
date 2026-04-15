"""RayPPOTrainer wrapper that pulls training data from a DynamicDataset Ray actor.

Integration point: `verl/verl/trainer/ppo/ray_trainer.py` (`RayPPOTrainer`).
We do not subclass it — we construct one internally in `build()`, passing a
custom `train_dataset` that is a map-style `torch.utils.data.Dataset` backed
by a Ray actor.

Design (plan B: background feeder):
  - `DynamicMapDataset` is map-style with a fixed `__len__` (= samples_per_epoch).
    `__getitem__(i)` pulls one task from the Ray buffer via
    `sample_batch.remote(1, strategy)`. If the buffer is empty, it sleeps and
    retries up to `max_wait_s`, then raises. The index `i` is intentionally
    ignored — len() only exists to satisfy verl's dataloader assertions.
  - `_FeederThread` is a daemon thread that, while the trainer is alive,
    polls the buffer's size and, whenever it drops below `low_watermark`,
    calls `generator.cold_start(feed_batch)` and pushes the result via
    `dataset.add_batch.remote(tasks, current_step)`.
  - `DynamicGRPOTrainer.build()` does a synchronous warmup (enough cold_start
    tasks to cross `low_watermark`) *before* constructing `RayPPOTrainer`, so
    verl's own `assert len(dataloader) >= 1` and the very first training step
    never block on an empty buffer.
  - `fit()` runs verl's loop and, in a `finally`, stops the feeder thread.

Sample schema matches verl's `RLHFDataset.__getitem__`
(`verl/verl/utils/dataset/rl_dataset.py:359`): `prompt` is a list of chat
messages (not a string), plus `raw_prompt`, `dummy_tensor`, `extra_info`,
`index`, `tools_kwargs`, `interaction_kwargs`, `reward_model`. We also carry
`spec_diag_task` as a non-tensor field so `ExecutorRewardManager` can pick
it up from `non_tensor_batch`.

Imports from `verl` are deferred so scaffolding / unit tests don't need verl
installed.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = (
    "You are given a Python function and an input. Predict `repr(f(input))` "
    "exactly. Respond with only the predicted repr string, no prose.\n\n"
    "```python\n{code}\n```\n"
    "Input: `f({inputs})`\n"
    "Answer:"
)


def _task_to_sample(task: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Convert a spec_diag task dict into a verl-compatible dataset row.

    Mirrors `verl.utils.dataset.rl_dataset.RLHFDataset.__getitem__` output
    schema. Tokenization is deferred to verl's AgentLoop downstream; we only
    produce the chat-format messages list.
    """
    import torch

    content = _PROMPT_TEMPLATE.format(
        code=task.get("code", ""),
        inputs=task.get("inputs", ""),
    )
    messages = [{"role": "user", "content": content}]
    return {
        "prompt": messages,
        "raw_prompt": messages,
        "data_source": "spec_diag_code",
        "reward_model": {"style": "rule", "ground_truth": task},
        "spec_diag_task": task,
        # Required by verl's collate / downstream; mirror RLHFDataset fields:
        "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        "extra_info": {"index": index},
        "index": index,
        "tools_kwargs": {},
        "interaction_kwargs": {},
    }


class DynamicMapDataset:
    """Map-style torch Dataset backed by a `DynamicDataset` Ray actor.

    - `__len__` is a fixed, configured value (`samples_per_epoch`). It exists
      purely to satisfy torch DataLoader / verl's `_create_dataloader` assert.
    - `__getitem__(i)` pulls **one** task from the Ray actor on every call.
      The index `i` is ignored. If the buffer is empty, retries with backoff
      until `max_wait_s`; then raises RuntimeError so the failure is loud.
    - `num_workers` must be 0 on the dataloader — Ray actor handles do not
      survive being forked into DataLoader worker processes, and we want the
      feeder thread + __getitem__ to share a single buffer view.
    - Not a subclass of `torch.utils.data.Dataset`. PyTorch's DataLoader
      duck-types map-style datasets (needs only `__len__` + `__getitem__`),
      so we skip the base class to keep this module importable without
      torch for unit tests.
    """

    def __init__(
        self,
        dataset_handle: Any,
        samples_per_epoch: int,
        sample_strategy: str = "mixed",
        poll_interval_s: float = 0.5,
        max_wait_s: float = 300.0,
    ) -> None:
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self._handle = dataset_handle
        self._len = int(samples_per_epoch)
        self._strategy = sample_strategy
        self._poll_interval = float(poll_interval_s)
        self._max_wait = float(max_wait_s)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> dict[str, Any]:
        import ray

        waited = 0.0
        warned = False
        while True:
            tasks = ray.get(
                self._handle.sample_batch.remote(1, self._strategy)
            )
            if tasks:
                return _task_to_sample(tasks[0], index=int(index))
            if not warned:
                logger.warning(
                    "DynamicMapDataset: buffer empty at idx=%d, waiting for "
                    "feeder (poll=%.1fs, max_wait=%.1fs)",
                    index, self._poll_interval, self._max_wait,
                )
                warned = True
            if waited >= self._max_wait:
                raise RuntimeError(
                    f"DynamicMapDataset: buffer stayed empty for "
                    f"{self._max_wait:.1f}s at idx={index}. Feeder thread may "
                    f"be dead or generator.cold_start() is failing — check "
                    f"logs for feeder exceptions."
                )
            time.sleep(self._poll_interval)
            waited += self._poll_interval


class _FeederThread(threading.Thread):
    """Daemon thread that keeps the DynamicDataset buffer above a watermark.

    Every `poll_interval_s` it asks the Ray actor for its current size; if the
    buffer has dropped below `low_watermark` it calls
    `generator.cold_start(feed_batch)` and pushes the result. Exceptions in
    any single iteration are logged and swallowed — we never want the feeder
    to crash the trainer, but we do want them visible.
    """

    def __init__(
        self,
        generator: Any,
        dataset_handle: Any,
        feed_batch: int,
        low_watermark: int,
        poll_interval_s: float,
        step_provider: Callable[[], int] | None = None,
    ) -> None:
        super().__init__(name="spec_diag-feeder", daemon=True)
        self._generator = generator
        self._handle = dataset_handle
        self._feed_batch = int(feed_batch)
        self._low_watermark = int(low_watermark)
        self._poll_interval = float(poll_interval_s)
        self._step_provider = step_provider or (lambda: 0)
        self._stop_event = threading.Event()
        self._iter_count = 0
        self._fail_count = 0

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=timeout)

    def run(self) -> None:
        import ray

        logger.info(
            "spec_diag feeder: started (feed_batch=%d, low_watermark=%d, "
            "poll=%.1fs)",
            self._feed_batch, self._low_watermark, self._poll_interval,
        )
        while not self._stop_event.is_set():
            try:
                stats = ray.get(self._handle.stats.remote())
                size = int(stats.get("size", 0))
                if size < self._low_watermark:
                    step = int(self._step_provider())
                    tasks = self._generator.cold_start(self._feed_batch)
                    if tasks:
                        ray.get(
                            self._handle.add_batch.remote(tasks, step)
                        )
                        logger.info(
                            "spec_diag feeder: +%d tasks @step=%d (buffer "
                            "was %d, target≥%d)",
                            len(tasks), step, size, self._low_watermark,
                        )
                    else:
                        logger.warning(
                            "spec_diag feeder: cold_start returned 0 valid "
                            "tasks (all failed validity?)"
                        )
                self._iter_count += 1
            except Exception as e:  # noqa: BLE001
                self._fail_count += 1
                logger.exception(
                    "spec_diag feeder: iteration %d failed: %s",
                    self._iter_count, e,
                )
            # Interruptible sleep — wake immediately on stop().
            self._stop_event.wait(self._poll_interval)
        logger.info(
            "spec_diag feeder: stopped (iters=%d, failures=%d)",
            self._iter_count, self._fail_count,
        )


class DynamicGRPOTrainer:
    """Wrapper around verl's `RayPPOTrainer` with a dynamic-curriculum feeder.

    Usage sketch::

        trainer = DynamicGRPOTrainer(
            config=cfg,
            dynamic_dataset_handle=dataset_actor,
            generator=react_generator,
            tokenizer=tok,
            role_worker_mapping=...,
            resource_pool_manager=...,
        )
        trainer.fit()   # synchronous warmup → starts feeder → RayPPOTrainer.fit

    Config keys (under `cfg.spec_diag.*`):
        samples_per_epoch: int    — len() of the map-style dataset
        sample_strategy:  str    — "uniform" | "recency_weighted" | "mixed"
        feeder.feed_batch:      int   — tasks per cold_start call
        feeder.low_watermark:   int   — trigger cold_start when size < this
        feeder.poll_interval_s: float — seconds between feeder checks
        feeder.warmup_tasks:    int   — synchronous pre-fill before build
        feeder.getitem_max_wait_s: float — dataset blocking timeout
        feeder.getitem_poll_interval_s: float — dataset retry cadence while
            blocked on empty buffer (should be << poll_interval_s)
    """

    def __init__(
        self,
        config: Any,
        dynamic_dataset_handle: Any,
        generator: Any = None,
        tokenizer: Any = None,
        processor: Any = None,
        role_worker_mapping: Any = None,
        resource_pool_manager: Any = None,
        ray_worker_group_cls: Any = None,
        device_name: Optional[str] = None,
    ) -> None:
        self.config = config
        self.dynamic_dataset_handle = dynamic_dataset_handle
        self.generator = generator
        self.tokenizer = tokenizer
        self.processor = processor
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self._inner = None
        self._feeder: Optional[_FeederThread] = None

    # ---- config helpers ----

    def _sd_cfg(self, key: str, default: Any) -> Any:
        sd = getattr(self.config, "spec_diag", None)
        if sd is None:
            return default
        return sd.get(key, default) if hasattr(sd, "get") else getattr(sd, key, default)

    def _feeder_cfg(self, key: str, default: Any) -> Any:
        sd = getattr(self.config, "spec_diag", None)
        if sd is None:
            return default
        feeder = sd.get("feeder", None) if hasattr(sd, "get") else getattr(sd, "feeder", None)
        if feeder is None:
            return default
        return feeder.get(key, default) if hasattr(feeder, "get") else getattr(feeder, key, default)

    # ---- buffer warmup + feeder lifecycle ----

    def _warmup_buffer(self, n_tasks: int) -> int:
        """Synchronously fill the buffer with n_tasks cold-start tasks."""
        import ray

        if self.generator is None or n_tasks <= 0:
            return 0
        logger.info("spec_diag warmup: cold-starting %d tasks...", n_tasks)
        produced = 0
        # Do it in chunks to give the generator natural batching.
        chunk = int(self._feeder_cfg("feed_batch", 32))
        while produced < n_tasks:
            want = min(chunk, n_tasks - produced)
            tasks = self.generator.cold_start(want)
            if not tasks:
                logger.warning(
                    "spec_diag warmup: cold_start(%d) returned 0 valid "
                    "tasks; aborting warmup at %d/%d",
                    want, produced, n_tasks,
                )
                break
            ray.get(
                self.dynamic_dataset_handle.add_batch.remote(tasks, 0)
            )
            produced += len(tasks)
        logger.info("spec_diag warmup: buffer now has %d tasks", produced)
        return produced

    def _start_feeder(self) -> None:
        if self.generator is None:
            logger.warning(
                "spec_diag: no generator provided — feeder thread will NOT "
                "start. Buffer will drain and training will stall."
            )
            return
        if self._feeder is not None and self._feeder.is_alive():
            return
        self._feeder = _FeederThread(
            generator=self.generator,
            dataset_handle=self.dynamic_dataset_handle,
            feed_batch=int(self._feeder_cfg("feed_batch", 32)),
            low_watermark=int(self._feeder_cfg("low_watermark", 128)),
            poll_interval_s=float(self._feeder_cfg("poll_interval_s", 10.0)),
            step_provider=self._current_step,
        )
        self._feeder.start()

    def _stop_feeder(self) -> None:
        if self._feeder is not None:
            self._feeder.stop()
            self._feeder = None

    def _current_step(self) -> int:
        if self._inner is not None:
            return int(getattr(self._inner, "global_steps", 0))
        return 0

    # ---- build / fit ----

    def build(self):
        """Warm up buffer, start feeder, construct the RayPPOTrainer."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        warmup = int(self._feeder_cfg("warmup_tasks", 128))
        produced = self._warmup_buffer(warmup)
        low_wm = int(self._feeder_cfg("low_watermark", 128))
        if produced < low_wm:
            logger.warning(
                "spec_diag warmup: only produced %d tasks but low_watermark "
                "is %d — first few __getitem__ calls may block.",
                produced, low_wm,
            )

        self._start_feeder()

        samples_per_epoch = int(self._sd_cfg("samples_per_epoch", 1024))
        getitem_wait = float(self._feeder_cfg("getitem_max_wait_s", 300.0))
        getitem_poll = float(self._feeder_cfg("getitem_poll_interval_s", 0.5))
        train_dataset = DynamicMapDataset(
            dataset_handle=self.dynamic_dataset_handle,
            samples_per_epoch=samples_per_epoch,
            sample_strategy=str(self._sd_cfg("sample_strategy", "mixed")),
            poll_interval_s=getitem_poll,
            max_wait_s=getitem_wait,
        )
        # Validation is disabled in config (trainer.val_before_train=false,
        # test_freq=-1) but verl still asserts len(val_dataloader) >= 1. Pass
        # a tiny 1-sample dataset sharing the same buffer — if val ever does
        # run (e.g. via CLI override), it will pull real tasks from the same
        # DynamicDataset actor.
        val_dataset = DynamicMapDataset(
            dataset_handle=self.dynamic_dataset_handle,
            samples_per_epoch=1,
            sample_strategy=str(self._sd_cfg("sample_strategy", "mixed")),
            poll_interval_s=getitem_poll,
            max_wait_s=getitem_wait,
        )

        self._inner = RayPPOTrainer(
            config=self.config,
            tokenizer=self.tokenizer,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=self.resource_pool_manager,
            ray_worker_group_cls=self.ray_worker_group_cls or RayWorkerGroup,
            processor=self.processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=None,
            train_sampler=None,
            device_name=self.device_name,
        )
        return self._inner

    def init_workers(self) -> None:
        """Bring up the Ray worker groups (actor / ref / rollout / critic).

        Mirrors verl's own `main_ppo` pattern: `trainer.init_workers()` must
        happen between constructing `RayPPOTrainer` and calling `fit()`.
        """
        if self._inner is None:
            self.build()
        assert self._inner is not None
        self._inner.init_workers()

    def fit(self) -> None:
        if self._inner is None:
            self.build()
        assert self._inner is not None
        # Idempotent-ish: if the caller already called init_workers, verl's
        # RayPPOTrainer tolerates it being called twice in practice because
        # the method is usually guarded, but we still prefer the explicit
        # `init_workers() → fit()` order from verl's main_ppo. Callers that
        # have already initialized workers can skip this and call fit()
        # directly on `self._inner`.
        try:
            self._inner.fit()
        finally:
            self._stop_feeder()
