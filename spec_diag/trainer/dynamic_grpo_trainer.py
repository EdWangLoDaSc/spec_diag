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
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


_PROMPT_CODE_O = (
    "You are given a Python function and an input. Predict `repr(f(input))` "
    "exactly. Respond with only the predicted repr string, no prose.\n\n"
    "```python\n{code}\n```\n"
    "Input: `f({inputs})`\n"
    "Answer:"
)

_PROMPT_CODE_I = (
    "You are given a Python function and its output. Provide one possible "
    "input that produces this output. Format: comma-separated positional "
    "args (quote strings). Respond with only the input, no prose.\n\n"
    "```python\n{code}\n```\n"
    "Output: `{gold_output}`\n"
    "Input:"
)

_PROMPT_CODE_E = (
    "You are given a Python function and an input. Deduce the error type "
    "that will be raised when the code is executed with this input. If "
    'there are no errors, answer "NoError". Respond with only the error '
    "type name (e.g., ValueError, TypeError, NoError), no prose.\n\n"
    "```python\n{code}\n```\n"
    "Input: `f({inputs})`\n"
    "Error type:"
)


def _task_to_sample(task: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Convert a spec_diag task dict into a verl-compatible dataset row.

    Mirrors `verl.utils.dataset.rl_dataset.RLHFDataset.__getitem__` output
    schema. Tokenization is deferred to verl's AgentLoop downstream; we only
    produce the chat-format messages list.

    Supports task_type: "code_o" (output prediction, default),
    "code_i" (input prediction), "code_e" (error prediction).
    """
    import torch

    task_type = task.get("task_type", "code_o")
    if task_type == "code_i":
        content = _PROMPT_CODE_I.format(
            code=task.get("code", ""),
            gold_output=task.get("gold_output", ""),
        )
    elif task_type == "code_e":
        content = _PROMPT_CODE_E.format(
            code=task.get("code", ""),
            inputs=task.get("inputs", ""),
        )
    else:
        content = _PROMPT_CODE_O.format(
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


class _CheckpointThread(threading.Thread):
    """Daemon thread that periodically saves DynamicDataset buffer to disk."""

    def __init__(
        self,
        dataset_handle: Any,
        save_dir: Path,
        interval_s: float = 300.0,
        step_provider: Callable[[], int] | None = None,
    ) -> None:
        super().__init__(name="spec_diag-checkpoint", daemon=True)
        self._handle = dataset_handle
        self._save_dir = Path(save_dir)
        self._interval = float(interval_s)
        self._step_provider = step_provider or (lambda: 0)
        self._stop_event = threading.Event()
        self._save_count = 0
        self._fail_count = 0
        self._save_dir.mkdir(parents=True, exist_ok=True)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=timeout)

    def run(self) -> None:
        import ray

        logger.info(
            "spec_diag checkpoint: started (dir=%s, interval=%.1fs)",
            self._save_dir, self._interval,
        )
        while not self._stop_event.is_set():
            try:
                step = int(self._step_provider())
                # Save as buffer.json (overwrite) + buffer_step_<step>.json (versioned)
                buf_path = self._save_dir / "buffer.json"
                version_path = self._save_dir / f"buffer_step_{step}.json"

                # Use ray actor's save method
                ray.get(self._handle.save.remote(str(buf_path)))

                # Also keep a versioned copy (keep at most 3)
                ray.get(self._handle.save.remote(str(version_path)))
                self._cleanup_old_checkpoints()

                self._save_count += 1
                logger.info(
                    "spec_diag checkpoint: saved buffer @step=%d (n_tasks=%d, versioned=%s)",
                    step,
                    ray.get(self._handle.stats.remote()).get("size", 0),
                    version_path.name,
                )
            except Exception as e:  # noqa: BLE001
                self._fail_count += 1
                logger.exception(
                    "spec_diag checkpoint: save failed: %s", e,
                )
            self._stop_event.wait(self._interval)
        logger.info(
            "spec_diag checkpoint: stopped (saves=%d, failures=%d)",
            self._save_count, self._fail_count,
        )

    def _cleanup_old_checkpoints(self) -> None:
        """Keep only the 3 most recent versioned checkpoints."""
        checkpoints = sorted(
            self._save_dir.glob("buffer_step_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # Keep buffer.json + 3 versioned
        for old in checkpoints[3:]:
            try:
                old.unlink()
            except Exception:  # noqa: BLE001
                pass


class _FeederThread(threading.Thread):
    """Daemon thread that keeps the DynamicDataset buffer above a watermark.

    Phase 0: calls ``generator.cold_start(feed_batch)`` whenever the buffer
    drops below ``low_watermark``.

    Phase 1 (when ``reward_tracker_handle`` is provided): queries the
    RewardTracker for per-tag pass rates, updates ``GeneratorMemory``,
    periodically refreshes the student profile, and calls
    ``generator.generate(memory, feed_batch)`` instead of ``cold_start``.
    Falls back to ``cold_start`` when no reward data is available yet.
    """

    def __init__(
        self,
        generator: Any,
        dataset_handle: Any,
        feed_batch: int,
        low_watermark: int,
        poll_interval_s: float,
        step_provider: Callable[[], int] | None = None,
        # Phase 1
        memory: Any = None,
        reward_tracker_handle: Any = None,
        generator_config: dict | None = None,
        student_model_name: str = "",
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
        # Phase 1
        self._memory = memory
        self._reward_tracker = reward_tracker_handle
        self._gen_config = generator_config or {}
        self._student_model_name = student_model_name
        self._last_report_step: int = 0
        self._last_generate_step: int = 0
        self._memory_update_count: int = 0
        self._profile_refresh_every: int = int(
            (self._gen_config.get("memory") or {}).get("profile_refresh_every", 4)
        )
        # Regenerate every N training steps (regardless of buffer size)
        self._regenerate_every: int = int(
            (self._gen_config.get("react") or {}).get("regenerate_every_steps", 8)
        )
        # Task logging
        self._task_log_dir: Path | None = None
        self._batch_counter: int = 0

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=timeout)

    def set_task_log_dir(self, path: Path) -> None:
        """Set the directory where generated tasks are saved as JSONL."""
        self._task_log_dir = Path(path) / "tasks"
        self._task_log_dir.mkdir(parents=True, exist_ok=True)

    def _save_tasks(
        self, tasks: list[dict], step: int, mode: str,
    ) -> None:
        if self._task_log_dir is None:
            return
        import json
        self._batch_counter += 1
        # Collect LLM prompt/response logs from the generator
        chat_logs = getattr(self._generator, "_last_chat_logs", None) or []
        record = {
            "batch": self._batch_counter,
            "step": step,
            "mode": mode,
            "n_tasks": len(tasks),
            "tasks": tasks,
            "llm_calls": chat_logs,
        }
        out = self._task_log_dir / f"batch_{self._batch_counter:05d}_step{step}.json"
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.exception("spec_diag feeder: failed to save tasks to %s", out)

    def _update_memory(self, step: int) -> bool:
        """Query RewardTracker → update memory.  Returns True if memory was updated."""
        if self._reward_tracker is None or self._memory is None:
            return False
        import ray

        try:
            self._reward_tracker.set_current_step.remote(step)
            report = ray.get(
                self._reward_tracker.get_report.remote(self._last_report_step)
            )
            if not report or not report.get("per_tag_pass_rates"):
                return False

            self._memory.update(report)
            self._last_report_step = step + 1  # exclusive: next report starts after this step
            self._memory_update_count += 1

            # Refresh student profile every K rounds
            if self._memory_update_count % self._profile_refresh_every == 0:
                logger.info(
                    "spec_diag feeder: triggering student profile refresh "
                    "(update_count=%d, every=%d)",
                    self._memory_update_count, self._profile_refresh_every,
                )
                from spec_diag.generator.student_profiler import build_student_profile
                enriched = {
                    **report,
                    "capability_trajectory": dict(self._memory.capability_trajectory),
                }
                try:
                    profile = build_student_profile(enriched, self._student_model_name)
                    if profile:
                        self._memory.student_profile = profile
                        logger.info(
                            "spec_diag feeder: refreshed student profile (%d chars)",
                            len(profile),
                        )
                    else:
                        logger.warning("spec_diag feeder: build_student_profile returned empty")
                except Exception:
                    logger.exception("spec_diag feeder: build_student_profile failed")
            return True
        except Exception:
            logger.exception("spec_diag feeder: memory update failed")
            return False

    def run(self) -> None:
        import ray

        logger.info(
            "spec_diag feeder: started (feed_batch=%d, low_watermark=%d, "
            "poll=%.1fs, phase1=%s)",
            self._feed_batch, self._low_watermark, self._poll_interval,
            self._reward_tracker is not None,
        )
        while not self._stop_event.is_set():
            try:
                stats = ray.get(self._handle.stats.remote())
                size = int(stats.get("size", 0))
                step = int(self._step_provider())

                # Trigger new task generation when:
                #   (a) buffer is below watermark, OR
                #   (b) enough training steps have passed since last generation
                #       (buffer is sampled with replacement, so it never drains
                #        on its own — we must periodically refresh the curriculum)
                steps_since_last = step - self._last_generate_step
                need_generate = (
                    size < self._low_watermark
                    or (steps_since_last >= self._regenerate_every and step > 0)
                )

                if need_generate:
                    # Phase 1: try memory-conditioned generation
                    use_memory = self._update_memory(step)

                    if use_memory:
                        tasks = self._generator.generate(
                            self._memory, self._feed_batch
                        )
                    else:
                        tasks = self._generator.cold_start(self._feed_batch)

                    if tasks:
                        ray.get(
                            self._handle.add_batch.remote(tasks, step)
                        )
                        self._last_generate_step = step
                        mode = "memory-conditioned" if use_memory else "cold_start"
                        self._save_tasks(tasks, step, mode)
                        logger.info(
                            "spec_diag feeder: +%d %s tasks @step=%d "
                            "(buffer was %d, regen_every=%d)",
                            len(tasks), mode, step, size,
                            self._regenerate_every,
                        )
                    else:
                        logger.warning(
                            "spec_diag feeder: generator returned 0 valid tasks"
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
        checkpoint_dir: str     — directory to save/load buffer checkpoints
        checkpoint_interval_s: float — seconds between checkpoint saves (default 300s)
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
        # Phase 1
        reward_tracker_handle: Any = None,
        generator_config: dict | None = None,
    ) -> None:
        self.config = config
        self.dynamic_dataset_handle = dynamic_dataset_handle
        self.generator = generator
        self.tokenizer = tokenizer
        self.processor = processor
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.reward_tracker_handle = reward_tracker_handle
        self.generator_config = generator_config or {}
        self.device_name = device_name
        self._inner = None
        self._feeder: Optional[_FeederThread] = None
        self._checkpoint: Optional[_CheckpointThread] = None
        self._checkpoint_dir: Optional[Path] = None

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

    def _warmup_buffer(self, n_tasks: int, run_dir: Path | None = None) -> int:
        """Synchronously fill the buffer with n_tasks cold-start tasks.

        If a checkpoint exists in checkpoint_dir, load it first and skip warmup
        if the loaded buffer already meets n_tasks.
        """
        import ray

        checkpoint_dir_cfg = self._sd_cfg("checkpoint_dir", None)
        if checkpoint_dir_cfg is not None and run_dir is not None:
            checkpoint_dir = Path(checkpoint_dir_cfg)
            buf_path = checkpoint_dir / "buffer.json"
            if buf_path.exists():
                logger.info(
                    "spec_diag warmup: found checkpoint at %s, loading...",
                    buf_path,
                )
                loaded = ray.get(self.dynamic_dataset_handle.load.remote(str(buf_path)))
                stats = ray.get(self.dynamic_dataset_handle.stats.remote())
                logger.info(
                    "spec_diag warmup: loaded %d tasks from checkpoint (tags=%s)",
                    loaded, stats.get("tag_counts", {}),
                )
                if loaded >= n_tasks:
                    logger.info(
                        "spec_diag warmup: checkpoint has %d tasks >= %d target, skipping warmup",
                        loaded, n_tasks,
                    )
                    return loaded

        if self.generator is None or n_tasks <= 0:
            return 0
        logger.info("spec_diag warmup: cold-starting %d tasks...", n_tasks)
        produced = 0
        consecutive_failures = 0
        max_consecutive_failures = 3  # only abort after 3 consecutive 0-result rounds
        # Do it in chunks to give the generator natural batching.
        chunk = int(self._feeder_cfg("feed_batch", 32))
        while produced < n_tasks:
            want = min(chunk, n_tasks - produced)
            tasks = self.generator.cold_start(want)
            if not tasks:
                consecutive_failures += 1
                logger.warning(
                    "spec_diag warmup: cold_start(%d) returned 0 valid "
                    "tasks (attempt %d/%d); %d/%d produced so far",
                    want, consecutive_failures, max_consecutive_failures,
                    produced, n_tasks,
                )
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(
                        "spec_diag warmup: aborting after %d consecutive "
                        "failures at %d/%d tasks",
                        max_consecutive_failures, produced, n_tasks,
                    )
                    break
                continue
            consecutive_failures = 0
            ray.get(
                self.dynamic_dataset_handle.add_batch.remote(tasks, 0)
            )
            produced += len(tasks)
            logger.info(
                "spec_diag warmup: +%d tasks (%d/%d)",
                len(tasks), produced, n_tasks,
            )
        logger.info("spec_diag warmup: buffer now has %d tasks", produced)
        return produced

    def _start_feeder(self, run_dir: Path | None = None) -> None:
        if self.generator is None:
            logger.warning(
                "spec_diag: no generator provided — feeder thread will NOT "
                "start. Buffer will drain and training will stall."
            )
            return
        if self._feeder is not None and self._feeder.is_alive():
            return

        # Phase 1: create memory if reward tracker is available
        memory = None
        if self.reward_tracker_handle is not None:
            from spec_diag.generator.memory import GeneratorMemory
            memory = GeneratorMemory()

        self._feeder = _FeederThread(
            generator=self.generator,
            dataset_handle=self.dynamic_dataset_handle,
            feed_batch=int(self._feeder_cfg("feed_batch", 32)),
            low_watermark=int(self._feeder_cfg("low_watermark", 128)),
            poll_interval_s=float(self._feeder_cfg("poll_interval_s", 10.0)),
            step_provider=self._current_step,
            memory=memory,
            reward_tracker_handle=self.reward_tracker_handle,
            generator_config=self.generator_config,
            student_model_name=str(
                getattr(self.config.actor_rollout_ref.model, "path", "")
            ),
        )
        if run_dir is not None:
            self._feeder.set_task_log_dir(run_dir)
        self._feeder.start()

    def _stop_feeder(self) -> None:
        if self._feeder is not None:
            self._feeder.stop()
            self._feeder = None

    def _start_checkpoint(self, run_dir: Path | None) -> None:
        checkpoint_dir_cfg = self._sd_cfg("checkpoint_dir", None)
        if checkpoint_dir_cfg is None:
            return
        self._checkpoint_dir = Path(checkpoint_dir_cfg)
        self._checkpoint = _CheckpointThread(
            dataset_handle=self.dynamic_dataset_handle,
            save_dir=self._checkpoint_dir,
            interval_s=float(self._sd_cfg("checkpoint_interval_s", 300.0)),
            step_provider=self._current_step,
        )
        self._checkpoint.start()
        logger.info(
            "spec_diag checkpoint: enabled (dir=%s)",
            self._checkpoint_dir,
        )

    def _stop_checkpoint(self, final_save: bool = True) -> None:
        if self._checkpoint is not None:
            if final_save:
                try:
                    # Final save before shutdown
                    import ray
                    step = self._current_step()
                    buf_path = self._checkpoint_dir / "buffer_final.json"
                    ray.get(self.dynamic_dataset_handle.save.remote(str(buf_path)))
                    logger.info(
                        "spec_diag checkpoint: final save @step=%d -> %s",
                        step, buf_path,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.exception("spec_diag checkpoint: final save failed: %s", e)
            self._checkpoint.stop()
            self._checkpoint = None

    def _current_step(self) -> int:
        if self._inner is not None:
            return int(getattr(self._inner, "global_steps", 0))
        return 0

    # ---- validation dataset ----

    def _build_val_dataset(self, poll_interval: float, max_wait: float):
        """Build validation dataset: CRUXEval if available, else dummy."""
        test_freq = getattr(self.config.trainer, "test_freq", -1)
        if test_freq > 0:
            try:
                from spec_diag.eval.cruxeval_dataset import CRUXEvalDataset
                val_ds = CRUXEvalDataset()
                logger.info(
                    "Using CRUXEval validation dataset (%d samples, "
                    "test_freq=%d)", len(val_ds), test_freq,
                )
                return val_ds
            except Exception:
                logger.warning(
                    "Failed to load CRUXEval. "
                    "Falling back to dummy val dataset.",
                    exc_info=True,
                )
        # Fallback: 1-sample dummy so verl's len(val_dataloader) >= 1 passes
        return DynamicMapDataset(
            dataset_handle=self.dynamic_dataset_handle,
            samples_per_epoch=1,
            sample_strategy=str(self._sd_cfg("sample_strategy", "mixed")),
            poll_interval_s=poll_interval,
            max_wait_s=max_wait,
        )

    # ---- build / fit ----

    def build(self, run_dir: Path | None = None):
        """Warm up buffer, start feeder, construct the RayPPOTrainer."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        warmup = int(self._feeder_cfg("warmup_tasks", 128))
        produced = self._warmup_buffer(warmup, run_dir=run_dir)
        low_wm = int(self._feeder_cfg("low_watermark", 128))
        if produced < low_wm:
            logger.warning(
                "spec_diag warmup: only produced %d tasks but low_watermark "
                "is %d — first few __getitem__ calls may block.",
                produced, low_wm,
            )

        self._start_feeder(run_dir)
        self._start_checkpoint(run_dir)

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
        # Use HumanEval as validation dataset if test_freq > 0.
        # Falls back to a dummy 1-sample dataset if evalplus not available.
        val_dataset = self._build_val_dataset(getitem_poll, getitem_wait)

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
            self._stop_checkpoint(final_save=True)
