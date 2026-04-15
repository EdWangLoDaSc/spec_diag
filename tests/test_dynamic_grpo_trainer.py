"""Tests for DynamicMapDataset + _FeederThread + sample schema.

These exercise the trainer scaffolding **without** verl / GPUs. They require
torch + ray (local_mode is fine). If either is unavailable, the whole module
is skipped.
"""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def ray_local():
    ray = pytest.importorskip("ray")

    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True, log_to_driver=False)
    yield ray
    # Leave ray running — other test modules in the same session may need it.


def _fake_task(i: int) -> dict:
    return {
        "domain": "code",
        "code": f"def f(x):\n    return x + {i}",
        "inputs": "1",
        "imports": [],
        "capability_tags": ["arithmetic"],
        "gold_output": repr(1 + i),
    }


# ---------------------------------------------------------------- _task_to_sample


def test_task_to_sample_schema():
    """Sample must match verl RLHFDataset.__getitem__ shape."""
    import torch

    from spec_diag.trainer.dynamic_grpo_trainer import _task_to_sample

    sample = _task_to_sample(_fake_task(3), index=7)

    # prompt must be a list of chat messages, not a string.
    assert isinstance(sample["prompt"], list)
    assert sample["prompt"] == sample["raw_prompt"]
    assert all(isinstance(m, dict) and "role" in m and "content" in m
               for m in sample["prompt"])
    # spec_diag_task carries the original dict for the reward manager.
    assert sample["spec_diag_task"]["gold_output"] == repr(4)
    # reward_model.ground_truth = task for verl-native fallback.
    assert sample["reward_model"]["ground_truth"]["code"].startswith("def f")
    # dummy_tensor is present + a torch Tensor.
    assert isinstance(sample["dummy_tensor"], torch.Tensor)
    # verl-required bookkeeping fields.
    assert sample["index"] == 7
    assert sample["extra_info"] == {"index": 7}
    assert sample["tools_kwargs"] == {}
    assert sample["interaction_kwargs"] == {}
    assert sample["data_source"] == "spec_diag_code"


# ---------------------------------------------------------------- DynamicMapDataset


def test_dynamic_map_dataset_len_and_getitem(ray_local):
    from spec_diag.dataset.dynamic_dataset import DynamicDataset
    from spec_diag.trainer.dynamic_grpo_trainer import DynamicMapDataset

    handle = DynamicDataset.remote(max_size=100)
    ray_local.get(handle.add_batch.remote([_fake_task(i) for i in range(5)], 0))

    ds = DynamicMapDataset(
        dataset_handle=handle,
        samples_per_epoch=64,
        sample_strategy="uniform",
        poll_interval_s=0.01,
        max_wait_s=1.0,
    )
    assert len(ds) == 64

    # __getitem__ pulls a real sample; index is only for bookkeeping.
    sample = ds[0]
    assert "prompt" in sample
    assert "spec_diag_task" in sample


def test_dynamic_map_dataset_blocks_then_unblocks(ray_local):
    """Empty buffer → __getitem__ blocks; after add_batch it returns."""
    from spec_diag.dataset.dynamic_dataset import DynamicDataset
    from spec_diag.trainer.dynamic_grpo_trainer import DynamicMapDataset

    handle = DynamicDataset.remote(max_size=100)
    ds = DynamicMapDataset(
        dataset_handle=handle,
        samples_per_epoch=8,
        sample_strategy="uniform",
        poll_interval_s=0.02,
        max_wait_s=5.0,
    )

    result_box: dict = {}

    def _worker():
        result_box["sample"] = ds[0]

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    # Give it a moment to block on the empty buffer.
    time.sleep(0.1)
    assert t.is_alive(), "should still be blocked on empty buffer"

    ray_local.get(handle.add_batch.remote([_fake_task(0)], 0))
    t.join(timeout=5.0)
    assert not t.is_alive(), "should have unblocked after add_batch"
    assert "sample" in result_box


def test_dynamic_map_dataset_times_out(ray_local):
    """Buffer stays empty past max_wait_s → RuntimeError."""
    from spec_diag.dataset.dynamic_dataset import DynamicDataset
    from spec_diag.trainer.dynamic_grpo_trainer import DynamicMapDataset

    handle = DynamicDataset.remote(max_size=100)
    ds = DynamicMapDataset(
        dataset_handle=handle,
        samples_per_epoch=1,
        sample_strategy="uniform",
        poll_interval_s=0.05,
        max_wait_s=0.2,
    )
    with pytest.raises(RuntimeError, match="stayed empty"):
        _ = ds[0]


# ---------------------------------------------------------------- _FeederThread


class _MockGenerator:
    """Drop-in replacement for ReActGenerator that skips the vLLM call."""

    def __init__(self, fail_first: bool = False) -> None:
        self.call_count = 0
        self.fail_first = fail_first

    def cold_start(self, n: int) -> list[dict]:
        self.call_count += 1
        if self.fail_first and self.call_count == 1:
            raise RuntimeError("simulated network error")
        return [_fake_task(self.call_count * 100 + i) for i in range(n)]


def test_feeder_refills_below_watermark(ray_local):
    from spec_diag.dataset.dynamic_dataset import DynamicDataset
    from spec_diag.trainer.dynamic_grpo_trainer import _FeederThread

    handle = DynamicDataset.remote(max_size=1000)
    gen = _MockGenerator()
    feeder = _FeederThread(
        generator=gen,
        dataset_handle=handle,
        feed_batch=8,
        low_watermark=16,
        poll_interval_s=0.05,
    )
    feeder.start()
    try:
        # Let it run a few iterations.
        deadline = time.time() + 3.0
        while time.time() < deadline:
            size = ray_local.get(handle.stats.remote())["size"]
            if size >= 16:
                break
            time.sleep(0.05)
        size = ray_local.get(handle.stats.remote())["size"]
        assert size >= 16, f"feeder failed to cross watermark, size={size}"
        assert gen.call_count >= 2
    finally:
        feeder.stop()
    assert not feeder.is_alive()


def test_feeder_survives_generator_exception(ray_local):
    from spec_diag.dataset.dynamic_dataset import DynamicDataset
    from spec_diag.trainer.dynamic_grpo_trainer import _FeederThread

    handle = DynamicDataset.remote(max_size=1000)
    gen = _MockGenerator(fail_first=True)
    feeder = _FeederThread(
        generator=gen,
        dataset_handle=handle,
        feed_batch=4,
        low_watermark=4,
        poll_interval_s=0.05,
    )
    feeder.start()
    try:
        deadline = time.time() + 3.0
        while time.time() < deadline:
            size = ray_local.get(handle.stats.remote())["size"]
            if size >= 4:
                break
            time.sleep(0.05)
        # First call raised, later calls succeeded.
        assert gen.call_count >= 2
        assert feeder._fail_count >= 1
        assert feeder.is_alive(), "feeder should survive exceptions"
    finally:
        feeder.stop()
