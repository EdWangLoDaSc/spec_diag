"""Stub tests for DynamicDataset. All xfail until Phase 0 implementation lands."""

import pytest


def test_import():
    from spec_diag.dataset.dynamic_dataset import DynamicDataset  # noqa: F401


@pytest.mark.xfail(reason="Phase 0 — not implemented yet")
def test_add_batch_and_sample():
    import ray

    from spec_diag.dataset.dynamic_dataset import DynamicDataset

    if not ray.is_initialized():
        ray.init(local_mode=True)

    ds = DynamicDataset.remote(max_size=100)
    ray.get(ds.add_batch.remote([{"id": 1}, {"id": 2}], step=0))
    sampled = ray.get(ds.sample_batch.remote(n=2))
    assert len(sampled) == 2


@pytest.mark.xfail(reason="Phase 0 — not implemented yet")
def test_stats():
    import ray

    from spec_diag.dataset.dynamic_dataset import DynamicDataset

    if not ray.is_initialized():
        ray.init(local_mode=True)

    ds = DynamicDataset.remote()
    stats = ray.get(ds.stats.remote())
    assert "size" in stats
