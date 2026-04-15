"""spec_diag Phase 0 smoke test entry point.

This is NOT a real training run. It verifies the pipeline wiring end-to-end
without verl / GPUs:

  1. Ray init (local).
  2. Spawn a `DynamicDataset` actor.
  3. Load `spec_diag/configs/generator.yaml` → build a `ReActGenerator`.
  4. `generator.cold_start(N)` — hits the vLLM OpenAI endpoint, gets N
     validated code tasks back with filled-in `gold_output`.
  5. Push them into the actor via `add_batch`.
  6. Sample a few out, feed each to `CodeExecutor.eval_student` with both
     the correct gold answer and a wrong answer, check rewards.
  7. Print stats and exit cleanly.

Env:
  OPENAI_BASE_URL   default http://localhost:8000/v1
  OPENAI_API_KEY    default "dummy"
  SPEC_DIAG_N       how many cold-start tasks to request (default 4)

Run:
  python -m spec_diag.main
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

from spec_diag.executors.code_executor import CodeExecutor
from spec_diag.generator.react_generator import ReActGenerator


def _load_generator_config() -> dict[str, Any]:
    cfg_path = Path(__file__).parent / "configs" / "generator.yaml"
    with cfg_path.open("r") as f:
        return yaml.safe_load(f) or {}


def _print_stats(dataset_handle) -> None:
    import ray

    stats = ray.get(dataset_handle.stats.remote())
    print(f"[dataset] stats: {stats}")


def main() -> int:
    import ray

    from spec_diag.dataset.dynamic_dataset import DynamicDataset

    cfg = _load_generator_config()
    model_cfg = cfg.get("model", {}) or {}
    react_cfg = cfg.get("react", {}) or {}

    # Allow OPENAI_BASE_URL-style override; fall back to the served name so
    # vLLM's --served-model-name=generator alias works out of the box.
    model_name = os.environ.get("SPEC_DIAG_MODEL") or model_cfg.get(
        "name", "generator"
    )
    n_cold = int(os.environ.get("SPEC_DIAG_N", "4"))

    print("=" * 60)
    print(f"[spec_diag] Phase 0 smoke test")
    print(f"[spec_diag] OPENAI_BASE_URL={os.environ.get('OPENAI_BASE_URL', 'http://localhost:8000/v1')}")
    print(f"[spec_diag] model={model_name}")
    print(f"[spec_diag] n_cold_start={n_cold}")
    print("=" * 60)

    # --- 1. Ray ---
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=True)

    # --- 2. DynamicDataset actor ---
    dataset = DynamicDataset.remote(max_size=1024)
    print("[ray] DynamicDataset actor spawned")

    # --- 3. Generator ---
    generator = ReActGenerator(
        model_name=model_name,
        config={
            "temperature": float(model_cfg.get("temperature", 0.7)),
            "max_tokens": int(model_cfg.get("max_tokens", 4096)),
            "validity_timeout": 5,
            "validity_workers": 2,
        },
    )

    # --- 4. Cold start ---
    try:
        tasks = generator.cold_start(n_cold)
    except Exception as e:
        print(f"[FATAL] generator.cold_start failed: {e}", file=sys.stderr)
        print(
            "  Is vLLM running? Check `curl $OPENAI_BASE_URL/models`.",
            file=sys.stderr,
        )
        generator.close()
        return 2

    print(f"[generator] got {len(tasks)} valid tasks from cold_start")
    for i, t in enumerate(tasks):
        print(f"  [{i}] tags={t.get('capability_tags')} inputs={t.get('inputs')!r} "
              f"gold={t.get('gold_output')!r}")

    if not tasks:
        print("[FATAL] zero valid tasks survived validity check.", file=sys.stderr)
        generator.close()
        return 3

    # --- 5. Push into DynamicDataset ---
    size = ray.get(dataset.add_batch.remote(tasks, 0))
    print(f"[dataset] add_batch → buffer size = {size}")
    _print_stats(dataset)

    # --- 6. Sample + grade ---
    sampled = ray.get(dataset.sample_batch.remote(min(len(tasks), 3), "uniform"))
    print(f"[dataset] sampled {len(sampled)} tasks for grading")

    grader = CodeExecutor(timeout_length=5, max_workers=2, ast_check=True)
    try:
        correct_rewards = []
        wrong_rewards = []
        for t in sampled:
            gold = t["gold_output"]
            r_correct = grader.eval_student(t, gold)
            r_wrong = grader.eval_student(t, "__definitely_wrong__")
            correct_rewards.append(r_correct)
            wrong_rewards.append(r_wrong)
            print(
                f"  task tags={t.get('capability_tags')} "
                f"r(correct)={r_correct:.2f} r(wrong)={r_wrong:.2f}"
            )

        if not correct_rewards or any(r < 1.0 for r in correct_rewards):
            print(
                "[WARN] some 'correct' answers did not score 1.0 — check "
                "gold_output formatting / determinism.",
                file=sys.stderr,
            )
        if any(r > 0.0 for r in wrong_rewards):
            print(
                "[WARN] some 'wrong' answers scored > 0 — grader may be too loose.",
                file=sys.stderr,
            )
    finally:
        grader.close()
        generator.close()

    print("=" * 60)
    print("[spec_diag] smoke test done.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
