"""spec_diag GRPO training entry point.

Hydra-managed launcher that wires verl's PPO bootstrap to spec_diag's
`DynamicGRPOTrainer`. The easiest way to think about this file:

    verl/trainer/main_ppo.py::TaskRunner.run
    + swap in DynamicDataset Ray actor + ReActGenerator
    + swap RayPPOTrainer → DynamicGRPOTrainer

We do NOT call `verl.trainer.main_ppo.run_ppo` directly (even though it
supports a `task_runner_class` hook) because that entry point invokes
`migrate_legacy_reward_impl` and other verl-internal rewrites that we want
to opt out of. Instead we reproduce the minimal bootstrap here:

    ray.init → SpecDiagTaskRunner.remote().run(cfg)

Run:

    python -m spec_diag.train \\
        actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \\
        trainer.n_gpus_per_node=2 trainer.nnodes=1 \\
        spec_diag.feeder.warmup_tasks=128

Overrides follow standard Hydra syntax — anything verl's ppo_trainer config
accepts is accepted here too.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any

import hydra
import ray
from omegaconf import OmegaConf


_CONFIG_DIR = str(Path(__file__).parent / "configs")


# --------------------------------------------------------------------- logging


def _configure_logging(run_dir: Path | None) -> None:
    """Wire up root logging so every `logging.getLogger(...)` call inside
    spec_diag (trainer, feeder, reward manager, …) actually produces output.

    Two handlers:
      - stderr (captured by the orchestrator's `tee train.log`)
      - file handler at `$run_dir/train_python.log` if the orchestrator
        provided a run dir via `SPEC_DIAG_RUN_DIR`.

    We deliberately do NOT touch verl's loggers — they configure themselves
    and we don't want to double-print. We set propagate=False on ours and
    leave the verl logger tree alone.
    """
    fmt = "%(asctime)s %(levelname).1s %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    # Clear pre-existing handlers (Hydra installs its own) so we start clean.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            run_dir / "train_python.log", mode="a", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Keep the spec_diag subtree at INFO, noisy libs at WARNING.
    logging.getLogger("spec_diag").setLevel(logging.INFO)
    for noisy in ("urllib3", "httpx", "httpcore", "openai", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _dump_resolved_config(config, run_dir: Path | None) -> None:
    if run_dir is None:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "config_resolved.yaml"
    try:
        text = OmegaConf.to_yaml(config, resolve=True)
    except Exception as e:  # noqa: BLE001
        text = f"# OmegaConf.to_yaml(resolve=True) failed: {e}\n"
    out.write_text(text, encoding="utf-8")


def _load_generator_config() -> dict[str, Any]:
    """Read `configs/generator.yaml`. Kept separate from the hydra config
    tree so Student training and Generator inference can evolve independently.
    """
    import yaml

    cfg_path = Path(__file__).parent / "configs" / "generator.yaml"
    with cfg_path.open("r") as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------- runner


def _build_spec_diag_task_runner_cls():
    """Construct SpecDiagTaskRunner lazily so `import spec_diag.train` does
    not require verl at module import time (tests, tooling, etc.).
    """
    from verl.trainer.main_ppo import TaskRunner
    from verl.utils.config import validate_config
    from verl.trainer.ppo.utils import need_critic, need_reference_policy

    class SpecDiagTaskRunner(TaskRunner):
        """verl TaskRunner that builds a DynamicGRPOTrainer instead of
        RayPPOTrainer. Reuses all of the parent's worker / resource-pool
        helpers; only `run()` is overridden.
        """

        def run(self, config):
            from verl.utils import hf_processor, hf_tokenizer
            from verl.utils.fs import copy_to_local

            from spec_diag.dataset.dynamic_dataset import DynamicDataset
            from spec_diag.generator.react_generator import ReActGenerator
            from spec_diag.trainer.dynamic_grpo_trainer import DynamicGRPOTrainer

            # We are in a fresh Ray actor process — the driver's logging
            # config didn't propagate. Re-run it so trainer / feeder logs
            # flow into $SPEC_DIAG_RUN_DIR/train_python.log.
            run_dir_env = os.environ.get("SPEC_DIAG_RUN_DIR")
            actor_run_dir = Path(run_dir_env) if run_dir_env else None
            _configure_logging(actor_run_dir)

            actor_log = logging.getLogger("spec_diag.train.runner")
            actor_log.info(
                "SpecDiagTaskRunner starting host=%s pid=%d run_dir=%s",
                socket.gethostname(), os.getpid(), actor_run_dir,
            )
            OmegaConf.resolve(config)

            # ---- ensure custom reward function is wired ----
            # verl's agent_loop / RewardLoopWorker calls load_reward_manager()
            # which reads config.reward.custom_reward_function.  CLI overrides
            # and pkg:// paths can silently fail in the Ray worker env, so we
            # force-set them here to be safe.
            from omegaconf import open_dict
            with open_dict(config):
                if not config.reward.get("custom_reward_function"):
                    config.reward.custom_reward_function = {}
                config.reward.custom_reward_function.path = (
                    "pkg://spec_diag.rewards.spec_diag_score"
                )
                config.reward.custom_reward_function.name = "compute_score"
                # Also set legacy top-level key in case installed verl reads it
                if hasattr(config, "custom_reward_function"):
                    config.custom_reward_function.path = (
                        "pkg://spec_diag.rewards.spec_diag_score"
                    )
                    config.custom_reward_function.name = "compute_score"
            actor_log.info(
                "custom_reward_function forced: path=%s name=%s",
                config.reward.custom_reward_function.path,
                config.reward.custom_reward_function.name,
            )

            # ---- worker / resource-pool bootstrap (verl parent helpers) ----
            actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)

            # NOTE: different verl versions expose different helper methods on
            # TaskRunner. Keep this bootstrap backward-compatible so we can run
            # against both bundled and env-installed verl builds.
            add_rm_pool = getattr(self, "add_reward_model_resource_pool", None)
            if callable(add_rm_pool):
                add_rm_pool(config)
            else:
                actor_log.info(
                    "TaskRunner has no add_reward_model_resource_pool(); skipping."
                )

            add_teacher_pool = getattr(self, "add_teacher_model_resource_pool", None)
            if callable(add_teacher_pool):
                add_teacher_pool(config)
            else:
                actor_log.info(
                    "TaskRunner has no add_teacher_model_resource_pool(); skipping."
                )

            self.add_ref_policy_worker(config, actor_rollout_cls)

            validate_config(
                config=config,
                use_reference_policy=need_reference_policy(config),
                use_critic=need_critic(config),
            )

            # ---- tokenizer / processor ----
            local_path = copy_to_local(
                config.actor_rollout_ref.model.path,
                use_shm=config.actor_rollout_ref.model.get("use_shm", False),
            )
            trust_remote_code = config.data.get("trust_remote_code", False)
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            processor = hf_processor(
                local_path, trust_remote_code=trust_remote_code, use_fast=True
            )

            resource_pool_manager = self.init_resource_pool_mgr(config)

            # ---- spec_diag: DynamicDataset actor + ReActGenerator ----
            gen_cfg = _load_generator_config()
            model_cfg = gen_cfg.get("model", {}) or {}
            # Allow env override, fall back to vllm --served-model-name alias.
            generator_model = (
                os.environ.get("SPEC_DIAG_MODEL")
                or model_cfg.get("served_name")
                or "generator"
            )
            generator = ReActGenerator(
                model_name=generator_model,
                config={
                    "temperature": float(model_cfg.get("temperature", 0.7)),
                    "max_tokens": int(model_cfg.get("max_tokens", 4096)),
                    "validity_timeout": 5,
                    "validity_workers": 2,
                    # Resolve seed_data_path relative to project root
                    # (not cwd, which may differ in Ray actors)
                    "seed_data_path": str(
                        Path(__file__).resolve().parent.parent
                        / model_cfg["seed_data_path"]
                    ) if model_cfg.get("seed_data_path") else None,
                    "n_references": int(model_cfg.get("n_references", 6)),
                },
            )

            dataset_actor = DynamicDataset.remote(
                max_size=int(
                    OmegaConf.select(config, "spec_diag.dataset_max_size")
                    or 10_000
                ),
            )
            actor_log.info(
                "DynamicDataset actor spawned; stats=%s",
                ray.get(dataset_actor.stats.remote()),
            )

            # ---- Phase 1: RewardTracker named actor ----
            from spec_diag.rewards.reward_tracker import (
                RewardTracker, REWARD_TRACKER_NAME,
            )
            react_cfg = gen_cfg.get("react", {}) or {}
            reward_tracker_handle = RewardTracker.options(
                name=REWARD_TRACKER_NAME,
            ).remote(
                max_failures_per_tag=int(
                    react_cfg.get("failure_samples_per_tag", 3) * 4
                ),
            )
            actor_log.info(
                "RewardTracker named actor created: %s", REWARD_TRACKER_NAME,
            )

            # ---- construct DynamicGRPOTrainer ----
            trainer = DynamicGRPOTrainer(
                config=config,
                dynamic_dataset_handle=dataset_actor,
                generator=generator,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_tracker_handle=reward_tracker_handle,
                generator_config=gen_cfg,
            )

            # build() runs the synchronous warmup + starts the feeder +
            # constructs the inner RayPPOTrainer. Then init_workers spins up
            # the actor / ref / rollout worker groups, and fit() starts the
            # PPO loop.
            actor_log.info("trainer.build()")
            trainer.build(run_dir=actor_run_dir)
            actor_log.info("trainer.init_workers()")
            trainer.init_workers()
            actor_log.info("trainer.fit() — entering verl loop")
            try:
                trainer.fit()
            except Exception:
                actor_log.exception("trainer.fit() raised")
                raise
            finally:
                actor_log.info("trainer.fit() exited; cleaning up")
                try:
                    tracker_stats = ray.get(reward_tracker_handle.stats.remote())
                    actor_log.info("RewardTracker final stats: %s", tracker_stats)
                    ray.kill(reward_tracker_handle)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    generator.close()
                except Exception:  # noqa: BLE001
                    actor_log.exception("generator.close() raised")

    return SpecDiagTaskRunner


# --------------------------------------------------------------------- main


# Driver-side env vars that must propagate into Ray actors so the
# SpecDiagTaskRunner can construct ReActGenerator pointed at vLLM.
# Ray workers do NOT inherit the launching process's env by default —
# anything the generator needs has to go through runtime_env.env_vars.
_SPEC_DIAG_ENV_PASSTHROUGH = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "SPEC_DIAG_MODEL",
    "SPEC_DIAG_N",
    "SPEC_DIAG_RUN_DIR",
    "HF_HOME",
)


@hydra.main(config_path="configs", config_name="spec_diag_grpo", version_base=None)
def main(config) -> None:
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.utils.device import auto_set_device

    run_dir_env = os.environ.get("SPEC_DIAG_RUN_DIR")
    run_dir = Path(run_dir_env) if run_dir_env else None
    _configure_logging(run_dir)
    _dump_resolved_config(config, run_dir)

    driver_log = logging.getLogger("spec_diag.train")
    driver_log.info(
        "main() host=%s pid=%d run_dir=%s",
        socket.gethostname(), os.getpid(), run_dir,
    )

    auto_set_device(config)

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        # Pass spec_diag-specific env vars through to every Ray worker.
        passthrough_env = {
            k: os.environ[k]
            for k in _SPEC_DIAG_ENV_PASSTHROUGH
            if os.environ.get(k) is not None
        }
        # Ensure the project root is on PYTHONPATH so that
        # `import spec_diag.*` works in ALL Ray workers (including
        # RewardLoopWorker spawned by verl internals, not just our
        # SpecDiagTaskRunner).
        project_root = str(Path(__file__).resolve().parent.parent)
        existing_pp = os.environ.get("PYTHONPATH", "")
        if project_root not in existing_pp.split(os.pathsep):
            passthrough_env["PYTHONPATH"] = (
                f"{project_root}{os.pathsep}{existing_pp}" if existing_pp
                else project_root
            )
        if passthrough_env:
            default_runtime_env.setdefault("env_vars", {}).update(passthrough_env)
            print(
                "[spec_diag] passing through to Ray actors: "
                f"{sorted(passthrough_env.keys())}"
            )

        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create(
            {**ray_init_kwargs, "runtime_env": runtime_env}
        )
        print(f"[spec_diag] ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner_cls = ray.remote(num_cpus=1)(_build_spec_diag_task_runner_cls())
    runner = runner_cls.remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    main()
