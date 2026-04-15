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

import os
import socket
from pathlib import Path
from typing import Any

import hydra
import ray
from omegaconf import OmegaConf


_CONFIG_DIR = str(Path(__file__).parent / "configs")


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
            from pprint import pprint

            from verl.utils import hf_processor, hf_tokenizer
            from verl.utils.fs import copy_to_local

            from spec_diag.dataset.dynamic_dataset import DynamicDataset
            from spec_diag.generator.react_generator import ReActGenerator
            from spec_diag.trainer.dynamic_grpo_trainer import DynamicGRPOTrainer

            print(
                f"[spec_diag] SpecDiagTaskRunner host={socket.gethostname()} "
                f"pid={os.getpid()}"
            )
            pprint(OmegaConf.to_container(config, resolve=True))
            OmegaConf.resolve(config)

            # ---- worker / resource-pool bootstrap (verl parent helpers) ----
            actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.add_reward_model_resource_pool(config)
            self.add_teacher_model_resource_pool(config)
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
                },
            )

            dataset_actor = DynamicDataset.remote(
                max_size=int(
                    OmegaConf.select(config, "spec_diag.dataset_max_size")
                    or 10_000
                ),
            )
            print(
                f"[spec_diag] DynamicDataset actor spawned "
                f"(max_size={ray.get(dataset_actor.stats.remote()).get('size', 0)} "
                f"tasks initially)"
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
            )

            # build() runs the synchronous warmup + starts the feeder +
            # constructs the inner RayPPOTrainer. Then init_workers spins up
            # the actor / ref / rollout worker groups, and fit() starts the
            # PPO loop.
            trainer.build()
            trainer.init_workers()
            try:
                trainer.fit()
            finally:
                try:
                    generator.close()
                except Exception:  # noqa: BLE001
                    pass

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
    "HF_HOME",
)


@hydra.main(config_path="configs", config_name="spec_diag_grpo", version_base=None)
def main(config) -> None:
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.utils.device import auto_set_device

    auto_set_device(config)

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        # Pass spec_diag-specific env vars through to every Ray worker.
        passthrough_env = {
            k: os.environ[k]
            for k in _SPEC_DIAG_ENV_PASSTHROUGH
            if os.environ.get(k) is not None
        }
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
