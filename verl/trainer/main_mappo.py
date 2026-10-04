# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.
"""

import json
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.mappo_trainer import RayMAPPOTrainer, RayRiskAverseTrainer, create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type


@hydra.main(config_path="config", config_name="mappo_trainer", version_base=None)
def main(config):
    """Main entry point for MAPPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_mappo(config)


# Define a function to run the PPO-like training process
def run_mappo(config) -> None:
    """Initialize Ray cluster and run distributed MAPPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed MAPPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available
        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        from verl.trainer.ppo.ray_trainer import Role
        from verl.trainer.ppo.utils import need_critic, need_reference_policy

        # read multi-agent config
        ma = OmegaConf.select(config, "multi_agent") or {}
        num_agents = int(ma.get("num_agents", 1))
        assert num_agents >= 2, "MAPPO expected multi_agent.num_agents >= 2"
        agents_cfg = ma.get("agents", [])
        if not agents_cfg or len(agents_cfg) != num_agents:
            raise ValueError(
                "Please provide multi_agent.agents list (length == num_agents) with per-agent resource/model entries."
            )

        resource_pool_spec = {}
        colocate_count_dict = {}

        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")
            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
            colocate_count_dict["reward_pool"] = 1  # only RewardModelWorker

        # Divide the cluster's total nodes equally among agents.
        # nnodes is always global (config.trainer.nnodes); per-agent n_gpus_per_node may differ.
        nodes_per_agent = config.trainer.nnodes // num_agents
        if nodes_per_agent <= 0:
            raise ValueError(
                f"nodes_per_agent must be >=1; got trainer.nnodes={config.trainer.nnodes} "
                f"with num_agents={num_agents}. Increase trainer.nnodes or reduce num_agents."
            )

        # max_colocate_count drives num_gpus = 1/N per Ray actor (see
        # verl/single_controller/ray/base.py:_create_worker). For MAPPO,
        # create_colocated_worker_cls bundles actor+critic(+ref) into ONE WorkerDict
        # Ray actor per bundle, so N=1 (full integer GPU). Setting it to the count
        # of internal logical workers (e.g. 2 for actor+critic) re-introduces the
        # which crashes the worker-node actor with "no CUDA accelerator available".
        for i in range(num_agents):
            a = agents_cfg[i]
            n_gpus = int(a.get("n_gpus_per_node", config.trainer.n_gpus_per_node))
            resource_pool_spec[f"agent_pool_{i}"] = [n_gpus] * nodes_per_agent
            colocate_count_dict[f"agent_pool_{i}"] = 1
            self.mapping[f"agent_pool_{i}"] = f"agent_pool_{i}"
            self.mapping[f"critic_pool_{i}"] = f"agent_pool_{i}"

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
            colocate_count_dict=colocate_count_dict,
        )
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker
                else:
                    raise NotImplementedError
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import RewardModelWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        from verl.trainer.ppo.ray_trainer import Role
        # read multi-agent config
        ma = OmegaConf.select(config, "multi_agent") or {}
        num_agents = int(ma.get("num_agents", 1))
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            for i in range(num_agents):
                self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
                self.mapping[f"ref_pool_{i}"] = f"agent_pool_{i}"
            # self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            # self.mapping[Role.RefPolicy] = f"agent_pool_{agent_idx}"

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)
        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizers={}
        processors={}
        reward_fns={}
        val_reward_fns={}
        train_datasets={}
        val_datasets={}
        train_samplers={}

        ma = OmegaConf.select(config, "multi_agent", default={}) or {}
        agents_cfg = ma.get("agents", [])
        num_agents = int(ma.get("num_agents", 1))

        for i in range(num_agents):
            local_path=copy_to_local(
            agents_cfg[i].actor.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
            tokenizers[f"model_{i}"]= hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            processors[f"model_{i}"]= hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
            reward_fns[f"model_{i}"] = load_reward_manager(
                config, tokenizers[f"model_{i}"], **config.reward_model.get("reward_kwargs", {})
            )
            val_reward_fns[f"model_{i}"] = load_reward_manager(
                config, tokenizers[f"model_{i}"], **config.reward_model.get("reward_kwargs", {})
            )
            train_datasets[f"model_{i}"] = create_rl_dataset(config.data.train_files, config.data, tokenizers[f"model_{i}"], processors[f"model_{i}"], is_train=True)
            val_datasets[f"model_{i}"] = create_rl_dataset(config.data.val_files, config.data, tokenizers[f"model_{i}"], processors[f"model_{i}"], is_train=False)
            train_samplers[f"model_{i}"] = create_rl_sampler(config.data, train_datasets[f"model_{i}"])

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn


        # Initialize the PPO trainer.
        ma = OmegaConf.select(config, "multi_agent", default={}) or {}
        trainer_type = ma.get("trainer_type", "mappo")
        TrainerCls = RayRiskAverseTrainer if trainer_type == "risk_averse" else RayMAPPOTrainer
        trainer = TrainerCls(
            config=config,
            tokenizers=tokenizers,
            processors=processors,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fns=reward_fns,
            val_reward_fns=val_reward_fns,
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            collate_fn=collate_fn,
            train_samplers=train_samplers,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Probe-only mode: skip training; trainer.run_offline_probe handles the
        # full sequence (load ckpt, sync vLLM, run probe, dump).
        probe_cfg = OmegaConf.select(config, "multi_agent.cross_pair_probe", default={}) or {}
        if bool(probe_cfg.get("run_only", False)):
            out_path = probe_cfg.get("out_path", None) or None
            payload = trainer.run_offline_probe(out_path=out_path)
            if out_path:
                print(f"[cross_pair_probe] wrote {out_path}")
            print(f"[cross_pair_probe] {json.dumps(payload)}")
            return

        # Start the training process.
        trainer.mappo_fit()


if __name__ == "__main__":
    main()
