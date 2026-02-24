# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed.checkpoint as dcp
from monarch.actor import Actor, endpoint
from torch.distributed._tensor import DTensor

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.rl.unified.actors.grader import Episodes
from torchtitan.experiments.rl.unified.actors.utils import (
    compute_policy_gradient_loss,
    compute_token_log_probs,
    verify_logprob_identity,
)
from torchtitan.experiments.rl.unified.configs import PolicyOptimizationConfig
from torchtitan.experiments.rl.unified.models.utils import (
    replace_with_vllm_compatible_flash_attention,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils

logger = logging.getLogger(__name__)



def _set_nccl_determinism_envs() -> None:
    """Set environment variables to force deterministic NCCL collective operations.

    This configures NCCL to use a single-channel tree all-reduce with the Simple
    protocol, ensuring a fixed reduction order and bitwise-reproducible results
    at the cost of reduced throughput.
    """
    # Disable symmetric memory all-reduce (non-deterministic)
    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    # Deterministic cuBLAS
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # NCCL determinism: fixed tree algorithm, simple protocol,
    # single channel/thread to ensure identical reduction order
    os.environ["NCCL_LAUNCH_MODE"] = "GROUP"
    os.environ["NCCL_COLLNET_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["NCCL_P2P_NET_DISABLE"] = "1"
    os.environ["NCCL_MIN_NCHANNELS"] = "1"
    os.environ["NCCL_MAX_NCHANNELS"] = "1"
    os.environ["NCCL_PROTO"] = "Simple"
    os.environ["NCCL_ALGO"] = "allreduce:tree"
    os.environ["NCCL_NTHREADS"] = "1"
    os.environ["NCCL_SOCKET_NTHREADS"] = "1"


class PolicyTrainer(Actor, Configurable):
    """
    Updates policy based on collected trajectories using TorchTitan components.

    Uses ModelSpec for model construction, parallelization, and weight loading.

    Args:
        config: PolicyTrainer.Config for model/optimizer/parallelism settings.
        policy_optimization: GRPO hyperparameters.
        model_spec: Model specification (model config, parallelize_fn, state_dict_adapter).
        hf_assets_path: Path to HF assets folder for checkpoint loading.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """PolicyTrainer configuration for optimizer, training, and parallelism."""

        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        activation_checkpoint: ActivationCheckpointConfig = field(
            default_factory=ActivationCheckpointConfig
        )
        compile: CompileConfig = field(default_factory=CompileConfig)
        model_converters: ModelConvertersContainer.Config = field(
            default_factory=ModelConvertersContainer.Config
        )
        comm: CommConfig = field(default_factory=CommConfig)

    def __init__(
        self,
        config: Config,
        policy_optimization: PolicyOptimizationConfig,
        model_spec: ModelSpec,
        hf_assets_path: str = "./tests/assets/tokenizer",
        batch_invariant_mode: bool = True,
    ):
        self.config = config
        self.model_spec = model_spec

        # GRPO settings
        self.group_size = policy_optimization.group_size
        self.grpo_beta = policy_optimization.beta
        self.use_stable_grpo = policy_optimization.use_stable_grpo

        # Batch invariant mode: set NCCL determinism env vars
        if batch_invariant_mode:
            _set_nccl_determinism_envs()

        # Device setup
        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        # Initialize distributed
        # When running under Monarch, setup_env_for_distributed already
        # initializes the process group, so skip re-initialization.
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = dist_utils.init_distributed(config.comm)

        # Build parallel dims
        parallelism_config = config.parallelism
        self.parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
        )

        # Build model config from model_spec
        model_config = model_spec.model

        # Initialize state dict adapter for HF checkpoint loading
        if model_spec.state_dict_adapter is not None:
            self.sd_adapter = model_spec.state_dict_adapter(
                model_config, hf_assets_path
            )
        else:
            self.sd_adapter = None

        # Build model with meta init
        with torch.device("meta"):
            with utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
                model = model_config.build()

        # Replace attention with vLLM compatible attention for RL training
        # NOTE: We do this now for attention backward compatibility.
        # Long-term this will be replaced by pytorch attention supporting paged attention / kv cache
        replace_with_vllm_compatible_flash_attention(model)

        # Apply parallelization using model_spec.parallelize_fn.
        # The RL entry point (simple_grpo.py) patches this to the RL-specific
        # parallelize_qwen3 which includes inner_attention hooks for DTensor→local
        # conversion needed by vLLM's flash attention kernels.
        model = model_spec.parallelize_fn(
            model,
            parallel_dims=self.parallel_dims,
            training=config.training,
            model_converters=config.model_converters,
            parallelism=config.parallelism,
            compile_config=config.compile,
            ac_config=config.activation_checkpoint,
            dump_folder=".",
        )

        # Initialize model weights on device
        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        # Load initial weights from checkpoint if specified
        if config.checkpoint.initial_load_path:
            self._load_initial_weights(model, config.checkpoint.initial_load_path)

        model.train()
        self.model = model
        self.model_parts = [model]

        # Create reference model for KL divergence (frozen copy of initial policy)
        with torch.device("meta"):
            with utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
                ref_model = model_config.build()

        replace_with_vllm_compatible_flash_attention(ref_model)

        ref_model = model_spec.parallelize_fn(
            ref_model,
            parallel_dims=self.parallel_dims,
            training=config.training,
            model_converters=config.model_converters,
            parallelism=config.parallelism,
            compile_config=config.compile,
            ac_config=config.activation_checkpoint,
            dump_folder=".",
        )

        ref_model.to_empty(device=device_type)
        with torch.no_grad():
            ref_model.init_weights(buffer_device=None)

        if config.checkpoint.initial_load_path:
            self._load_initial_weights(
                ref_model, config.checkpoint.initial_load_path
            )
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()
        self.ref_model = ref_model

        # Build optimizer and LR scheduler
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )

        self.policy_version = 0
        self.generator: Optional[Any] = None

        logger.info(
            f"PolicyTrainer initialized: "
            f"group_size={self.group_size}, grpo_beta={self.grpo_beta}, "
            f"use_stable_grpo={self.use_stable_grpo}"
        )

    def _load_initial_weights(self, model, checkpoint_path: str) -> None:
        """Load model weights from HF checkpoint using DCP and state_dict_adapter.

        Args:
            model: The model to load weights into.
            checkpoint_path: Path to HF checkpoint directory.
        """
        if self.sd_adapter is None:
            logger.warning(
                "No state_dict_adapter available, skipping initial weight load"
            )
            return

        storage_reader = self.sd_adapter.get_hf_storage_reader(checkpoint_path)
        hf_state_dict = self.sd_adapter.to_hf(model.state_dict())
        dcp.load(hf_state_dict, storage_reader=storage_reader)
        torchtitan_state_dict = self.sd_adapter.from_hf(hf_state_dict)

        model_state_dict = dict(model.state_dict())

        # Convert to DTensor if target is DTensor (when the model is parallelized)
        for name, tensor in torchtitan_state_dict.items():
            if name in model_state_dict and isinstance(model_state_dict[name], DTensor):
                if isinstance(tensor, DTensor):
                    continue
                target_dtensor = model_state_dict[name]
                device_mesh = target_dtensor.device_mesh
                torchtitan_state_dict[name] = DTensor.from_local(
                    tensor.to(device_mesh.device_type),
                    device_mesh=device_mesh,
                    placements=target_dtensor.placements,
                )

        from torch.distributed.checkpoint.state_dict import (
            set_model_state_dict,
            StateDictOptions,
        )

        set_model_state_dict(
            model=model,
            model_state_dict=torchtitan_state_dict,
            options=StateDictOptions(strict=False),
        )
        logger.info(
            f"Loaded initial weights from {checkpoint_path} "
            f"({len(torchtitan_state_dict)} parameters)"
        )

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute GRPO advantages from rewards.

        Args:
            rewards: Reward tensor for all completions.

        Returns:
            Advantage tensor.
        """
        from torchtitan.experiments.rl.vllm_compat.simple_rl import (
            compute_grpo_advantages,
            compute_grpo_advantages_stable,
        )

        # Normalize rewards
        reward_mean = rewards.mean()
        reward_std = rewards.std()
        if reward_std > 1e-8:
            rewards_normalized = (rewards - reward_mean) / reward_std
        else:
            rewards_normalized = rewards - reward_mean

        if self.use_stable_grpo:
            return compute_grpo_advantages_stable(
                rewards_normalized, self.group_size
            )
        else:
            return compute_grpo_advantages(
                rewards_normalized, self.group_size, beta=self.grpo_beta
            )

    @endpoint
    async def get_weights(self) -> dict:
        """Get model weights for generator.

        Returns:
            model state dict with plain local tensors (DTensors unwrapped
            to avoid cross-mesh issues when transferring through Monarch).
        """
        titan_state = self.model.state_dict()

        # Unwrap DTensors to plain local tensors and clone to break shared storage.
        # Without clone, to_local() returns a view of the trainer's parameter data.
        # Since trainer and generator are collocated (same process), Monarch passes
        # by reference, so the generator's set_model_state_dict can corrupt the
        # trainer's Replicate params (norm weights) via in-place redistribution.
        return {
            k: v.to_local().clone() if isinstance(v, DTensor) else v.clone()
            for k, v in titan_state.items()
        }

    @endpoint
    async def step(self, episode: Episodes) -> dict:
        """Perform one training step.

        Computes advantages from rewards, then updates the policy.

        Args:
            episode: Trajectory data with rewards filled by Grader

        Returns:
            Training metrics
        """
        logger.info(
            f"{os.getpid()=} PolicyTrainer starts to train {self.policy_version} on traj:"
        )

        # Compute advantages from rewards
        advantages = self._compute_advantages(episode.rewards)

        # Compute reference log probs using frozen ref_model
        ref_token_log_probs = []
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for prompt_toks, gen_toks in zip(
                episode.prompt_token_ids, episode.vllm_token_ids
            ):
                token_lps = compute_token_log_probs(
                    self.ref_model, prompt_toks, gen_toks, device
                )
                ref_token_log_probs.append(token_lps)

        # Compute loss
        loss, loss_metrics, batch_token_log_probs = compute_policy_gradient_loss(
            self.model,
            episode.vllm_token_ids,
            episode.prompt_token_ids,
            advantages,
            ref_token_log_probs,
            kl_coef=0.1,
        )

        # Verify bitwise identity between vLLM and computed log probs
        verification_result = verify_logprob_identity(
            episode.vllm_token_log_probs,
            batch_token_log_probs,
        )
        logger.info(
            f"Logprob verification: bitwise_identical={verification_result['bitwise_identical']}, "
            f"max_delta={verification_result['max_delta']:.6e}, "
            f"avg_delta={verification_result['avg_delta']:.6e}, "
            f"tokens_checked={verification_result['total_tokens_checked']}"
        )

        # Update weights using torchtitan optimizers
        self.optimizers.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
        )

        self.optimizers.step()
        self.lr_schedulers.step()

        self.policy_version += 1

        # Return metrics
        metrics = {
            "loss": loss.item(),
            "reward_mean": episode.rewards.mean().item(),
            "reward_std": episode.rewards.std().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "sample_completion": episode.completions[0][:80],
            "policy_version": self.policy_version,
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
            "logprob_bitwise_identical": verification_result["bitwise_identical"],
            **loss_metrics,
        }
        logger.info(f"{os.getpid()=} PolicyTrainer finish step {self.policy_version}")
        return metrics
