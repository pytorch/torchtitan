# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Optional

import torch
import torch.distributed.checkpoint as dcp

import torchtitan.protocols.train_spec as train_spec_module
from monarch.actor import Actor, endpoint
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.rl.unified.actors.grader import Episodes
from torchtitan.experiments.rl.unified.actors.utils import (
    compute_policy_gradient_loss,
    compute_token_log_probs,
    verify_logprob_identity,
)
from torchtitan.experiments.rl.unified.job_config import JobConfig
from torchtitan.experiments.rl.unified.models.utils import (
    replace_with_vllm_compatible_flash_attention,
)
from torchtitan.experiments.rl.vllm_compat.simple_rl import (
    compute_grpo_advantages,
    compute_grpo_advantages_stable,
)
from torchtitan.tools import utils

logger = logging.getLogger(__name__)


class Trainer(Actor):
    """
    Updates policy based on collected trajectories using TorchTitan components.

    This trainer uses TorchTitan's train_spec, ParallelDims, and optimizer
    components for model initialization and parallelization.

    Args:
        job_config: JobConfig dataclass containing all configuration
    """

    def __init__(
        self,
        job_config: JobConfig,
    ):
        self.job_config = job_config

        # GRPO config for advantage computation
        self.group_size = job_config.rl.grpo_group_size
        self.grpo_beta = job_config.rl.grpo_beta
        self.use_stable_grpo = job_config.rl.use_stable_grpo

        # Device setup
        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        # Initialize distributed
        world_size = dist_utils.init_distributed(job_config.comm)

        # Build parallel dims
        parallelism_config = job_config.parallelism
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

        # Get train spec for the model
        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)

        # Build model using train_spec
        model_args = self.train_spec.model_args[job_config.model.flavor]
        model_args.update_from_config(job_config)
        self.model_args = model_args
        self.hf_assets_path = job_config.model.hf_assets_path

        # Initialize state dict adapter for HF checkpoint loading
        if self.train_spec.state_dict_adapter is not None:
            self.sd_adapter = self.train_spec.state_dict_adapter(
                model_args, self.hf_assets_path
            )
        else:
            self.sd_adapter = None

        # Build model with meta init
        with torch.device("meta"):
            with utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]):
                model = self.train_spec.model_cls(model_args)

        # Replace attention with vLLM compatible attention for RL training
        # NOTE: We do this now for attention backward compatibility.
        # Long-term this will be replaced by pytorch attention supporting paged attention / kv cache
        replace_with_vllm_compatible_flash_attention(model)

        # Apply parallelization using train_spec's parallelize_fn
        # NOTE: Apply a temporary parallel plan for Qwen3, need to replace with the plan from TrainSpec
        from torchtitan.experiments.rl.unified.infra.parallelize import (
            parallelize_qwen3,
        )

        model = parallelize_qwen3(
            model, self.parallel_dims, job_config, is_trainer=True
        )

        # Initialize model weights on device
        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        # Load initial weights from checkpoint if specified
        if job_config.checkpoint.initial_load_path:
            self._load_initial_weights(model, job_config.checkpoint.initial_load_path)

        model.train()
        self.model = model
        self.model_parts = [model]

        # Create reference model for KL divergence (frozen copy of initial policy)
        # Build a separate model instance without DDP since it has no trainable params
        with torch.device("meta"):
            with utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]):
                ref_model = self.train_spec.model_cls(model_args)

        replace_with_vllm_compatible_flash_attention(ref_model)

        # Use parallelize_qwen3 (TP only, no FSDP) for ref_model since it's for inference only
        ref_model = parallelize_qwen3(
            ref_model, self.parallel_dims, job_config, is_trainer=True
        )

        ref_model.to_empty(device=device_type)
        with torch.no_grad():
            ref_model.init_weights(buffer_device=None)
        # Load weights from the trained model (use strict=False for parallelized state dict)
        if job_config.checkpoint.initial_load_path:
            self._load_initial_weights(
                ref_model, job_config.checkpoint.initial_load_path
            )
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()
        self.ref_model = ref_model

        # Build optimizer using train_spec
        self.optimizers = self.train_spec.build_optimizers_fn(
            self.model_parts, job_config.optimizer, self.parallel_dims
        )

        # Build LR schedulers using train_spec
        self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
            self.optimizers, job_config.lr_scheduler, job_config.training.steps
        )

        self.policy_version = 0
        self.generator: Optional[Any] = None

        logger.info(
            f"Trainer initialized with TorchTitan trainer, "
            f"model={job_config.model.name}, flavor={job_config.model.flavor}, "
            f"group_size={self.group_size}, grpo_beta={self.grpo_beta}, "
            f"use_stable_grpo={self.use_stable_grpo}"
        )

    def _load_initial_weights(self, model: torch.nn.Module, model_path: str) -> None:
        """Load initial weights from HuggingFace checkpoint using state dict adapter."""
        logger.info(f"Loading initial weights from {model_path}")

        if self.sd_adapter is None:
            raise RuntimeError(
                "Cannot load HF checkpoint: state_dict_adapter is not defined in train_spec"
            )

        # Get the current model state dict
        state_dict = get_model_state_dict(model)

        # Convert to HF format, load from checkpoint, then convert back
        hf_state_dict = self.sd_adapter.to_hf(state_dict)
        hf_storage_reader = self.sd_adapter.get_hf_storage_reader(model_path)

        dcp.load(hf_state_dict, storage_reader=hf_storage_reader)

        native_state_dict = self.sd_adapter.from_hf(hf_state_dict)
        set_model_state_dict(
            model,
            model_state_dict=native_state_dict,
            options=StateDictOptions(strict=False),
        )
        logger.info("Initial weights loaded successfully")

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages from rewards using GRPO. Normalizes rewards and computes group-relative advantages.

        Args:
            rewards: Raw rewards tensor [batch_size]

        Returns:
            Advantages tensor [batch_size]
        """
        # Normalize rewards for stability
        reward_mean = rewards.mean()
        reward_std = rewards.std()
        if reward_std > 1e-8:
            rewards_normalized = (rewards - reward_mean) / reward_std
        else:
            rewards_normalized = rewards - reward_mean

        # Compute advantages using GRPO
        if self.use_stable_grpo:
            advantages = compute_grpo_advantages_stable(
                rewards_normalized, self.group_size
            )
        else:
            advantages = compute_grpo_advantages(
                rewards_normalized, self.group_size, beta=self.grpo_beta
            )

        return advantages

    @endpoint
    async def get_weights(self) -> dict:
        """Get model weights for generator.

        Returns:
            model state dict
        """
        titan_state = self.model.state_dict()
        return titan_state

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
            f"{os.getpid()=} Trainer starts to train {self.policy_version} on traj:"
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

        # Update weights using torchtitan optimizers
        self.optimizers.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
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
        logger.info(f"{os.getpid()=} Trainer finish step {self.policy_version}")
        return metrics
