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
from monarch.actor import Actor, endpoint
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.rl.unified.actors.generator import TrajectoryData
from torchtitan.experiments.rl.unified.configs import PolicyOptimizationConfig
from torchtitan.experiments.rl.unified.infra.parallelism_utils import (
    create_trainer_parallel_dims,
)
from torchtitan.experiments.rl.unified.models.utils import (
    replace_with_vllm_compatible_flash_attention,
)
from torchtitan.experiments.rl.vllm_compat.simple_rl import (
    compute_policy_gradient_loss_vllm,
)
from torchtitan.experiments.rl.vllm_compat.weights.converter import (
    torchtitan_to_vllm,
    vllm_to_torchtitan,
)
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


class PolicyTrainer(Actor, Configurable):
    """
    Updates policy based on collected Episodes.

    Run model forward on Episodes, computes loss, and run backward.
    Receives the top-level ``RLTrainer.Config`` and reads policy trainer
    settings (batch_invariant_mode, grpo) directly from it, plus model /
    optimizer / parallelism settings from the nested ``config.trainer``.

    TODO: Use torchtitan PolicyTrainer for model init and parallelism.

    Args:
        config: PolicyTrainer.Config for model/optimizer/parallelism settings.
        policy_optimization: GRPO hyperparameters.
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

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        policy_optimization: PolicyOptimizationConfig,
        batch_invariant_mode: bool,
    ):
        self.config = config
        self.model_spec = model_spec

        # Extract needed fields from config
        model_path = config.checkpoint.initial_load_path  # path to HF checkpoint
        learning_rate = config.optimizer.lr
        self.ddp_size = config.parallelism.data_parallel_replicate_degree
        self.tp_size = config.parallelism.tensor_parallel_degree

        # GRPO settings
        self.group_size = policy_optimization.group_size
        self.grpo_beta = policy_optimization.beta
        self.use_stable_grpo = policy_optimization.use_stable_grpo

        # Explicitly set cuda device for each trainer, otherwise different processes will use the same CUDA device
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)

        # Step1: Load trainer model from HF/vLLM checkpoint. TODO: Use torchtitan components
        model_config = model_spec.model
        titan_state_dict = vllm_to_torchtitan(model_path)

        # If weight tying is enabled but output.weight is missing from the checkpoint
        if model_config.enable_weight_tying and "output.weight" not in titan_state_dict:
            titan_state_dict["output.weight"] = titan_state_dict[
                "tok_embeddings.weight"
            ]

        self.model = model_config.build()
        self.model.load_state_dict(titan_state_dict, strict=True)

        # Step2: Replace attention kernel be to vLLM's attention.
        if batch_invariant_mode:
            replace_with_vllm_compatible_flash_attention(self.model)
            # vLLM's Attention requires bfloat16 inputs.
            # TODO: Refine the dtype journey in trainer / generator
            self.model.to(torch.bfloat16)

        self.parallel_dims = create_trainer_parallel_dims(self.ddp_size, self.tp_size)

        # apply PT-D Parallelism
        # TODO: right now it only works for qwen3 model, need to formalize this to use parallize_fn from model_spec
        if self.ddp_size > 1:
            from torchtitan.models.llama3.parallelize import apply_ddp

            apply_ddp(
                self.model,
                self.parallel_dims.get_mesh("dp_replicate"),
                enable_compile=False,
            )

        self.model = self.model.to(device)
        self.model.train()

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.policy_version = 0
        self.generator: Optional[Any] = None

        logger.debug(
            f"PolicyTrainer initialized: "
            f"group_size={self.group_size}, grpo_beta={self.grpo_beta}, "
            f"use_stable_grpo={self.use_stable_grpo}"
        )

    @endpoint
    async def get_weights(self) -> dict:
        """Get vLLM weights for generator.

        Returns:
            vLLM state dict
        """
        titan_state = self.model.state_dict()
        vllm_state = torchtitan_to_vllm(titan_state)
        return vllm_state

    @endpoint
    async def step(self, trajectory: TrajectoryData) -> dict:
        """Perform one training step.

        Returns:
            Training metrics
        """
        logger.debug(
            f"{os.getpid()=} PolicyTrainer starts to train {self.policy_version} on traj:"
        )
        # Compute loss
        loss, loss_metrics = compute_policy_gradient_loss_vllm(
            self.model,
            trajectory.vllm_token_ids,
            trajectory.vllm_token_log_probs,
            trajectory.prompt_token_ids,
            trajectory.advantages,
            kl_coef=0.1,
        )

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.policy_version += 1

        # Return metrics
        metrics = {
            "loss": loss.item(),
            "reward_mean": trajectory.rewards.mean().item(),
            "reward_std": trajectory.rewards.std().item(),
            "advantage_mean": trajectory.advantages.mean().item(),
            "advantage_std": trajectory.advantages.std().item(),
            "sample_completion": trajectory.completions[0][:80],
            "policy_version": self.policy_version,
            **loss_metrics,
        }
        logger.debug(f"{os.getpid()=} PolicyTrainer finish step {self.policy_version}")
        return metrics
