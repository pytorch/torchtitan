# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Optional

import torch
from monarch.actor import Actor, endpoint
from torchtitan.experiments.rl.unified.actors.generator import TrajectoryData
from torchtitan.experiments.rl.unified.configs import RLTrainer
from torchtitan.experiments.rl.unified.infra.parallelism_utils import (
    create_trainer_parallel_dims,
)
from torchtitan.experiments.rl.unified.models.utils import load_trainer_model
from torchtitan.experiments.rl.vllm_compat.simple_rl import (
    compute_policy_gradient_loss_vllm,
)

logger = logging.getLogger(__name__)


class Trainer(Actor):
    """
    Updates policy based on collected trajectories.

    Run model forward on trajectories, computes loss, and run backward.
    Receives the top-level ``RLTrainer.Config`` and reads policy trainer
    settings (batch_invariant_mode, grpo) directly from it, plus model /
    optimizer / parallelism settings from the nested ``config.trainer``.

    TODO: Use torchtitan Trainer for model init and parallelisation.

    Args:
        config: Top-level RLTrainer.Config containing all configuration.
    """

    def __init__(
        self,
        config: RLTrainer.Config,
    ):
        self.config = config
        trainer_cfg = config.trainer

        # Extract needed fields from config
        model_path = trainer_cfg.checkpoint.initial_load_path  # path to HF checkpoint
        learning_rate = trainer_cfg.optimizer.lr
        self.ddp_size = trainer_cfg.parallelism.data_parallel_replicate_degree
        self.tp_size = trainer_cfg.parallelism.tensor_parallel_degree

        # GRPO settings from top-level config
        self.group_size = config.policy_optimization.group_size
        self.grpo_beta = config.policy_optimization.beta
        self.use_stable_grpo = config.policy_optimization.use_stable

        # Explicitly set cuda device for each trainer, otherwise different processes will use the same CUDA device
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)

        # load trainer model and patch to vllm.Attention()
        self.model = load_trainer_model(model_path)
        self.parallel_dims = create_trainer_parallel_dims(self.ddp_size, self.tp_size)

        # apply PT-D Parallelism
        # TODO: right now it only works for qwen3 model, need to formalize this to use parallize_fn from train_spec
        if self.ddp_size > 1:
            from torchtitan.models.llama3.infra.parallelize import apply_ddp

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

        logger.info(
            f"Trainer initialized: "
            f"group_size={self.group_size}, grpo_beta={self.grpo_beta}, "
            f"use_stable_grpo={self.use_stable_grpo}"
        )

    @endpoint
    async def get_weights(self) -> dict:
        """Get model weights for generator.

        Returns:
            model state dict
        """
        titan_state = self.model.state_dict()
        return titan_state

    @endpoint
    async def step(self, trajectory: TrajectoryData) -> dict:
        """Perform one training step.

        Returns:
            Training metrics
        """
        logger.info(
            f"{os.getpid()=} Trainer starts to train {self.policy_version} on traj:"
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
        logger.info(f"{os.getpid()=} Trainer finish step {self.policy_version}")
        return metrics
