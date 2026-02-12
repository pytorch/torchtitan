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
from torchtitan.experiments.rl.unified.infra.parallelism_utils import (
    create_trainer_parallel_dims,
)
from torchtitan.experiments.rl.unified.models.utils import load_model, ModelMode
from torchtitan.experiments.rl.vllm_compat.simple_rl import (
    compute_policy_gradient_loss_vllm,
)
from torchtitan.experiments.rl.vllm_compat.weights_vllm_compat import (
    torchtitan_to_vllm_compat,
)

logger = logging.getLogger(__name__)


class Trainer(Actor):
    """
    Updates policy based on collected trajectories.

    Run model forward on trajectories, computes loss, and run backward.

    Args:
        titan_checkpoint_path: Path to TorchTitan checkpoint
        model_path: Path to HuggingFace model
        learning_rate: Learning rate for optimizer
        model_mode: Indicates which model to use. Train inferece unified model, batch invariant Torchtitan model,
            or plain Torchtitan model
    """

    def __init__(
        self,
        titan_checkpoint_path: str,
        model_path: str,
        learning_rate: float = 1e-5,
        model_mode: str = ModelMode.VLLM_COMPAT,
        ddp_size: int = 1,
        tp_size: int = 1,
    ):
        # Explicitly set cuda device for each trainer, otherwise different processes will use the same CUDA device
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)

        self.model = load_model(
            titan_checkpoint_path, model_path, model_mode=model_mode
        )
        self.ddp_size = ddp_size
        self.tp_size = tp_size
        self.parallel_dims = create_trainer_parallel_dims(self.ddp_size, self.tp_size)

        # apply PT-D Parallelism
        # TODO: right now it only works for qwen3 model, need to formalize this to use parallelize_fn from ModelSpec
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

        logger.info("Trainer initialized with TorchTitan model")

    @endpoint
    async def get_weights(self) -> dict:
        """Get vLLM-compatible weights for generator.

        Returns:
            vLLM-compatible state dict
        """
        titan_state = self.model.state_dict()
        vllm_compat_state = torchtitan_to_vllm_compat(titan_state)
        return vllm_compat_state

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

        # TODO: save dcp checkpoint to file here instead of sending weight dicts

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
