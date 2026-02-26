# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from monarch.actor import Actor, endpoint
from torchtitan.experiments.rl.unified.configs import GRPOConfig
from torchtitan.experiments.rl.vllm_compat.simple_rl import trivial_reward_function

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """
    Data from one generation batch.

    Attributes:
        policy_version: Version of policy that produced this batch
        completions: List of completion strings
        vllm_token_ids: List of token ID lists for each completion
        vllm_token_log_probs: List of per-token log prob lists.
            This is used for checking bit-wise identity between trainer and generator.
        prompt_token_ids: List of prompt token ID lists
        expected_answers: List of expected answers for reward computation
        rewards: Rewards for each completion (initialized to zeros, filled by Grader)
    """

    policy_version: int
    completions: List[str]
    vllm_token_ids: List[List[int]]
    vllm_token_log_probs: List[List[float]]
    prompt_token_ids: List[List[int]]
    expected_answers: List[str]
    rewards: torch.Tensor


class Grader(Actor):
    """
    Evaluates completions and assigns rewards to episode data.

    The Grader receives episode data from the Generator
    and computes rewards using a reward function. Advantage computation
    is done by the Trainer.

    Args:
        grpo_config: GRPO config containing group_size.
        reward_fn: Optional custom reward function. If not provided,
                   uses trivial_reward_function from simple_rl.
    """

    def __init__(
        self,
        grpo_config: GRPOConfig,
        reward_fn: Optional[Callable] = None,
    ):
        self.group_size = grpo_config.group_size

        # Set reward function
        self.reward_fn = reward_fn if reward_fn is not None else trivial_reward_function

        logger.info(f"Grader initialized with group_size={self.group_size}")

    @endpoint
    async def score(self, episode: Episode) -> Episode:
        """
        Score an episode by computing rewards.

        Args:
            episode: Episode data (with or without rewards)

        Returns:
            Episode with computed rewards
        """
        logger.info(f"Grader scoring episode (policy v{episode.policy_version})...")

        # Compute rewards using reward function
        rewards = self.reward_fn(
            episode.completions,
            episode.expected_answers,
            self.group_size,
        )

        reward_mean = rewards.mean()
        reward_std = rewards.std()

        # Update episode with rewards
        episode.rewards = rewards

        logger.info(
            f"Grader finished scoring: "
            f"reward_mean={reward_mean.item():.4f}, reward_std={reward_std.item():.4f}"
        )

        return episode

    @endpoint
    async def set_reward_fn(self, reward_fn: Callable) -> None:
        """
        Update the reward function.

        Args:
            reward_fn: New reward function to use
        """
        self.reward_fn = reward_fn
        logger.info("Grader reward function updated")
