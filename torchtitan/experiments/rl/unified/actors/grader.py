# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Optional

import torch
from monarch.actor import Actor, endpoint
from torchtitan.experiments.rl.unified.types import Episode
from torchtitan.experiments.rl.vllm_compat.simple_rl import trivial_reward_function

logger = logging.getLogger(__name__)


class Grader(Actor):
    """
    Evaluates completions and assigns rewards to episode data.

    The Grader receives episode data from the Generator
    and computes rewards using a reward function. Advantage computation
    is done by the Trainer.

    Args:
        reward_fn: Optional custom reward function. If not provided,
                   uses trivial_reward_function from simple_rl.
    """

    def __init__(
        self,
        reward_fn: Optional[Callable] = None,
    ):
        # Set reward function
        self.reward_fn = reward_fn if reward_fn is not None else trivial_reward_function

        logger.info("Grader initialized")

    @endpoint
    async def score(self, episodes: list[Episode]) -> list[Episode]:
        """
        Score episodes by computing rewards.

        Args:
            episodes: List of Episode data (one per prompt, with completions)

        Returns:
            Episodes with computed rewards
        """
        policy_version = episodes[0].policy_version if episodes else -1
        logger.info(
            f"Grader scoring {len(episodes)} episodes (policy v{policy_version})..."
        )

        all_rewards = []
        for episode in episodes:
            completion_texts = [c.text for c in episode.completions]
            rewards = self.reward_fn(completion_texts, episode.expected_answer)
            episode.rewards = rewards
            all_rewards.append(rewards)

        all_rewards_cat = torch.cat(all_rewards)
        reward_mean = all_rewards_cat.mean()
        reward_std = all_rewards_cat.std()

        logger.info(
            f"Grader finished scoring: "
            f"reward_mean={reward_mean.item():.4f}, reward_std={reward_std.item():.4f}"
        )

        return episodes
