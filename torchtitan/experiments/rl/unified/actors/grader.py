# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable

import torch
from monarch.actor import Actor, endpoint
from torchtitan.experiments.rl.unified.types import Episode

logger = logging.getLogger(__name__)


class Grader(Actor):
    """
    Evaluates completions and assigns rewards to episode data.

    The Grader receives episode data from the Generator
    and computes rewards using a reward function. Advantage computation
    is done by the Trainer.

    Args:
        reward_fn: Reward function that takes (completions: list[str], expected_answer: str)
                   and returns a tensor of rewards.
    """

    def __init__(
        self,
        reward_fn: Callable,
    ):
        self.reward_fn = reward_fn

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
        logger.info(
            f"Grader scoring {len(episodes)} episodes "
            f"(policy v{episodes[0].policy_version})..."
        )

        all_rewards = []
        for episode in episodes:
            completion_texts = [c.text for c in episode.completions]
            rewards = self.reward_fn(completion_texts, episode.expected_answer)
            for completion, reward in zip(episode.completions, rewards):
                completion.reward = reward.item()
            all_rewards.append(rewards)

        all_rewards_cat = torch.cat(all_rewards)
        reward_mean = all_rewards_cat.mean()
        reward_std = all_rewards_cat.std()

        logger.info(
            f"Grader finished scoring: "
            f"reward_mean={reward_mean.item():.4f}, reward_std={reward_std.item():.4f}"
        )

        return episodes
