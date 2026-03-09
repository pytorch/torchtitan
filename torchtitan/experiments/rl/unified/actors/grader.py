# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable

import torch
from monarch.actor import Actor, endpoint
from torchtitan.experiments.rl.unified.types import EpisodeGroup

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
    async def score(self, groups: list[EpisodeGroup]) -> list[EpisodeGroup]:
        """
        Score episodes by computing rewards.

        Each Group contains episodes from the same prompt. The reward_fn
        is called once per group with all completion texts.

        Args:
            groups: List of Groups (one per prompt), each containing
                    Episode objects to score.

        Returns:
            Groups with computed rewards.
        """
        total = sum(len(g) for g in groups)
        logger.debug(
            f"Grader scoring {total} episodes in {len(groups)} groups "
            f"(policy v{groups[0][0].policy_version})..."
        )

        all_rewards = []
        for group in groups:
            completion_texts = [ep.text for ep in group]
            expected_answer = group[0].expected_answer
            rewards = self.reward_fn(completion_texts, expected_answer)
            for episode, reward in zip(group, rewards):
                episode.reward = reward.item()
            all_rewards.append(rewards)

        all_rewards_cat = torch.cat(all_rewards)
        reward_mean = all_rewards_cat.mean()
        reward_std = all_rewards_cat.std()

        logger.debug(
            f"Grader finished scoring: "
            f"reward_mean={reward_mean.item():.4f}, reward_std={reward_std.item():.4f}"
        )

        return groups
