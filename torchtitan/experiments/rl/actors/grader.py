# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable

import torch
from monarch.actor import Actor, endpoint
from torchtitan.experiments.rl.types import Episode

logger = logging.getLogger(__name__)


class Grader(Actor):
    """
    Evaluates completions and assigns rewards to episodes.

    The Grader receives a flat list of Episodes and computes rewards
    using a reward function. It scores each episode independently.

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

        Calls the reward_fn with each episode's completion text and
        expected answer, then sets the reward on each episode.

        Args:
            episodes: Flat list of Episodes to score.

        Returns:
            Episodes with rewards filled in.
        """
        logger.debug(f"Grader scoring {len(episodes)} episodes...")

        # Score each episode individually
        for ep in episodes:
            rewards = self.reward_fn([ep.text], ep.expected_answer)
            ep.reward = rewards[0].item()

        all_rewards = torch.tensor([ep.reward for ep in episodes])
        logger.debug(
            f"Grader finished scoring: "
            f"reward_mean={all_rewards.mean().item():.4f}, "
            f"reward_std={all_rewards.std().item():.4f}"
        )

        return episodes
