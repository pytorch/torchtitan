# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable

import torch
from monarch.actor import Actor, endpoint
from torchtitan.experiments.rl.types import Completion, ScoredCompletion

logger = logging.getLogger(__name__)


class Grader(Actor):
    """Evaluates completions and assigns rewards.

    The Grader receives a flat list of Completions and a corresponding list
    of expected answers (one per completion, pre-expanded by the orchestrator).
    It scores each completion independently and returns ScoredCompletions.

    Args:
        reward_fn: Reward function that takes (completions: list[str], expected_answer: str)
                   and returns a tensor of rewards.
    """

    def __init__(self, reward_fn: Callable):
        self.reward_fn = reward_fn
        logger.info("Grader initialized")

    @endpoint
    async def score(
        self,
        completions: list[Completion],
        expected_answers: list[str],
    ) -> list[ScoredCompletion]:
        """Score completions by computing rewards.

        Args:
            completions: Flat list of Completions to score.
            expected_answers: Expected answer per completion (pre-expanded
                by the orchestrator so len matches completions).

        Returns:
            List of ScoredCompletions with rewards filled in.
        """
        logger.debug(f"Grader scoring {len(completions)} completions...")

        scored = []
        for comp, answer in zip(completions, expected_answers):
            rewards = self.reward_fn([comp.text], answer)
            scored.append(ScoredCompletion(completion=comp, reward=rewards[0].item()))

        all_rewards = torch.tensor([sc.reward for sc in scored])
        logger.debug(
            f"Grader finished scoring: "
            f"reward_mean={all_rewards.mean().item():.4f}, "
            f"reward_std={all_rewards.std().item():.4f}"
        )

        return scored
