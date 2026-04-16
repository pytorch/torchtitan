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
    """
    Scores generated completions using a reward function.

    Scores each completion individually. Grouping for advantage
    computation lives in the controller, not here.

    Args:
        reward_fn: Callable ``(completions: list[str], expected_answer: str) -> torch.Tensor``
            returning one reward per input completion.
    """

    def __init__(
        self,
        reward_fn: Callable,
    ):
        self.reward_fn = reward_fn

        logger.info("Grader initialized")

    @endpoint
    async def score(
        self,
        completions: list[Completion],
        expected_answers: list[str],
    ) -> list[ScoredCompletion]:
        """Score completions and return ScoredCompletions in input order.

        Args:
            completions: Flat list of Completions to score.
            expected_answers: One expected answer per prompt, indexed by
                ``completion.prompt_idx``.

        Returns:
            Flat list of ScoredCompletions in input order.
        """
        # TODO: batch reward_fn across a prompt's n completions when the
        # reward function benefits from batching (e.g. a reward model).
        scored: list[ScoredCompletion] = []
        for c in completions:
            rewards = self.reward_fn([c.text], expected_answers[c.prompt_idx])
            scored.append(ScoredCompletion(completion=c, reward=rewards[0].item()))

        all_rewards = torch.tensor([sc.reward for sc in scored])
        logger.debug(
            f"Grader finished scoring {len(scored)} completions: "
            f"reward_mean={all_rewards.mean().item():.4f}, "
            f"reward_std={all_rewards.std().item():.4f}"
        )

        return scored
