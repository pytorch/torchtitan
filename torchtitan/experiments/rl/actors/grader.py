# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable

from monarch.actor import Actor, endpoint
from torchtitan.experiments.rl.types import Completion, ScoredCompletion

logger = logging.getLogger(__name__)


class Grader(Actor):
    """
    Scores a group of completions sharing a prompt.

    One call scores one prompt's completions so ``reward_fn`` sees all
    n samples at once -- its natural batched shape. The controller
    loops over prompts and handles advantage grouping.

    Args:
        reward_fn: Callable
            ``(completions: list[Completion], expected_answer: str) -> torch.Tensor``
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
        expected_answer: str,
    ) -> list[ScoredCompletion]:
        """Score one prompt's completions.

        Args:
            completions: Completions generated for a single prompt.
            expected_answer: Expected answer for that prompt.

        Returns:
            ScoredCompletions in input order.
        """
        rewards = self.reward_fn(completions, expected_answer)
        scored = [
            ScoredCompletion(completion=c, reward=r.item())
            for c, r in zip(completions, rewards)
        ]

        reward_mean = sum(sc.reward for sc in scored) / len(scored)
        logger.debug(
            f"Grader scored {len(scored)} completions for prompt "
            f"{completions[0].prompt_idx}: reward_mean={reward_mean:.4f}"
        )

        return scored
