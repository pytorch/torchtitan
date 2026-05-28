# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from dataclasses import dataclass

from torchtitan.experiments.rl.recipes.sum_digits.data import SumDigitsInput
from torchtitan.experiments.rl.rollouts import last_assistant_text, Rollout
from torchtitan.experiments.rl.rubrics import RewardFn, Rubric


_ANSWER_RE = re.compile(r"\[ANSWER\]\s*(-?\d+)")
_FORMAT_RE = re.compile(r"\[ANSWER\]\s*-?\d+")


async def reward_correct(rollout: Rollout, env_input: SumDigitsInput) -> float:
    """Reward exact final answer match.

    Args:
        rollout: Rollout containing the assistant response to grade.
        env_input: SumDigits payload with the target digit sum.

    Returns:
        `1.0` when the last `[ANSWER] <n>` equals the target, else `0.0`.
    """
    text = last_assistant_text(rollout)
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return 0.0
    return 1.0 if int(matches[-1]) == env_input.target else 0.0


async def reward_format(rollout: Rollout, _env_input: object) -> float:
    """Reward presence of an answer tag.

    Args:
        rollout: Rollout containing the assistant response to grade.
        _env_input: Unused; accepted to satisfy the reward-fn interface.

    Returns:
        `1.0` when the response contains `[ANSWER] <n>`, else `0.0`.
    """
    return 1.0 if _FORMAT_RE.search(last_assistant_text(rollout)) else 0.0


class SumDigitsRubric(Rubric):
    """SumDigits rubric: weighted `reward_correct` + `reward_format`."""

    @dataclass(kw_only=True, slots=True)
    class Config(Rubric.Config):
        """Config for `SumDigitsRubric`.

        Args:
            correctness_weight: Weight on the answer-matches-target reward.
            format_weight: Weight on the `[ANSWER] <n>` tag-presence reward.
        """

        correctness_weight: float = 1.0
        format_weight: float = 0.3

    def register_funcs(self) -> list[RewardFn]:
        cfg = self._config
        return [
            RewardFn(fn=reward_correct, weight=cfg.correctness_weight),
            RewardFn(fn=reward_format, weight=cfg.format_weight),
        ]


__all__ = ["SumDigitsRubric", "reward_correct", "reward_format"]
