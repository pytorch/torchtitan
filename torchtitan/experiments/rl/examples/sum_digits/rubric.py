# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from dataclasses import dataclass

from torchtitan.experiments.rl.examples.sum_digits.data import SumDigitsExample

from torchtitan.experiments.rl.rollout import last_completion_text, Rollout
from torchtitan.experiments.rl.rubrics import RewardFn


_ANSWER_RE = re.compile(r"\[ANSWER\]\s*(-?\d+)")
_FORMAT_RE = re.compile(r"\[ANSWER\]\s*-?\d+")


class RewardCorrect(RewardFn):
    """`1.0` if the last `[ANSWER] <n>` equals the target, else `0.0`."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: SumDigitsExample) -> float:
        text = last_completion_text(rollout)
        matches = _ANSWER_RE.findall(text)
        if not matches:
            return 0.0
        return 1.0 if int(matches[-1]) == env_input.target else 0.0


class RewardFormat(RewardFn):
    """`1.0` if the response contains `[ANSWER] <n>`, else `0.0`."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        return 1.0 if _FORMAT_RE.search(last_completion_text(rollout)) else 0.0


__all__ = ["RewardCorrect", "RewardFormat"]
