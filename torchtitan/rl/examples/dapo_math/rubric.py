# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.rl.examples.dapo_math.data import DapoMathSample
from torchtitan.rl.examples.dapo_math.grader import MathVerifyPool
from torchtitan.rl.rollout import Rollout
from torchtitan.rl.rubric import RewardFn


class RewardMathVerify(RewardFn):
    """Binary reward for a mathematically equivalent final answer."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        timeout_seconds: float = 5.0
        """Limit to score one answer, counted from when a grader process receives it
        (queue wait and process start excluded); a slower answer scores 0. Most answers
        take 2-20 ms; the slowest symbolic answers we measured took ~4 s."""

        num_processes: int = 4
        """Grader processes per rollout worker (~70 MB each); each scores one answer at a time."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._pool = MathVerifyPool(
            num_processes=config.num_processes, timeout_seconds=config.timeout_seconds
        )

    async def __call__(self, rollout: Rollout, env_input: DapoMathSample) -> float:
        """Return 1 when Math-Verify equates the response and ground truth."""
        if not rollout.turns:
            return 0.0
        completion_message = rollout.turns[-1].completion_message
        response = (
            (completion_message.get("content") or "") if completion_message else ""
        )
        return await self._pool.score(
            response=response, ground_truth=env_input.ground_truth
        )
