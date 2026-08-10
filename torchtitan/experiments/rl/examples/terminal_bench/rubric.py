# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.experiments.rl.rollout import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn


class RewardTerminalBench(RewardFn):
    """Returns the reward emitted by the isolated Terminal-Bench verifier."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        if not rollout.turns:
            return 0.0
        return float(rollout.turns[-1].env_rewards.get("terminal_bench", 0.0))
