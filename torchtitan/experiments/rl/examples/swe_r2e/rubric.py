# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reward for the R2E coding-agent example.

Grading an R2E patch requires booting a fresh sandbox and running hidden tests,
which the rollouter already does while collecting the rollout. So the reward fn
does not re-grade: ``SWER2ERollouter`` stamps the grade onto the rollout's last
turn ``env_rewards`` (key ``r2e_reward``) and this fn simply reads it back. This
keeps the reward on the standard rubric/advantage path (and in the reward
breakdown metric) without duplicating the expensive sandbox eval.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.experiments.rl.rollout.types import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn

# env_rewards key the rollouter stamps the R2E grade under.
R2E_REWARD_KEY = "r2e_reward"


class RewardR2E(RewardFn):
    """Reads the R2E test-pass reward (0.0/1.0) the rollouter attached to the rollout."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        for turn in reversed(rollout.turns):
            if R2E_REWARD_KEY in turn.env_rewards:
                return float(turn.env_rewards[R2E_REWARD_KEY])
        return 0.0
