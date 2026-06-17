# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""R2E-Gym reward: a thin reader of the env's grading signal.

The scoring already happened inside the env's terminal step (the sandbox is gone
by the time rubrics run), so this reward fn only SELECTS from ``env_rewards`` --
it never re-runs tests. Default = the 0/1 ``resolved`` signal. ``use_partial_credit``
opts into the per-test pass fraction so partially-correct rollouts get a small,
non-zero gradient even when nothing fully resolved.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.experiments.rl.examples.swe.data import R2EGymSample
from torchtitan.experiments.rl.rollout import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn


def _last_grade(rollout: Rollout) -> dict[str, float] | None:
    """The most recent turn's ``env_rewards`` that carry a grade (``resolved``).

    A rollout truncated mid-edit (token/turn budget) never reached the grading
    step, so no turn has ``resolved`` -> returns None -> reward 0.
    """
    for turn in reversed(rollout.turns):
        if "resolved" in (turn.env_rewards or {}):
            return turn.env_rewards
    return None


class RewardR2EGym(RewardFn):
    """Reward = the env's R2E-Gym grade. ``resolved`` (0/1) by default."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        use_partial_credit: bool = False
        """If True, an unresolved rollout earns its ``passed_frac`` (fraction of
        expected test statuses reproduced) instead of 0. Default = strict 0/1."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._use_partial_credit = config.use_partial_credit

    async def __call__(self, rollout: Rollout, env_input: R2EGymSample) -> float:
        del env_input  # grading already happened in the env; we only read it back
        grade = _last_grade(rollout)
        if grade is None:
            return 0.0
        if grade["resolved"] == 1.0:
            return 1.0
        if self._use_partial_credit:
            return grade.get("passed_frac", 0.0)
        return 0.0


__all__ = ["RewardR2EGym"]
