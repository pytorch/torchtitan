# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Rubric's truncation_penalty."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from torchtitan.rl.rollout.rubric import RewardFn, Rubric
from torchtitan.rl.rollout.types import RolloutStatus


class _PartialCredit(RewardFn):
    """Scores the rollout's `credit` attribute, e.g. the fraction of tests it passed."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout, env_input) -> float:
        return rollout.credit


def _score(rubric: Rubric, *rollouts) -> list[tuple[float, dict[str, float]]]:
    outputs = asyncio.run(rubric.score_group(list(rollouts), env_input=None))
    return [(output.reward, output.reward_breakdown) for output in outputs]


def _rollout(credit: float, status: RolloutStatus = RolloutStatus.COMPLETED):
    return SimpleNamespace(credit=credit, status=status)


def test_truncation_penalty_is_subtracted_from_the_partial_credit() -> None:
    rubric = Rubric.Config(
        reward_fns=[_PartialCredit.Config(weight=2.0)], truncation_penalty=0.2
    ).build()
    [(reward, breakdown)] = _score(
        rubric, _rollout(0.5, RolloutStatus.TRUNCATED_LENGTH)
    )
    assert reward == pytest.approx(0.3)
    assert breakdown == {"_PartialCredit": 0.5, "truncation_penalty": 0.2}


def test_truncation_penalty_leaves_finished_and_errored_rollouts_alone() -> None:
    rubric = Rubric.Config(
        reward_fns=[_PartialCredit.Config()], truncation_penalty=0.2
    ).build()
    finished, errored = _score(
        rubric,
        _rollout(0.5),
        _rollout(0.5, RolloutStatus.ERROR_TIMEOUT),
    )
    assert finished == (0.5, {"_PartialCredit": 0.5})
    assert errored == (0.5, {"_PartialCredit": 0.5})


def test_truncation_penalty_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="truncation_penalty"):
        Rubric.Config(reward_fns=[_PartialCredit.Config()], truncation_penalty=-0.2)
