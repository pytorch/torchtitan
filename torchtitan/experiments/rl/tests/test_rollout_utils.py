# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.experiments.rl.rollout.utils.rollout_to_episode`."""

from __future__ import annotations

from torchtitan.experiments.rl.rollout import (
    Rollout,
    rollout_to_episode,
    RolloutStatus,
    RolloutTurn,
)


def _turn(
    *, completion_token_ids: list[int], content: str, version: int
) -> RolloutTurn:
    return RolloutTurn(
        prompt_token_ids=[1, 2],
        completion_token_ids=completion_token_ids,
        completion_logprobs=[-0.1] * len(completion_token_ids),
        policy_version=version,
        completion_message={"role": "assistant", "content": content},
    )


def _scored_rollout(
    turns: list[RolloutTurn], *, reward: float, advantage: float
) -> Rollout:
    return Rollout(
        group_id="step=1/group=0",
        sample_id="step=1/group=0/sample=0",
        status=RolloutStatus.COMPLETED,
        turns=turns,
        reward=reward,
        advantage=advantage,
    )


def test_single_turn_yields_one_episode() -> None:
    rollout = _scored_rollout(
        [_turn(completion_token_ids=[4, 5], content="a b", version=2)],
        reward=1.0,
        advantage=0.5,
    )
    episode = rollout_to_episode(rollout)
    assert episode.prompt_token_ids == [1, 2]
    assert episode.completion_token_ids == [4, 5]
    assert episode.completion_text == "a b"
    assert episode.policy_version == 2
    assert episode.reward == 1.0 and episode.advantage == 0.5


def test_multiturn_yields_one_episode_from_last_turn() -> None:
    # TODO(prefix-matching): once turns pack into one sequence, this trains all turns; for now
    # it's the last turn (its prompt holds the full history).
    rollout = _scored_rollout(
        [
            _turn(completion_token_ids=[4], content="t0", version=2),
            _turn(completion_token_ids=[5, 6], content="t1", version=2),
            _turn(completion_token_ids=[7], content="t2", version=3),
        ],
        reward=0.8,
        advantage=-0.2,
    )
    episode = rollout_to_episode(rollout)
    assert episode.completion_text == "t2"
    assert episode.completion_token_ids == [7]
    assert episode.policy_version == 3
    assert episode.reward == 0.8 and episode.advantage == -0.2


def test_uses_last_turn_that_has_a_completion() -> None:
    # The last turn is prompt-only (empty completion), so the prior trainable turn is used.
    rollout = _scored_rollout(
        [
            _turn(completion_token_ids=[4], content="kept", version=1),
            _turn(completion_token_ids=[], content="", version=1),  # prompt-only/empty
        ],
        reward=0.0,
        advantage=0.0,
    )
    episode = rollout_to_episode(rollout)
    assert episode.completion_text == "kept"
    assert episode.completion_token_ids == [4]


def test_none_advantage_defaults_to_zero() -> None:
    rollout = Rollout(
        group_id="g",
        sample_id="g/sample=0",
        status=RolloutStatus.COMPLETED,
        turns=[_turn(completion_token_ids=[4], content="x", version=1)],
        reward=1.0,
        advantage=None,
    )
    episode = rollout_to_episode(rollout)
    assert episode.advantage == 0.0
