# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.experiments.rl.rollout.utils.rollout_to_episodes`."""

from __future__ import annotations

from torchtitan.experiments.rl.rollout import (
    Rollout,
    rollout_to_episodes,
    RolloutStatus,
    RolloutTurn,
)


def _turn(
    *,
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    version: int,
    content: str = "x",
) -> RolloutTurn:
    return RolloutTurn(
        prompt_token_ids=prompt_token_ids,
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


def test_single_turn_packs_one_episode() -> None:
    rollout = _scored_rollout(
        [_turn(prompt_token_ids=[1, 2], completion_token_ids=[4, 5], version=2)],
        reward=1.0,
        advantage=0.5,
    )
    [episode] = rollout_to_episodes(rollout)
    assert episode.token_ids == [1, 2, 4, 5]
    assert episode.loss_mask == [False, False, True, True]
    assert episode.logprobs == [0.0, 0.0, -0.1, -0.1]
    assert episode.policy_version == 2
    # advantage on the two completion tokens, 0.0 on the prompt
    assert episode.advantage == [0.0, 0.0, 0.5, 0.5]
    assert episode.sample_id == rollout.sample_id


def test_multiturn_with_growing_prefix_packs_into_one_episode() -> None:
    # Each turn's prompt extends the prior turn's prompt+completion with the env reply
    # ([8], then [9]), so the whole trajectory packs into one episode.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=2),
            _turn(
                prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[5, 6], version=2
            ),
            _turn(
                prompt_token_ids=[1, 2, 4, 8, 5, 6, 9],
                completion_token_ids=[7],
                version=2,
            ),
        ],
        reward=0.8,
        advantage=-0.2,
    )
    [episode] = rollout_to_episodes(rollout)
    #               P     P    a0    E1    a1    a1    E2    a2
    assert episode.token_ids == [1, 2, 4, 8, 5, 6, 9, 7]
    assert episode.loss_mask == [False, False, True, False, True, True, False, True]
    assert episode.logprobs == [0.0, 0.0, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1]
    # advantage broadcast onto every assistant token, 0.0 on prompt/env tokens
    assert episode.advantage == [0.0, 0.0, -0.2, 0.0, -0.2, -0.2, 0.0, -0.2]
    assert episode.sample_id == rollout.sample_id


def test_history_edit_branches_into_separate_episodes() -> None:
    # Turn 1's prompt does NOT extend turn 0's prompt+completion (the env rewrote history),
    # so the trajectory splits into two episodes.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[90, 91], completion_token_ids=[5], version=1),
        ],
        reward=0.5,
        advantage=0.1,
    )
    first, second = rollout_to_episodes(rollout)
    assert first.token_ids == [1, 2, 4]
    assert first.loss_mask == [False, False, True]
    assert first.sample_id == rollout.sample_id
    assert second.token_ids == [90, 91, 5]
    assert second.loss_mask == [False, False, True]
    assert second.sample_id == f"{rollout.sample_id}/branch=1"
    assert first.advantage == [0.0, 0.0, 0.1]
    assert second.advantage == [0.0, 0.0, 0.1]


def test_prompt_only_turn_is_skipped() -> None:
    # The last turn is prompt-only (empty completion), so only the first turn is packed.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[], version=1),
        ],
        reward=0.0,
        advantage=0.0,
    )
    [episode] = rollout_to_episodes(rollout)
    assert episode.token_ids == [1, 2, 4]
    assert episode.loss_mask == [False, False, True]


def test_none_advantage_defaults_to_zero() -> None:
    rollout = Rollout(
        group_id="g",
        sample_id="g/sample=0",
        status=RolloutStatus.COMPLETED,
        turns=[_turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1)],
        reward=1.0,
        advantage=None,
    )
    [episode] = rollout_to_episodes(rollout)
    assert episode.advantage == [0.0, 0.0, 0.0]  # None -> 0.0, broadcast over the token
