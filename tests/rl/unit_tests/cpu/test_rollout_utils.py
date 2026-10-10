# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `TrainingSampleBuilder.rollout_to_training_samples`."""

from __future__ import annotations

import pytest
import torch

from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
from torchtitan.rl.rollout import Rollout, RolloutGroup, RolloutStatus, RolloutTurn
from torchtitan.rl.types import RolloutTurnID

_GROUP_ID = "step=1/group=0"

# rollout_to_training_samples is the TrainingSampleBuilder override hook; the transform is config-independent.
rollout_to_training_samples = (
    TrainingSampleBuilder.Config().build().rollout_to_training_samples
)


def _turn(
    *,
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    version: int,
    max_version: int | None = None,
    content: str = "x",
) -> RolloutTurn:
    return RolloutTurn(
        rollout_id=RolloutTurnID(group_id=_GROUP_ID, rollout_id=0, turn_id=0),
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        completion_logprobs=[-0.1] * len(completion_token_ids),
        min_policy_version=version,
        max_policy_version=version if max_version is None else max_version,
        completion_message={"role": "assistant", "content": content},
    )


def _scored_rollout(
    turns: list[RolloutTurn], *, reward: float, advantage: float
) -> Rollout:
    return Rollout(
        group_id=_GROUP_ID,
        rollout_id=0,
        status=RolloutStatus.COMPLETED,
        turns=turns,
        reward=reward,
        advantage=advantage,
    )


def test_single_turn_packs_one_training_sample() -> None:
    rollout = _scored_rollout(
        [_turn(prompt_token_ids=[1, 2], completion_token_ids=[4, 5], version=2)],
        reward=1.0,
        advantage=0.5,
    )
    [training_sample] = rollout_to_training_samples(rollout)
    assert training_sample.token_ids == [1, 2, 4, 5]
    assert training_sample.loss_mask == [False, False, True, True]
    assert training_sample.logprobs == [0.0, 0.0, -0.1, -0.1]
    assert training_sample.min_policy_version == 2
    # advantage on the two completion tokens, 0.0 on the prompt
    assert training_sample.advantage == [0.0, 0.0, 0.5, 0.5]
    assert training_sample.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=0
    )
    # single turn at version 2 -> min == max == 2
    assert training_sample.max_policy_version == 2


def test_packs_min_and_max_version_across_turns() -> None:
    # Two turns sampled at different versions (a weight pull landed between them); the packed
    # training_sample records the min (oldest, turn-0) and max (newest, turn-1) version.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4, 5], version=3),
            _turn(
                prompt_token_ids=[1, 2, 4, 5, 9], completion_token_ids=[7], version=4
            ),
        ],
        reward=1.0,
        advantage=0.5,
    )
    [training_sample] = rollout_to_training_samples(rollout)
    assert training_sample.token_ids == [1, 2, 4, 5, 9, 7]
    assert training_sample.min_policy_version == 3  # oldest (turn 0)
    assert training_sample.max_policy_version == 4  # newest (turn 1)


def test_multiturn_with_growing_prefix_packs_into_one_training_sample() -> None:
    # Each turn's prompt extends the prior turn's prompt+completion with the env reply
    # ([8], then [9]), so the whole trajectory packs into one training_sample.
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
    [training_sample] = rollout_to_training_samples(rollout)
    #               P     P    a0    E1    a1    a1    E2    a2
    assert training_sample.token_ids == [1, 2, 4, 8, 5, 6, 9, 7]
    assert training_sample.loss_mask == [
        False,
        False,
        True,
        False,
        True,
        True,
        False,
        True,
    ]
    assert training_sample.logprobs == [0.0, 0.0, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1]
    # advantage broadcast onto every assistant token, 0.0 on prompt/env tokens
    assert training_sample.advantage == [0.0, 0.0, -0.2, 0.0, -0.2, -0.2, 0.0, -0.2]
    assert training_sample.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=0
    )


def test_history_edit_branches_into_separate_training_samples() -> None:
    # Turn 1's prompt does NOT extend turn 0's prompt+completion (the env rewrote history),
    # so the trajectory splits into two training_samples.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[90, 91], completion_token_ids=[5], version=1),
        ],
        reward=0.5,
        advantage=0.1,
    )
    first, second = rollout_to_training_samples(rollout)
    assert first.token_ids == [1, 2, 4]
    assert first.loss_mask == [False, False, True]
    # first segment opens at turn 0, the branch (history edit) opens at turn 1
    assert first.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=0
    )
    assert second.token_ids == [90, 91, 5]
    assert second.loss_mask == [False, False, True]
    assert second.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=1
    )
    assert first.advantage == [0.0, 0.0, 0.1]
    assert second.advantage == [0.0, 0.0, 0.1]


def test_empty_completion_on_a_later_turn_raises() -> None:
    # An empty completion is only expected on the first turn (initial prompt too long). On any later
    # turn it is an anomaly, so building the training samples raises.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[], version=1),
        ],
        reward=0.0,
        advantage=0.0,
    )
    with pytest.raises(ValueError, match="no completion"):
        rollout_to_training_samples(rollout)


def test_empty_completion_on_first_turn_is_skipped() -> None:
    # First turn with no completion = initial prompt too long; expected, so the rollout yields no samples.
    rollout = _scored_rollout(
        [_turn(prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[], version=1)],
        reward=0.0,
        advantage=0.0,
    )
    assert rollout_to_training_samples(rollout) == []


def test_group_filters_preserve_group_id_for_acknowledgement() -> None:
    builder = TrainingSampleBuilder.Config().build()

    failed = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=7, rollouts=[])
    )

    empty_completion = _scored_rollout(
        [_turn(prompt_token_ids=[1], completion_token_ids=[], version=1)],
        reward=0.0,
        advantage=0.0,
    )
    empty_completion.group_id = 8
    untrainable = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=8, rollouts=[empty_completion])
    )

    first = _scored_rollout(
        [_turn(prompt_token_ids=[1], completion_token_ids=[2], version=1)],
        reward=1.0,
        advantage=0.0,
    )
    second = _scored_rollout(
        [_turn(prompt_token_ids=[1], completion_token_ids=[3], version=1)],
        reward=1.0,
        advantage=0.0,
    )
    first.group_id = second.group_id = 9
    zero_std = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=9, rollouts=[first, second])
    )

    invalid_logprob = _scored_rollout(
        [_turn(prompt_token_ids=[1], completion_token_ids=[2], version=1)],
        reward=1.0,
        advantage=1.0,
    )
    invalid_logprob.group_id = 10
    invalid_logprob.turns[0].completion_logprobs = [float("nan")]
    no_valid_tokens = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=10, rollouts=[invalid_logprob])
    )

    filtered_groups = [failed, untrainable, zero_std, no_valid_tokens]
    assert [group.group_id for group in filtered_groups] == [7, 8, 9, 10]
    assert all(not group.training_samples for group in filtered_groups)


def _routed_expert_ids(*, turn: int, num_positions: int) -> torch.Tensor:
    """Rows tagged ``16 * turn + position``, so a test can tell which turn and position a row came from."""
    return (16 * turn + torch.arange(num_positions, dtype=torch.uint8)).view(-1, 1, 1)


def test_routed_expert_ids_keep_each_turns_rows_and_take_the_boundary_from_the_next_prefill() -> (
    None
):
    turns = [
        _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=2),
        _turn(prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[5, 6], version=2),
        _turn(
            prompt_token_ids=[1, 2, 4, 8, 5, 6, 9], completion_token_ids=[7], version=2
        ),
    ]
    # vLLM returns one row per forward input: the prompt plus every completion token but the last.
    for turn_id, rollout_turn in enumerate(turns):
        num_inputs = (
            len(rollout_turn.prompt_token_ids)
            + len(rollout_turn.completion_token_ids)
            - 1
        )
        rollout_turn.routed_expert_ids = _routed_expert_ids(
            turn=turn_id, num_positions=num_inputs
        )
    rollout = _scored_rollout(turns, reward=0.8, advantage=-0.2)

    [training_sample] = rollout_to_training_samples(rollout)

    # inputs:           1       2       4       8       5       6       9
    # Token 4 (turn 0's last completion token) and token 6 (turn 1's) never ran forward in
    # their own turn, so their rows come from the next turn's prefill.
    assert training_sample.routed_expert_ids.flatten().tolist() == [
        16 * 0 + 0,
        16 * 0 + 1,
        16 * 1 + 2,
        16 * 1 + 3,
        16 * 1 + 4,
        16 * 2 + 5,
        16 * 2 + 6,
    ]
    assert len(training_sample.routed_expert_ids) == len(training_sample.token_ids) - 1


def test_routed_expert_ids_restart_at_a_branch() -> None:
    turns = [
        _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
        _turn(prompt_token_ids=[90, 91], completion_token_ids=[5], version=1),
    ]
    turns[0].routed_expert_ids = _routed_expert_ids(turn=0, num_positions=2)
    turns[1].routed_expert_ids = _routed_expert_ids(turn=1, num_positions=2)
    rollout = _scored_rollout(turns, reward=0.5, advantage=0.1)

    first, second = rollout_to_training_samples(rollout)

    assert first.routed_expert_ids.flatten().tolist() == [0, 1]
    assert second.routed_expert_ids.flatten().tolist() == [16, 17]


def test_samples_without_routed_expert_ids_keep_none() -> None:
    rollout = _scored_rollout(
        [_turn(prompt_token_ids=[1, 2], completion_token_ids=[4, 5], version=2)],
        reward=1.0,
        advantage=0.5,
    )

    [training_sample] = rollout_to_training_samples(rollout)

    assert training_sample.routed_expert_ids is None
