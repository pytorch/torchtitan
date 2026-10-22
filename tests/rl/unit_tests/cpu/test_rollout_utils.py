# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `TrainingSampleBuilder.rollout_to_training_samples` and `split_prompt`."""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest
from torchtitan.rl.rollout import Rollout, RolloutGroup, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.types import RolloutTurnID, split_prompt

from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder

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
        # The full prompt for now; `_scored_rollout` stores it as a delta like the Rollouter does
        prompt_prefix_len=0,
        prompt_delta_token_ids=prompt_token_ids,
        completion_token_ids=np.array(completion_token_ids, dtype=np.int32),
        completion_logprobs=np.full(len(completion_token_ids), -0.1, dtype=np.float32),
        min_policy_version=version,
        max_policy_version=version if max_version is None else max_version,
        completion_message={"role": "assistant", "content": content},
    )


def _scored_rollout(
    turns: list[RolloutTurn], *, reward: float, advantage: float
) -> Rollout:
    previous_token_ids: list[int] = []
    for rollout_turn in turns:
        prompt_token_ids = rollout_turn.prompt_delta_token_ids
        prefix_len, delta = split_prompt(prompt_token_ids, previous_token_ids)
        rollout_turn.prompt_prefix_len = prefix_len
        rollout_turn.prompt_delta_token_ids = delta
        previous_token_ids = (
            prompt_token_ids + rollout_turn.completion_token_ids.tolist()
        )
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
    assert training_sample.token_ids.tolist() == [1, 2, 4, 5]
    assert training_sample.loss_mask.tolist() == [False, False, True, True]
    assert training_sample.logprobs.tolist() == pytest.approx([0.0, 0.0, -0.1, -0.1])
    assert training_sample.min_policy_version == 2
    # advantage on the two completion tokens, 0.0 on the prompt
    assert training_sample.advantage.tolist() == pytest.approx([0.0, 0.0, 0.5, 0.5])
    assert training_sample.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=0
    )
    # single turn at version 2 -> min == max == 2
    assert training_sample.max_policy_version == 2


def test_packs_min_and_max_version_across_turns() -> None:
    # Two turns sampled at different versions (a model state dict load landed between them); the packed
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
    assert training_sample.token_ids.tolist() == [1, 2, 4, 5, 9, 7]
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
    assert training_sample.token_ids.tolist() == [1, 2, 4, 8, 5, 6, 9, 7]
    assert training_sample.loss_mask.tolist() == [
        False,
        False,
        True,
        False,
        True,
        True,
        False,
        True,
    ]
    assert training_sample.logprobs.tolist() == pytest.approx(
        [0.0, 0.0, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1]
    )
    # advantage broadcast onto every assistant token, 0.0 on prompt/env tokens
    assert training_sample.advantage.tolist() == pytest.approx(
        [0.0, 0.0, -0.2, 0.0, -0.2, -0.2, 0.0, -0.2]
    )
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
    assert first.token_ids.tolist() == [1, 2, 4]
    assert first.loss_mask.tolist() == [False, False, True]
    # first segment opens at turn 0, the branch (history edit) opens at turn 1
    assert first.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=0
    )
    assert second.token_ids.tolist() == [90, 91, 5]
    assert second.loss_mask.tolist() == [False, False, True]
    assert second.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=1
    )
    assert first.advantage.tolist() == pytest.approx([0.0, 0.0, 0.1])
    assert second.advantage.tolist() == pytest.approx([0.0, 0.0, 0.1])


def test_rewrite_that_keeps_a_prefix_branches_then_continues() -> None:
    # Turn 1 keeps [1, 2] but rewrites turn 0's completion [4] as [7] (e.g. thinking stripped), so it
    # stores only [7] and opens a new training_sample; turn 2 continues it with the env reply [9].
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[1, 2, 7], completion_token_ids=[5], version=1),
            _turn(
                prompt_token_ids=[1, 2, 7, 5, 9], completion_token_ids=[6], version=2
            ),
        ],
        reward=0.5,
        advantage=0.1,
    )
    stored = [
        (rollout_turn.prompt_prefix_len, rollout_turn.prompt_delta_token_ids.tolist())
        for rollout_turn in rollout.turns
    ]
    assert stored == [(0, [1, 2]), (2, [7]), (4, [9])]
    first, second = rollout_to_training_samples(rollout)
    assert first.token_ids.tolist() == [1, 2, 4]
    assert second.token_ids.tolist() == [1, 2, 7, 5, 9, 6]
    assert second.loss_mask.tolist() == [False, False, False, True, False, True]
    assert second.rollout_id.turn_id == 1
    assert (second.min_policy_version, second.max_policy_version) == (1, 2)


def test_routed_expert_ids_follow_the_trainer_inputs_across_turns_and_branches() -> None:
    # The same turns as above. Each turn's generator rows are tagged turn * 100 + position;
    # a turn keeps the rows from its prefix's last token on, like its prompt delta.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[1, 2, 7], completion_token_ids=[5], version=1),
            _turn(
                prompt_token_ids=[1, 2, 7, 5, 9], completion_token_ids=[6], version=2
            ),
        ],
        reward=0.5,
        advantage=0.1,
    )
    for turn_idx, rollout_turn in enumerate(rollout.turns):
        # One completion token: prompt + completion - 1 rows.
        num_rows = rollout_turn.num_prompt_tokens
        rows = np.arange(num_rows).reshape(-1, 1, 1) + 100 * turn_idx
        rollout_turn.routed_expert_ids = rows[
            max(rollout_turn.prompt_prefix_len - 1, 0) :
        ]

    first, second = rollout_to_training_samples(rollout)

    # [1, 2, 4] -> rows of inputs [1, 2]. [1, 2, 7, 5, 9, 6] -> rows of [1, 2, 7, 5, 9]: the branch
    # copies the shared prefix's row from the first sample, then takes each turn's delta.
    assert first.routed_expert_ids.flatten().tolist() == [0, 1]
    assert second.routed_expert_ids.flatten().tolist() == [0, 101, 102, 203, 204]


def test_a_sample_with_generator_handles_instead_of_rows_raises() -> None:
    # Handles, uint8 [tokens, 4], would set the batcher's buffer shape and pack without an error.
    rollout_turn = _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1)
    rollout_turn.routed_expert_ids = np.zeros((2, 4), np.uint8)
    rollout = _scored_rollout([rollout_turn], reward=1.0, advantage=0.5)
    with pytest.raises(ValueError, match="generator handles"):
        rollout_to_training_samples(rollout)


def test_router_replay_drops_a_group_with_a_turn_without_routed_expert_ids() -> None:
    def rollout(rows: np.ndarray | None) -> Rollout:
        rollout_turn = _turn(
            prompt_token_ids=[1, 2], completion_token_ids=[4], version=1
        )
        rollout_turn.routed_expert_ids = rows
        return _scored_rollout([rollout_turn], reward=1.0, advantage=0.5)

    group = RolloutGroup(
        group_id=_GROUP_ID,
        rollouts=[rollout(np.zeros((2, 1, 1), np.uint8)), rollout(None)],
    )
    group.rollouts[1].reward = 0.0

    replay = TrainingSampleBuilder.Config().build(replay_routed_experts=True)
    output = replay.build_from_group(rollout_group=group)
    assert output.training_samples == []
    assert "training_sample_builder/num_groups_dropped_no_routed_expert_ids" in [
        metric.key for metric in output.metrics
    ]


@pytest.mark.parametrize(
    "prompt_prefix_len, prompt_delta_token_ids",
    [
        # past the previous turn's end: would train on a truncated prompt
        (9, [8]),
        # a full prompt stored as the delta: would open a sample per turn
        (0, [1, 2, 4, 8]),
    ],
)
def test_prefix_that_is_not_the_longest_raises(
    prompt_prefix_len: int, prompt_delta_token_ids: list[int]
) -> None:
    rollout = _scored_rollout(
        [_turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1)],
        reward=0.0,
        advantage=0.0,
    )
    rollout.turns.append(
        RolloutTurn(
            rollout_id=RolloutTurnID(group_id=_GROUP_ID, rollout_id=0, turn_id=1),
            prompt_prefix_len=prompt_prefix_len,
            prompt_delta_token_ids=np.array(prompt_delta_token_ids, dtype=np.int32),
            completion_token_ids=np.array([5], dtype=np.int32),
            completion_logprobs=np.array([-0.1], dtype=np.float32),
            min_policy_version=1,
            max_policy_version=1,
        )
    )
    with pytest.raises(ValueError, match="split_prompt"):
        rollout_to_training_samples(rollout)


@pytest.mark.parametrize(
    "prompt, previous, expected",
    [
        ([1, 2, 4, 5, 9], [1, 2, 4, 5], (4, [9])),  # continues
        ([1, 2, 7], [1, 2, 4, 5], (2, [7])),  # history rewritten after [1, 2]
        ([1, 2], [1, 2, 4, 5], (2, [])),  # strict prefix of the previous tokens
        ([1, 2], [], (0, [1, 2])),  # first turn
        ([], [1, 2], (0, [])),  # empty prompt
    ],
)
def test_split_prompt(prompt, previous, expected) -> None:
    prompt_prefix_len, prompt_delta_token_ids = split_prompt(
        prompt, previous_token_ids=previous
    )
    assert (prompt_prefix_len, prompt_delta_token_ids.tolist()) == expected


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


def test_build_from_group_drops_a_rollout_with_an_empty_later_completion() -> None:
    # e.g. turn 1 aborted before its first token: drop that rollout and count it, keep the others
    empty_later = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[], version=1),
        ],
        reward=0.0,
        advantage=1.0,
    )
    kept = _scored_rollout(
        [_turn(prompt_token_ids=[1, 2], completion_token_ids=[5], version=1)],
        reward=1.0,
        advantage=1.0,
    )
    kept.rollout_id = 1

    group = (
        TrainingSampleBuilder.Config()
        .build()
        .build_from_group(
            rollout_group=RolloutGroup(group_id=_GROUP_ID, rollouts=[empty_later, kept])
        )
    )

    assert [sample.rollout_id.rollout_id for sample in group.training_samples] == [1]
    [dropped] = [
        metric
        for metric in group.metrics
        if metric.key == "training_sample_builder/num_rollouts_dropped_empty_completion"
    ]
    assert dropped.value.value == 1.0


def test_build_from_group_skips_an_error_system_rollout_but_trains_its_siblings() -> None:
    # Its code raised after a turn: the partial turn isn't trained. Its siblings still train.
    failed = Rollout(
        group_id=_GROUP_ID,
        rollout_id=0,
        status=RolloutStatus.ERROR_SYSTEM,
        turns=[_turn(prompt_token_ids=[1, 2], completion_token_ids=[5], version=1)],
    )
    siblings = []
    for rollout_id, reward in [(1, 0.0), (2, 1.0)]:
        sibling = _scored_rollout(
            [_turn(prompt_token_ids=[1, 2], completion_token_ids=[5], version=1)],
            reward=reward,
            advantage=reward - 0.5,
        )
        sibling.rollout_id = rollout_id
        siblings.append(sibling)

    group = (
        TrainingSampleBuilder.Config()
        .build()
        .build_from_group(
            rollout_group=RolloutGroup(group_id=_GROUP_ID, rollouts=[failed, *siblings])
        )
    )

    assert [sample.rollout_id.rollout_id for sample in group.training_samples] == [1, 2]


def test_build_from_group_drops_a_group_left_with_one_rollout() -> None:
    # 1 of 2 siblings raised: the one left has advantage 0, so the group teaches nothing.
    failed = Rollout(
        group_id=_GROUP_ID, rollout_id=0, status=RolloutStatus.ERROR_SYSTEM
    )
    left = _scored_rollout(
        [_turn(prompt_token_ids=[1, 2], completion_token_ids=[5], version=1)],
        reward=1.0,
        advantage=0.0,
    )
    left.rollout_id = 1

    group = (
        TrainingSampleBuilder.Config()
        .build()
        .build_from_group(
            rollout_group=RolloutGroup(group_id=_GROUP_ID, rollouts=[failed, left])
        )
    )

    assert group.training_samples == []
    assert "training_sample_builder/num_groups_dropped_zero_std" in [
        metric.key for metric in group.metrics
    ]


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
    invalid_logprob.turns[0].completion_logprobs = np.array([np.nan], dtype=np.float32)
    no_valid_tokens = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=10, rollouts=[invalid_logprob])
    )

    filtered_groups = [failed, untrainable, zero_std, no_valid_tokens]
    assert [group.group_id for group in filtered_groups] == [7, 8, 9, 10]
    assert all(not group.training_samples for group in filtered_groups)


def test_topk_rows_follow_loss_tokens_across_turns() -> None:
    first = replace(
        _turn(prompt_token_ids=[1, 2], completion_token_ids=[4, 5], version=0),
        completion_topk_token_ids=np.array([[4, 9], [5, 8]], dtype=np.int32),
        completion_topk_logprobs=np.array([[-0.1, -2.0], [-0.2, -3.0]], np.float32),
    )
    second = replace(
        _turn(prompt_token_ids=[1, 2, 4, 5, 9], completion_token_ids=[7], version=0),
        completion_topk_token_ids=np.array([[7, 3]], dtype=np.int32),
        completion_topk_logprobs=np.array([[-0.3, -1.5]], np.float32),
    )

    [training_sample] = rollout_to_training_samples(
        _scored_rollout([first, second], reward=1.0, advantage=0.5)
    )

    # One row per loss token; the prompt and the env reply (token 9) get none.
    assert training_sample.token_ids.tolist() == [1, 2, 4, 5, 9, 7]
    assert training_sample.loss_mask.tolist() == [False, False, True, True, False, True]
    assert training_sample.topk_token_ids.tolist() == [[4, 9], [5, 8], [7, 3]]
    np.testing.assert_allclose(
        training_sample.topk_logprobs, [[-0.1, -2.0], [-0.2, -3.0], [-0.3, -1.5]]
    )


def _replayed_rollout(
    rollout_id: int,
    *,
    reward: float,
    status: RolloutStatus = RolloutStatus.COMPLETED,
    completions: tuple[list[int], ...] = ([5],),
) -> Rollout:
    """Turns of prompt [1, 2] (then the previous turn's tokens) with routed expert ids, one row per
    trainer input; a turn's rows start at its prompt prefix's last token."""
    turns = []
    token_ids = [1, 2]
    for completion in completions:
        turns.append(
            _turn(
                prompt_token_ids=token_ids, completion_token_ids=completion, version=1
            )
        )
        token_ids = token_ids + completion
    rollout = _scored_rollout(turns, reward=reward, advantage=reward - 0.5)
    for rollout_turn in rollout.turns:
        rollout_turn.rollout_id = replace(
            rollout_turn.rollout_id, rollout_id=rollout_id
        )
        end = (
            rollout_turn.prompt_prefix_len
            + len(rollout_turn.prompt_delta_token_ids)
            + len(rollout_turn.completion_token_ids)
        )
        num_rows = end - 1 - max(rollout_turn.prompt_prefix_len - 1, 0)
        rollout_turn.routed_expert_ids = np.full((num_rows, 2, 1), rollout_id, np.uint8)
    rollout.rollout_id = rollout_id
    rollout.status = status
    return rollout


@pytest.mark.parametrize(
    "rollouts",
    [
        # Zero reward spread: the whole group is left out.
        [_replayed_rollout(i, reward=1.0) for i in range(3)],
        # An ERROR_SYSTEM rollout and one whose second turn has no completion are left out;
        # the other two train.
        [
            _replayed_rollout(0, reward=1.0),
            _replayed_rollout(1, reward=0.0),
            _replayed_rollout(2, reward=0.0, status=RolloutStatus.ERROR_SYSTEM),
            _replayed_rollout(3, reward=1.0, completions=([5], [])),
        ],
    ],
    ids=["zero_std", "mixed"],
)
def test_rollouts_left_out_get_no_training_samples(rollouts) -> None:
    """Stripping the routed expert ids of every rollout `rollouts_left_out` names leaves the training
    samples the same, and doesn't make router replay drop the group."""
    builder = TrainingSampleBuilder.Config().build(replay_routed_experts=True)
    reference = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=_GROUP_ID, rollouts=rollouts)
    )
    stripped_rollouts = [
        replace(
            rollout,
            turns=[replace(rollout_turn) for rollout_turn in rollout.turns],
        )
        for rollout in rollouts
    ]
    stripped_group = RolloutGroup(group_id=_GROUP_ID, rollouts=stripped_rollouts)
    left_out = builder.rollouts_left_out(stripped_group)
    assert left_out
    for rollout in left_out:
        for rollout_turn in rollout.turns:
            rollout_turn.routed_expert_ids = None

    stripped = builder.build_from_group(rollout_group=stripped_group)

    assert not {rollout.rollout_id for rollout in left_out} & {
        sample.rollout_id.rollout_id for sample in reference.training_samples
    }
    assert len(stripped.training_samples) == len(reference.training_samples)
    for got, want in zip(
        stripped.training_samples, reference.training_samples, strict=True
    ):
        assert got.rollout_id == want.rollout_id
        assert np.array_equal(got.token_ids, want.token_ids)
        assert np.array_equal(got.routed_expert_ids, want.routed_expert_ids)


def test_rollouts_left_out_matches_build_from_group_on_random_groups() -> None:
    """Over random groups and configs, a rollout gets training samples iff `rollouts_left_out`
    doesn't name it."""
    rng = random.Random(0)
    statuses = [
        RolloutStatus.COMPLETED,
        RolloutStatus.TRUNCATED_LENGTH,
        RolloutStatus.ERROR_SYSTEM,
    ]
    completions = [([5],), ([5], [6]), ([5], []), ([],)]
    for _ in range(500):
        builder = TrainingSampleBuilder.Config(
            drop_zero_std_reward_groups=rng.random() < 0.5
        ).build(replay_routed_experts=True)
        rollouts = [
            _replayed_rollout(
                rollout_id,
                reward=rng.choice([0.0, 0.5, 1.0]),
                status=rng.choice(statuses),
                completions=rng.choice(completions),
            )
            for rollout_id in range(rng.randint(1, 4))
        ]
        group = RolloutGroup(group_id=_GROUP_ID, rollouts=rollouts)

        left_out = {rollout.rollout_id for rollout in builder.rollouts_left_out(group)}
        sampled = {
            sample.rollout_id.rollout_id
            for sample in builder.build_from_group(rollout_group=group).training_samples
        }
        assert not left_out & sampled
        assert left_out | sampled == {rollout.rollout_id for rollout in rollouts}
