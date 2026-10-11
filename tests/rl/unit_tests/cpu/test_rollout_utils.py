# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `TrainingSampleBuilder.rollout_to_training_samples` and `split_prompt`."""

from __future__ import annotations

import numpy as np
import pytest

from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
from torchtitan.rl.rollout import Rollout, RolloutGroup, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.types import RolloutTurnID, split_prompt

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
        # A root with the full prompt for now; `_scored_rollout` stores it as a delta on its parent
        parent_turn_id=None,
        prompt_prefix_len=0,
        prompt_delta_token_ids=np.array(prompt_token_ids, dtype=np.int32),
        completion_token_ids=np.array(completion_token_ids, dtype=np.int32),
        completion_logprobs=np.full(len(completion_token_ids), -0.1, dtype=np.float32),
        min_policy_version=version,
        max_policy_version=version if max_version is None else max_version,
        completion_message={"role": "assistant", "content": content},
    )


def _scored_rollout(
    turns: list[RolloutTurn],
    *,
    reward: float,
    advantage: float,
    parent_turn_ids: list[int | None] | None = None,
) -> Rollout:
    """Store each turn's full prompt as a delta on its parent: the previous turn unless given."""
    if parent_turn_ids is None:
        parent_turn_ids = [None, *range(len(turns) - 1)]
    # Each turn's prompt + completion, to split later prompts against
    token_ids: list[list[int]] = []
    for rollout_turn, parent_turn_id in zip(turns, parent_turn_ids, strict=True):
        prompt_token_ids = rollout_turn.prompt_delta_token_ids.tolist()
        prefix_len, delta = split_prompt(
            prompt_token_ids,
            token_ids[parent_turn_id] if parent_turn_id is not None else [],
        )
        rollout_turn.parent_turn_id = parent_turn_id
        rollout_turn.prompt_prefix_len = prefix_len
        rollout_turn.prompt_delta_token_ids = delta
        token_ids.append(prompt_token_ids + rollout_turn.completion_token_ids.tolist())
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
            parent_turn_id=0,
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


@pytest.mark.parametrize("parent_turn_id", [1, 2, -1])
def test_parent_that_is_not_an_earlier_turn_raises(parent_turn_id: int) -> None:
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[5], version=1),
        ],
        reward=0.0,
        advantage=0.0,
    )
    rollout.turns[1].parent_turn_id = parent_turn_id
    with pytest.raises(ValueError, match="not an earlier turn"):
        rollout_to_training_samples(rollout)


@pytest.mark.parametrize(
    "prompt, parent, expected",
    [
        ([1, 2, 4, 5, 9], [1, 2, 4, 5], (4, [9])),  # continues
        ([1, 2, 7], [1, 2, 4, 5], (2, [7])),  # history rewritten after [1, 2]
        ([1, 2], [1, 2, 4, 5], (2, [])),  # strict prefix of the parent's tokens
        ([1, 2], [], (0, [1, 2])),  # no parent
        ([], [1, 2], (0, [])),  # empty prompt
    ],
)
def test_split_prompt(prompt, parent, expected) -> None:
    prompt_prefix_len, prompt_delta_token_ids = split_prompt(
        prompt, parent_token_ids=parent
    )
    assert (prompt_prefix_len, prompt_delta_token_ids.tolist()) == expected


def test_fork_from_an_older_turn_opens_a_sample_with_its_prefix_as_context() -> None:
    # Turn 2 retries turn 0's tool call with another reply [9]. Turn 1 already continued turn 0's
    # sample, so turn 2 opens a new one; [4] trains once, in sample 0.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[5], version=2),
            _turn(prompt_token_ids=[1, 2, 4, 9], completion_token_ids=[6], version=3),
        ],
        reward=0.5,
        advantage=0.1,
        parent_turn_ids=[None, 0, 0],
    )
    stored = [
        (
            turn.parent_turn_id,
            turn.prompt_prefix_len,
            turn.prompt_delta_token_ids.tolist(),
        )
        for turn in rollout.turns
    ]
    assert stored == [(None, 0, [1, 2]), (0, 3, [8]), (0, 3, [9])]

    first, second = rollout_to_training_samples(rollout)
    assert first.token_ids.tolist() == [1, 2, 4, 8, 5]
    assert first.loss_mask.tolist() == [False, False, True, False, True]
    assert second.token_ids.tolist() == [1, 2, 4, 9, 6]
    assert second.loss_mask.tolist() == [False, False, False, False, True]
    assert second.logprobs.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, -0.1])
    assert second.advantage.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.1])
    assert second.rollout_id.turn_id == 2
    assert (second.min_policy_version, second.max_policy_version) == (3, 3)


def test_resample_and_mid_completion_branch() -> None:
    # Turn 1 resamples turn 0 (same prompt, prefix stops before turn 0's completion); turn 2 branches
    # inside turn 0's completion, after [4]. Each stores only its new tokens.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4, 5, 6], version=1),
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[7], version=1),
            _turn(prompt_token_ids=[1, 2, 4], completion_token_ids=[8, 9], version=1),
        ],
        reward=0.5,
        advantage=1.0,
        parent_turn_ids=[None, 0, 0],
    )
    stored = [
        (turn.prompt_prefix_len, turn.prompt_delta_token_ids.tolist())
        for turn in rollout.turns
    ]
    assert stored == [(0, [1, 2]), (2, []), (3, [])]
    samples = rollout_to_training_samples(rollout)
    assert [sample.token_ids.tolist() for sample in samples] == [
        [1, 2, 4, 5, 6],
        [1, 2, 7],
        [1, 2, 4, 8, 9],
    ]
    # [4] trains in sample 0 only; in sample 2 it is context
    assert samples[2].loss_mask.tolist() == [False, False, False, True, True]


def test_interleaved_subagent_turns_continue_their_own_samples() -> None:
    # The main agent (turns 0, 1, 3) and a subagent (turns 2, 4) alternate; each turn names its own
    # agent's previous turn, so each agent packs into one sample and nothing is re-forwarded.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[3], version=1),
            _turn(prompt_token_ids=[1, 2, 3, 4], completion_token_ids=[5], version=1),
            _turn(prompt_token_ids=[50, 51], completion_token_ids=[52], version=2),
            _turn(
                prompt_token_ids=[1, 2, 3, 4, 5, 6], completion_token_ids=[7], version=3
            ),
            _turn(
                prompt_token_ids=[50, 51, 52, 53], completion_token_ids=[54], version=3
            ),
        ],
        reward=0.5,
        advantage=1.0,
        parent_turn_ids=[None, 0, None, 1, 2],
    )
    assert [turn.prompt_delta_token_ids.tolist() for turn in rollout.turns] == [
        [1, 2],
        [4],
        [50, 51],
        [6],
        [53],
    ]
    main_agent, subagent = rollout_to_training_samples(rollout)
    assert main_agent.token_ids.tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert np.flatnonzero(main_agent.loss_mask).tolist() == [
        2,
        4,
        6,
    ]  # completions [3], [5], [7]
    assert (main_agent.min_policy_version, main_agent.max_policy_version) == (1, 3)
    assert subagent.token_ids.tolist() == [50, 51, 52, 53, 54]
    assert subagent.rollout_id.turn_id == 2
    assert (subagent.min_policy_version, subagent.max_policy_version) == (2, 3)


def test_fork_from_the_middle_of_a_sample() -> None:
    # Turn 3 forks from turn 1, which sits inside sample 0: its context is turn 1's prompt + completion.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[3], version=1),
            _turn(prompt_token_ids=[1, 2, 3, 4], completion_token_ids=[5], version=1),
            _turn(
                prompt_token_ids=[1, 2, 3, 4, 5, 6], completion_token_ids=[7], version=1
            ),
            _turn(
                prompt_token_ids=[1, 2, 3, 4, 5, 8], completion_token_ids=[9], version=1
            ),
        ],
        reward=0.5,
        advantage=1.0,
        parent_turn_ids=[None, 0, 1, 1],
    )
    first, second = rollout_to_training_samples(rollout)
    assert first.token_ids.tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert second.token_ids.tolist() == [1, 2, 3, 4, 5, 8, 9]
    assert np.flatnonzero(second.loss_mask).tolist() == [6]


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
