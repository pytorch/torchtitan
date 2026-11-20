# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.rl.rollout.finished_group`."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, fields, replace

import numpy as np
import pytest
from renderers import PlaceholderRange

from torchtitan.rl.controller.metrics import (
    compute_hop_metrics,
    compute_rollout_metrics,
)
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import Rollout, RolloutGroup, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.finished_group import (
    finish_rollout_group,
    slim_rollout_group,
)
from torchtitan.rl.rollout.packed_group import unpack_rollout_group
from torchtitan.rl.rollout.recorder import RolloutSampleEncoder, RolloutSampleRecorder
from torchtitan.rl.rollout.types import PromptImage, RolloutTurnID
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder

from tests.rl.unit_tests.cpu.test_rollout_utils import rollout_to_training_samples

_IMAGE_BYTES = 1 << 20


@dataclass(kw_only=True, slots=True)
class _TurnWithTopkLogprobs(RolloutTurn):
    """A turn with one more per-token array, as score centering adds top-k logprobs."""

    completion_topk_logprobs: np.ndarray


def _group(
    *, group_size: int = 8, with_image: bool = True, image_per_rollout: bool = False
) -> RolloutGroup:
    """A scored group whose rollouts each read the same image file (or with
    `image_per_rollout`, their own file) into their own bytes."""
    rollouts = []
    for rollout_id in range(group_size):
        images = (
            (
                PromptImage(
                    hash_value=f"h{rollout_id}" if image_per_rollout else "h",
                    source_path="a.png",
                    grid_thw=(1, 2, 2),
                    # A copy per rollout, as each env reads the file.
                    data=bytes(bytearray(_IMAGE_BYTES)),
                ),
            )
            if with_image
            else ()
        )
        turn = _TurnWithTopkLogprobs(
            rollout_id=RolloutTurnID(group_id=3, rollout_id=rollout_id, turn_id=0),
            prompt_prefix_len=0,
            prompt_delta_token_ids=np.arange(4, dtype=np.int32),
            prompt_delta_images=images,
            prompt_delta_image_placeholders=(
                (PlaceholderRange(offset=1, length=1),) if with_image else ()
            ),
            completion_token_ids=np.array([5, 6], dtype=np.int32),
            completion_logprobs=np.array([-0.5, -1.5], dtype=np.float32),
            completion_topk_logprobs=np.zeros((2, 4), dtype=np.float32),
            min_policy_version=1,
            max_policy_version=1,
            prompt_messages=[{"role": "user", "content": "sort: b a"}],
            completion_message={"role": "assistant", "content": "a b"},
            env_messages=[{"role": "tool", "content": "ok"}],
            env_rewards={"format": 1.0},
            metrics=[m.Metric("generator/queue_time_ms", m.Mean(2.0))],
        )
        rollouts.append(
            Rollout(
                group_id=3,
                rollout_id=rollout_id,
                status=RolloutStatus.COMPLETED,
                turns=[turn],
                reward=float(rollout_id),
                reward_breakdown={"correct": float(rollout_id)},
                advantage=float(rollout_id) - 3.5,
                logs={"finish": "stop"},
            )
        )
    return RolloutGroup(
        group_id=3,
        rollouts=rollouts,
        metrics=[m.Metric("worker/x", m.Mean(1.0))],
    )


def test_slim_rollout_group_drops_only_the_logging_fields() -> None:
    group = _group(group_size=2)
    slim = slim_rollout_group(group)

    for rollout, slim_rollout in zip(group.rollouts, slim.rollouts, strict=True):
        assert (slim_rollout.reward_breakdown, slim_rollout.logs) == ({}, {})
        assert (slim_rollout.reward, slim_rollout.advantage, slim_rollout.status) == (
            rollout.reward,
            rollout.advantage,
            rollout.status,
        )
        (turn,), (slim_turn,) = rollout.turns, slim_rollout.turns
        # Keeps the turn's class and every array, including one RolloutTurn does not declare.
        assert isinstance(slim_turn, _TurnWithTopkLogprobs)
        assert isinstance(turn, _TurnWithTopkLogprobs)
        assert slim_turn.completion_topk_logprobs is turn.completion_topk_logprobs
        assert slim_turn.completion_logprobs is turn.completion_logprobs
        assert slim_turn.prompt_delta_images == turn.prompt_delta_images
        assert slim_turn.rollout_id == turn.rollout_id
        assert (slim_turn.prompt_messages, slim_turn.completion_message) == ([], None)
        assert (slim_turn.env_messages, slim_turn.env_rewards, slim_turn.metrics) == (
            [],
            {},
            [],
        )
        # The input group is left as it was.
        assert turn.prompt_messages and rollout.reward_breakdown
    assert slim.metrics == group.metrics


@pytest.mark.parametrize(
    "prompt_prefix_len, prompt_delta_token_ids, match",
    [
        # The full prompt as the delta: `split_prompt` would store (6, [7]).
        (0, [0, 1, 2, 3, 5, 6, 7], "could reuse more"),
        # Past the previous turn's end: its prompt + completion has 6 tokens.
        (9, [7], "previous turn has only 6"),
    ],
)
def test_finish_rollout_group_checks_every_rollouts_prompt_prefixes(
    prompt_prefix_len: int, prompt_delta_token_ids: list[int], match: str
) -> None:
    """The prompt prefix checks read token values, so the worker runs them before packing, on
    every rollout: the batcher gets the summary and decides from lengths only."""
    group = _group(group_size=2, with_image=False)
    (turn,) = group.rollouts[1].turns
    # Previous prompt + completion: [0, 1, 2, 3, 5, 6].
    group.rollouts[1].turns.append(
        replace(
            turn,
            prompt_prefix_len=prompt_prefix_len,
            prompt_delta_token_ids=np.array(prompt_delta_token_ids, dtype=np.int32),
        )
    )

    with pytest.raises(ValueError, match=match):
        finish_rollout_group(
            group,
            for_training=True,
            encoder=RolloutSampleEncoder(RolloutSampleRecorder.Config()),
        )


def test_finish_rollout_group_for_training_logs_records_then_slims() -> None:
    group = _group()
    expected_metrics = m.MetricsProcessor._aggregate_metrics(
        group.metrics
        + compute_rollout_metrics(prefix="rollout", rollouts=group.rollouts)
    )
    recorder_config = RolloutSampleRecorder.Config(log_logprobs=True)
    expected_lines = RolloutSampleEncoder(recorder_config).encode(
        is_validation=False, rollout_groups=[group]
    )

    finished = finish_rollout_group(
        group, for_training=True, encoder=RolloutSampleEncoder(recorder_config)
    )

    # Metrics and recorder lines come from the full group (messages, reward breakdown).
    assert (
        m.MetricsProcessor._aggregate_metrics(finished.summary.group.metrics)
        == expected_metrics
    )
    assert "rollout_reward/component/correct/mean" in expected_metrics
    assert finished.recorder_lines == expected_lines
    assert '"sort: b a"' in finished.recorder_lines
    # The group is slim and packed: the summary has no bytes, the regions every array and one
    # copy of the image file.
    for rollout in finished.summary.group.rollouts:
        assert rollout.reward_breakdown == {}
        assert rollout.turns[0].prompt_messages == []
        assert len(rollout.turns[0].completion_token_ids) == 0
    unpacked_turn = unpack_rollout_group(finished.packed).rollouts[0].turns[0]
    np.testing.assert_array_equal(unpacked_turn.completion_token_ids, [5, 6])
    assert unpacked_turn.completion_topk_logprobs.shape == (2, 4)
    assert unpacked_turn.prompt_messages == []
    assert _IMAGE_BYTES < finished.packed.num_bytes < 2 * _IMAGE_BYTES
    assert len(pickle.dumps(finished)) < 2 * _IMAGE_BYTES


def test_a_leaders_rollouts_carry_no_group_metrics_and_build_the_same_samples() -> None:
    """What the controller keeps of a training group (`FinishedRolloutGroup.packed`) and sends a
    trainer DP replica leader (`PackedRolloutGroup.select`) has no group metrics, messages or
    logs, and the leader builds the same training samples from it as from the full group."""
    group = _group(group_size=3, with_image=False)
    builder = TrainingSampleBuilder(TrainingSampleBuilder.Config())
    expected = [
        sample
        for rollout in group.rollouts
        for sample in rollout_to_training_samples(rollout)
    ]
    finished = finish_rollout_group(
        group,
        for_training=True,
        encoder=RolloutSampleEncoder(RolloutSampleRecorder.Config()),
    )
    # As the controller adds them after `run_group`.
    finished.summary.group.metrics.extend(
        compute_hop_metrics("run_group", wall_s=[1.0], handler_s=[0.9])
    )
    specs = builder.decide(
        rollout_group=finished.summary.group,
        turn_lengths=finished.summary.turn_lengths,
        num_mtp_layers=0,
    ).training_sample_specs

    leader_group = unpack_rollout_group(finished.packed.select({0, 1, 2}))

    assert finished.summary.group.metrics
    assert leader_group.metrics == []
    for rollout in leader_group.rollouts:
        assert (rollout.logs, rollout.reward_breakdown) == ({}, {})
        for turn in rollout.turns:
            assert (turn.prompt_messages, turn.completion_message) == ([], None)
            assert (turn.env_messages, turn.metrics) == ([], [])
    samples = builder.materialize(
        training_sample_specs=specs, rollout_group=leader_group
    )
    assert len(samples) == len(expected) == 3
    for sample, expected_sample in zip(samples, expected, strict=True):
        for sample_field in fields(sample):
            value = getattr(sample, sample_field.name)
            expected_value = getattr(expected_sample, sample_field.name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, expected_value)
            else:
                assert value == expected_value
