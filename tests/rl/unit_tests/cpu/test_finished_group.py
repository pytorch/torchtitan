# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.rl.rollout.finished_group`."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
from renderers import PlaceholderRange

from torchtitan.rl.controller.metrics import compute_rollout_metrics
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import Rollout, RolloutGroup, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.finished_group import (
    finish_rollout_group,
    slim_rollout_group,
)
from torchtitan.rl.rollout.recorder import RolloutSampleEncoder, RolloutSampleRecorder
from torchtitan.rl.rollout.types import PromptImage, RolloutTurnID

_IMAGE_BYTES = 1 << 20


@dataclass(kw_only=True, slots=True)
class _TurnWithTopkLogprobs(RolloutTurn):
    """A turn with one more per-token array, as score centering adds top-k logprobs."""

    completion_topk_logprobs: np.ndarray


def _group(*, group_size: int = 8, with_image: bool = True) -> RolloutGroup:
    """A scored group whose rollouts each read the same image file into their own bytes."""
    rollouts = []
    for rollout_id in range(group_size):
        images = (
            (
                PromptImage(
                    hash_value="h",
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
        assert slim_turn.prompt_delta_images[0].data is (
            slim.rollouts[0].turns[0].prompt_delta_images[0].data
        )
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


def test_a_slim_group_pickles_each_image_file_once() -> None:
    group = _group()
    assert len(pickle.dumps(group)) > 8 * _IMAGE_BYTES

    slim = slim_rollout_group(group)

    images = [
        image
        for rollout in slim.rollouts
        for turn in rollout.turns
        for image in turn.prompt_delta_images
    ]
    assert len(images) == 8
    assert len({id(image.data) for image in images}) == 1
    assert len(pickle.dumps(slim)) < 2 * _IMAGE_BYTES


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
        m.MetricsProcessor._aggregate_metrics(finished.group.metrics)
        == expected_metrics
    )
    assert "rollout_reward/component/correct/mean" in expected_metrics
    assert finished.recorder_lines == expected_lines
    assert '"sort: b a"' in finished.recorder_lines
    for rollout in finished.group.rollouts:
        assert rollout.reward_breakdown == {}
        assert rollout.turns[0].prompt_messages == []
    assert len(pickle.dumps(finished)) < 2 * _IMAGE_BYTES
