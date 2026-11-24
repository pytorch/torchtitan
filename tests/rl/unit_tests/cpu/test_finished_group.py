# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.rl.rollout.finished_group`."""

from __future__ import annotations

import numpy as np
import pytest

from torchtitan.rl.controller.metrics import compute_rollout_metrics
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import Rollout, RolloutGroup, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.finished_group import finish_rollout_group
from torchtitan.rl.rollout.recorder import RolloutSampleEncoder, RolloutSampleRecorder
from torchtitan.rl.rollout.types import RolloutTurnID


def _group(
    group_id: int = 3, *, num_completion_tokens: tuple[int, ...] = (2,) * 4
) -> RolloutGroup:
    """A scored group with messages and a reward breakdown, which the metrics and lines read."""
    rollouts = []
    for rollout_id, num_tokens in enumerate(num_completion_tokens):
        turn = RolloutTurn(
            rollout_id=RolloutTurnID(
                group_id=group_id, rollout_id=rollout_id, turn_id=0
            ),
            prompt_prefix_len=0,
            prompt_delta_token_ids=np.arange(4, dtype=np.int32),
            completion_token_ids=np.arange(num_tokens, dtype=np.int32),
            completion_logprobs=np.full(num_tokens, -0.5, dtype=np.float32),
            min_policy_version=1,
            max_policy_version=1,
            prompt_messages=[{"role": "user", "content": "sort: b a"}],
            completion_message={"role": "assistant", "content": "a b"},
            metrics=[m.Metric("generator/queue_time_ms", m.Mean(2.0))],
        )
        rollouts.append(
            Rollout(
                group_id=group_id,
                rollout_id=rollout_id,
                status=RolloutStatus.COMPLETED,
                turns=[turn],
                reward=float(rollout_id),
                reward_breakdown={"correct": float(rollout_id)},
                advantage=float(rollout_id) - 1.5,
            )
        )
    return RolloutGroup(
        group_id=group_id,
        rollouts=rollouts,
        metrics=[m.Metric("worker/x", m.Mean(1.0))],
    )


@pytest.mark.parametrize("is_validation", [False, True])
def test_finish_rollout_group_adds_the_metrics_and_lines_of_the_full_group(
    is_validation: bool,
) -> None:
    group = _group()
    prefix = "validation" if is_validation else "rollout"
    expected_metrics = m.MetricsProcessor._aggregate_metrics(
        group.metrics + compute_rollout_metrics(prefix=prefix, rollouts=group.rollouts)
    )
    recorder_config = RolloutSampleRecorder.Config(log_logprobs=True)
    expected_lines = RolloutSampleEncoder(recorder_config).encode(
        is_validation=is_validation, rollout_groups=[group]
    )

    finished = finish_rollout_group(
        group,
        is_validation=is_validation,
        encoder=RolloutSampleEncoder(recorder_config),
    )

    # The worker returns the scored group itself, with its metrics and lines added.
    assert finished is group
    assert m.MetricsProcessor._aggregate_metrics(group.metrics) == expected_metrics
    assert f"{prefix}_reward/component/correct/mean" in expected_metrics
    assert group.recorder_lines == expected_lines
    assert '"sort: b a"' in group.recorder_lines
    assert f'"is_validation": {str(is_validation).lower()}' in group.recorder_lines


def test_validation_metrics_merged_over_groups_equal_the_whole_batch() -> None:
    """Validation used to compute its metrics over every rollout at once; the workers now
    compute them per group, and logging merges them."""
    groups = [
        _group(-1, num_completion_tokens=(1, 3, 5, 7)),
        _group(-2, num_completion_tokens=(2, 2, 9, 4)),
    ]
    whole_batch = m.MetricsProcessor._aggregate_metrics(
        compute_rollout_metrics(
            prefix="validation",
            rollouts=[rollout for group in groups for rollout in group.rollouts],
        )
    )
    encoder = RolloutSampleEncoder(RolloutSampleRecorder.Config())
    for group in groups:
        group.metrics.clear()
        finish_rollout_group(group, is_validation=True, encoder=encoder)

    per_group = m.MetricsProcessor._aggregate_metrics(
        [metric for group in groups for metric in group.metrics]
    )

    assert per_group.keys() == whole_batch.keys()
    for key, value in whole_batch.items():
        assert per_group[key] == pytest.approx(value, rel=1e-12), key
