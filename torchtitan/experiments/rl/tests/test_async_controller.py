# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for async-controller pieces: batcher group-counting, the metrics timer drain, and RolloutID."""

from torchtitan.experiments.rl.batcher import Batcher
from torchtitan.experiments.rl.components.metrics_utils import MetricsTimer
from torchtitan.experiments.rl.types import (
    RolloutID,
    TrainingSample,
    TrainingSampleGroup,
)


def _training_sample(*, group_id: int, rollout_id: int) -> TrainingSample:
    return TrainingSample(
        min_policy_version=0,
        max_policy_version=0,
        rollout_id=RolloutID(group_id=group_id, rollout_id=rollout_id, turn_id=0),
        token_ids=[1, 2, 3],
        loss_mask=[False, True, True],
        logprobs=[0.0, 0.1, 0.2],
        advantage=[0.0, 1.0, 1.0],
    )


def _trainable_group(group_id: int, *, num_samples: int) -> TrainingSampleGroup:
    return TrainingSampleGroup(
        group_id=group_id,
        training_samples=[
            _training_sample(group_id=group_id, rollout_id=i)
            for i in range(num_samples)
        ],
        metrics=[],
    )


def _build_batcher(*, target_groups_per_batch: int) -> Batcher:
    return Batcher.Config().build(
        target_groups_per_batch=target_groups_per_batch,
        max_offpolicy_steps=3,
        trainer_policy_version_getter=lambda: 0,
        dp_degree=1,
        pad_id=0,
    )


def test_batcher_counts_trainable_groups_not_rollouts() -> None:
    # Target is 2 GROUPS. A single group with many rollouts is not a full batch; two groups are,
    # regardless of how many rollouts each contributes.
    batcher = _build_batcher(target_groups_per_batch=2)
    assert (
        batcher.add_training_samples(
            training_sample_group=_trainable_group(0, num_samples=8)
        )
        is None
    )
    batch = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=1)
    )
    assert batch is not None


def test_batcher_carries_metric_only_groups_until_trainable_batch() -> None:
    # Metric-only (empty) groups do not count toward the target and cannot form a zero-token batch;
    # they ride along until a trainable group completes the batch.
    batcher = _build_batcher(target_groups_per_batch=1)
    metric_only = TrainingSampleGroup(group_id=0, training_samples=[], metrics=[])
    assert batcher.add_training_samples(training_sample_group=metric_only) is None
    batch = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=2)
    )
    assert batch is not None
    assert batch.num_global_valid_tokens > 0


def test_metrics_timer_metrics_drains() -> None:
    timer = MetricsTimer()
    with timer.record("timing/x"):
        pass
    assert timer.metrics()  # non-empty on first read
    assert timer.metrics() == []  # drained on the second read


def test_rollout_id_to_string_is_callable_and_uses_int_group_id() -> None:
    rollout_id = RolloutID(group_id=5, rollout_id=2, turn_id=0)
    assert rollout_id.to_string() == "group=5/rollout=2/turn=0"
    assert rollout_id.to_string(include_turn=False) == "group=5/rollout=2"


def test_drop_rollout_group_if_any_stale_drops_whole_group() -> None:
    # A group with one stale (v0) + one fresh (v10) sample at trainer version 10, max_offpolicy_steps=1.
    def _sample(min_policy_version: int) -> TrainingSample:
        return TrainingSample(
            min_policy_version=min_policy_version,
            max_policy_version=min_policy_version,
            rollout_id=RolloutID(group_id=0, rollout_id=0, turn_id=0),
            token_ids=[1, 2, 3],
            loss_mask=[False, True, True],
            logprobs=[0.0, 0.1, 0.2],
            advantage=[0.0, 1.0, 1.0],
        )

    def _mixed_group() -> TrainingSampleGroup:
        return TrainingSampleGroup(
            group_id=0, training_samples=[_sample(0), _sample(10)], metrics=[]
        )

    def _batcher(*, drop_whole_group: bool) -> Batcher:
        return Batcher.Config(drop_rollout_group_if_any_stale=drop_whole_group).build(
            target_groups_per_batch=1,
            max_offpolicy_steps=1,
            trainer_policy_version_getter=lambda: 10,
            dp_degree=1,
            pad_id=0,
        )

    # flag on: the one stale sample drops the WHOLE group -> no trainable group -> no batch.
    assert (
        _batcher(drop_whole_group=True).add_training_samples(
            training_sample_group=_mixed_group()
        )
        is None
    )
    # flag off: only the stale sample drops -> the fresh one survives -> a batch is packed.
    assert (
        _batcher(drop_whole_group=False).add_training_samples(
            training_sample_group=_mixed_group()
        )
        is not None
    )
