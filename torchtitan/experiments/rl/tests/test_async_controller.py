# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for async-controller pieces: batcher group-counting, the active-slot buffer backpressure,
the consume-time staleness invariant, the metrics timer drain, and RolloutTurnID."""

import asyncio

import pytest

from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.components.work_buffer import (
    RolloutGroupWork,
    RolloutGroupWorkBuffer,
)
from torchtitan.experiments.rl.controller_metrics import (
    compute_perf_ratio_metrics,
    compute_policy_age_metrics,
    MetricsTimer,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.experiments.rl.types import (
    RolloutTurnID,
    TrainingSample,
    TrainingSampleGroup,
)


def _training_sample(*, group_id: int, rollout_id: int) -> TrainingSample:
    return TrainingSample(
        min_policy_version=0,
        max_policy_version=0,
        rollout_id=RolloutTurnID(group_id=group_id, rollout_id=rollout_id, turn_id=0),
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


def _build_batcher(*, num_groups_per_train_step: int) -> Batcher:
    return Batcher.Config().build(
        num_groups_per_train_step=num_groups_per_train_step,
        dp_degree=1,
        pad_id=0,
    )


def test_batcher_counts_trainable_groups_not_rollouts() -> None:
    # Target is 2 GROUPS. A single group with many rollouts is not a full batch; two groups are,
    # regardless of how many rollouts each contributes.
    batcher = _build_batcher(num_groups_per_train_step=2)
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
    batcher = _build_batcher(num_groups_per_train_step=1)
    metric_only = TrainingSampleGroup(group_id=0, training_samples=[], metrics=[])
    assert batcher.add_training_samples(training_sample_group=metric_only) is None
    batch = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=2)
    )
    assert batch is not None
    assert batch.num_global_valid_tokens > 0


def test_packing_flushes_microbatch_on_overflow() -> None:
    # rows_per_microbatch = local_batch_size * dp_degree = 4; seq_len=2 so each 2-token sample
    # fills one row. 5 samples: 4 fill microbatch 0's rows, the 5th fits none -> flush -> microbatch 1.
    batcher = Batcher.Config(batch=BatchConfig(local_batch_size=2, seq_len=2)).build(
        num_groups_per_train_step=1,
        dp_degree=2,
        pad_id=0,
    )
    batch = batcher.add_training_samples(
        training_sample_group=_trainable_group(0, num_samples=5)
    )
    assert batch is not None
    assert len(batch.microbatches) == 2  # flushed once on overflow
    assert all(len(ranks) == 2 for ranks in batch.microbatches)  # dp_degree ranks each
    # Every sample is trained: 5 samples x 2 trained tokens each (loss_mask=[F,T,T] -> [T,T]).
    assert batch.num_global_valid_tokens == 5 * 2


def _variable_length_group(group_id: int, *, lengths: list[int]) -> TrainingSampleGroup:
    # token_ids of length n -> n-1 packed tokens; loss_mask trains all but the first.
    def sample(rollout_id: int, n: int) -> TrainingSample:
        return TrainingSample(
            min_policy_version=0,
            max_policy_version=0,
            rollout_id=RolloutTurnID(
                group_id=group_id, rollout_id=rollout_id, turn_id=0
            ),
            token_ids=list(range(n)),
            loss_mask=[False] + [True] * (n - 1),
            logprobs=[0.0] * n,
            advantage=[0.0] + [1.0] * (n - 1),
        )

    return TrainingSampleGroup(
        group_id=group_id,
        training_samples=[sample(i, n) for i, n in enumerate(lengths)],
        metrics=[],
    )


def _metric_value(batch, key: str) -> float:
    return next(metric.value.value for metric in batch.metrics if metric.key == key)


def test_packing_balances_dp_rank_square_cost() -> None:
    # A few long sequences among many short ones. The longest-processing-time deal evens the
    # per-rank square (sum seq_len**2) attention cost, so no DP rank straggler gates the step.
    lengths = [63, 63, 33, 33, 33, 17, 17, 17, 17, 9, 9, 9, 9, 9, 9, 5, 5, 5]
    batcher = Batcher.Config(batch=BatchConfig(local_batch_size=2, seq_len=64)).build(
        num_groups_per_train_step=1, dp_degree=2, pad_id=0
    )
    batch = batcher.add_training_samples(
        training_sample_group=_variable_length_group(0, lengths=lengths)
    )
    assert batch is not None
    # 1.0 is perfect DP-rank balance; count-based round-robin on this data is ~1.1.
    assert _metric_value(batch, "train_batch/cost_imbalance") <= 1.05


def test_packing_minimizes_microbatches_on_bad_arrival_order() -> None:
    # rows_per_microbatch=1 (local_batch_size=1, dp_degree=1), so each microbatch is one row.
    # Longest-first packing pairs the two 30s into one row (58<=64); the two 40s can't share a row
    # (80>64) -> 3 microbatches, not 4 (which arrival-order packing would produce).
    lengths = [40, 30, 40, 30]
    batcher = Batcher.Config(batch=BatchConfig(local_batch_size=1, seq_len=64)).build(
        num_groups_per_train_step=1, dp_degree=1, pad_id=0
    )
    batch = batcher.add_training_samples(
        training_sample_group=_variable_length_group(0, lengths=lengths)
    )
    assert batch is not None
    assert _metric_value(batch, "train_batch/num_microbatches") == 3


def test_compute_perf_ratio_metrics_reads_flushed_means() -> None:
    time_metrics = [
        m.Metric("timing/step/total", m.Mean.from_list([2.0])),
        m.Metric("timing/step/forward_backward", m.Mean.from_list([0.5])),
        m.Metric("timing/step/optim", m.Mean.from_list([0.5])),
    ]
    ratios = {
        metric.key: metric.value.value
        for metric in compute_perf_ratio_metrics(
            num_global_valid_tokens=100, time_metrics=time_metrics
        )
    }
    assert ratios["perf/trainer/tokens_per_second_full_step"] == 50.0
    assert ratios["perf/trainer/step_time_ratio/fwd_bwd"] == 0.5
    assert ratios["perf/trainer/tokens_per_second_fwd_bwd"] == 100.0


def test_compute_perf_ratio_metrics_skips_missing_spans() -> None:
    # Only `total` recorded -> emit the full-step throughput, skip every ratio whose span is absent.
    time_metrics = [m.Metric("timing/step/total", m.Mean.from_list([2.0]))]
    keys = {
        metric.key
        for metric in compute_perf_ratio_metrics(
            num_global_valid_tokens=100, time_metrics=time_metrics
        )
    }
    assert keys == {"perf/trainer/tokens_per_second_full_step"}


def test_compute_perf_ratio_metrics_returns_empty_without_total() -> None:
    assert (
        compute_perf_ratio_metrics(num_global_valid_tokens=100, time_metrics=[]) == []
    )


def test_metrics_timer_flush_drains() -> None:
    timer = MetricsTimer()
    with timer.record("timing/x"):
        pass
    assert timer.flush()  # non-empty on first read
    assert timer.flush() == []  # drained on the second read


def test_rollout_id_to_string_is_callable_and_uses_int_group_id() -> None:
    rollout_id = RolloutTurnID(group_id=5, rollout_id=2, turn_id=0)
    assert rollout_id.to_string() == "group=5/rollout=2/turn=0"
    assert rollout_id.to_string(include_turn=False) == "group=5/rollout=2"


def test_take_finalized_does_not_release_active_slot() -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config().build(max_active_rollout_groups=1)
        if not await buffer.wait_for_slot():
            raise RuntimeError("buffer closed unexpectedly")
        await buffer.add_work(RolloutGroupWork(group_id=0, sample=object()))
        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[]))
        await buffer.take_finalized()

        waiter = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not waiter.done()

        await buffer.release_active_groups(1, reason="trained")
        assert await waiter

    asyncio.run(run())


def test_untrainable_group_releases_before_training() -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config().build(max_active_rollout_groups=1)
        batcher = Batcher.Config().build(
            num_groups_per_train_step=1,
            dp_degree=1,
            pad_id=0,
        )

        if not await buffer.wait_for_slot():
            raise RuntimeError("buffer closed unexpectedly")
        await buffer.add_work(RolloutGroupWork(group_id=0, sample=object()))

        training_sample_group = TrainingSampleGroup(
            group_id=0, training_samples=[], metrics=[]
        )
        await buffer.release_active_groups(1, reason="untrainable_group")
        assert (
            batcher.add_training_samples(training_sample_group=training_sample_group)
            is None
        )

    asyncio.run(run())


def test_compute_policy_age_metrics_raises_on_consume_time_staleness() -> None:
    with pytest.raises(RuntimeError, match="admitted stale training data"):
        compute_policy_age_metrics(
            trainer_policy_version=4,
            min_policy_versions=[0],
            max_offpolicy_steps=3,
        )
