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
    derive_window_size,
    RolloutGroupWork,
    RolloutGroupWorkBuffer,
)
from torchtitan.experiments.rl.controller_metrics import (
    compute_policy_age_metrics,
    compute_perf_ratio_metrics,
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


def _build_batcher(*, num_prompts_per_train_step: int) -> Batcher:
    return Batcher.Config().build(
        num_prompts_per_train_step=num_prompts_per_train_step,
        dp_degree=1,
        pad_id=0,
    )


def test_batcher_counts_trainable_groups_not_rollouts() -> None:
    # Target is 2 GROUPS. A single group with many rollouts is not a full batch; two groups are,
    # regardless of how many rollouts each contributes.
    batcher = _build_batcher(num_prompts_per_train_step=2)
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
    batcher = _build_batcher(num_prompts_per_train_step=1)
    metric_only = TrainingSampleGroup(group_id=0, training_samples=[], metrics=[])
    assert batcher.add_training_samples(training_sample_group=metric_only) is None
    batch = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=2)
    )
    assert batch is not None
    assert batch.num_global_valid_tokens > 0


def test_microbatch_grid_spreads_pad_rows_across_cells() -> None:
    # 5 real rows, local_batch_size=2, dp_degree=2 -> 4 cells x 2 = 8 rows (3 pad).
    # Round-robin dealing spreads the pad rows so no (microbatch, rank) cell is all-pad.
    batcher = Batcher.Config(batch=BatchConfig(local_batch_size=2, seq_len=2)).build(
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
    )
    batch = batcher.add_training_samples(
        training_sample_group=_trainable_group(0, num_samples=5)
    )
    assert batch is not None
    cells = [microbatch for ranks in batch.microbatches for microbatch in ranks]
    assert len(cells) == 4  # 2 microbatches x 2 ranks
    for cell in cells:
        assert cell.loss_mask.any(dim=1).any()  # at least one real (non-pad) row


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
            num_prompts_per_train_step=1,
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
            target_offpolicy_steps=3,
            max_offpolicy_steps=3,
        )


def test_compute_policy_age_metrics_uses_hard_offpolicy_limit() -> None:
    metrics = compute_policy_age_metrics(
        trainer_policy_version=4,
        min_policy_versions=[0],
        target_offpolicy_steps=3,
        max_offpolicy_steps=4,
    )
    assert any(metric.key == "train_batch/policy_age_max" for metric in metrics)

    with pytest.raises(RuntimeError, match="admitted stale training data"):
        compute_policy_age_metrics(
            trainer_policy_version=5,
            min_policy_versions=[0],
            target_offpolicy_steps=3,
            max_offpolicy_steps=4,
        )


def _fifo_buffer(*, capacity: int, window_size: int = 1) -> RolloutGroupWorkBuffer:
    return RolloutGroupWorkBuffer.Config().build(
        max_active_rollout_groups=capacity,
        window_size=window_size,
    )


def test_work_buffer_rejects_window_larger_than_capacity() -> None:
    with pytest.raises(ValueError, match="window_size"):
        _fifo_buffer(capacity=2, window_size=3)


async def _admit(buffer: RolloutGroupWorkBuffer, group_id: int) -> None:
    if not await buffer.wait_for_slot():
        raise RuntimeError("buffer closed unexpectedly")
    await buffer.add_work(RolloutGroupWork(group_id=group_id, sample=object()))


async def _finalize(buffer: RolloutGroupWorkBuffer, group_id: int) -> None:
    await buffer.finalize_work(RolloutGroup(group_id=group_id, rollouts=[]))


def test_derive_window_size_from_window_fifo_fraction() -> None:
    # P=3, S=2 -> B=9.
    assert (
        derive_window_size(
            num_prompts_per_train_step=3,
            target_offpolicy_steps=2,
            window_fifo_fraction=4 / 9,
        )
        == 4
    )
    assert (
        derive_window_size(
            num_prompts_per_train_step=3,
            target_offpolicy_steps=2,
            window_fifo_fraction=1 / 9,
        )
        == 1
    )
    assert (
        derive_window_size(
            num_prompts_per_train_step=3,
            target_offpolicy_steps=2,
            window_fifo_fraction=1.0,
        )
        == 9
    )
    assert (
        derive_window_size(
            num_prompts_per_train_step=8,
            target_offpolicy_steps=3,
            window_fifo_fraction=1.0,
        )
        == 32
    )


def test_windowed_fifo_takes_within_anchored_window() -> None:
    async def run() -> None:
        # Window [g0, g3]: g1/g2/g3 may bypass stuck g0; g4 remains blocked.
        buffer = _fifo_buffer(capacity=8, window_size=4)
        for group_id in range(5):
            await _admit(buffer, group_id)
        await buffer.claim_next()  # g0 -> INFLIGHT and stuck
        for group_id in (1, 2, 3, 4):
            await _finalize(buffer, group_id)

        assert (await buffer.take_finalized()).group_id == 1
        assert (await buffer.take_finalized()).group_id == 2
        assert (await buffer.take_finalized()).group_id == 3

        taker = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0)
        assert not taker.done()  # g4 is finalized but outside the anchored window

        await _finalize(buffer, 0)
        assert (await taker).group_id == 0
        assert (await buffer.take_finalized()).group_id == 4

    asyncio.run(run())
