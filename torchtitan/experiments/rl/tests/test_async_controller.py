# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for async-controller pieces: batcher group-counting, the active-slot buffer backpressure,
the consume-time staleness invariant, the metrics timer drain, and RolloutTurnID."""

import asyncio

import pytest
import torch

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
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_trainable_group(0, num_samples=8)
    )
    assert batch is None
    assert group_is_trainable
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=1)
    )
    assert batch is not None
    assert group_is_trainable


def test_batcher_carries_metric_only_groups_until_trainable_batch() -> None:
    # Metric-only (empty) groups do not count toward the target and cannot form a zero-token batch;
    # they ride along until a trainable group completes the batch.
    batcher = _build_batcher(num_prompts_per_train_step=1)
    metric_only = TrainingSampleGroup(group_id=0, training_samples=[], metrics=[])
    assert batcher.add_training_samples(training_sample_group=metric_only) == (
        None,
        False,
    )
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=2)
    )
    assert batch is not None
    assert group_is_trainable
    assert batch.num_global_valid_tokens > 0


def test_packing_flushes_microbatch_on_overflow() -> None:
    # Each rank has a four-token budget. Four two-token samples fill the first
    # microbatch across two ranks; the fifth starts a second microbatch.
    batcher = Batcher.Config(batch=BatchConfig(local_batch_size=2, seq_len=2)).build(
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
    )
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_trainable_group(0, num_samples=5)
    )
    assert batch is not None
    assert group_is_trainable
    assert len(batch.microbatches) == 2  # flushed once on overflow
    assert all(len(ranks) == 2 for ranks in batch.microbatches)  # dp_degree ranks each
    assert all(
        microbatch.token_ids.shape == (4,)
        for ranks in batch.microbatches
        for microbatch in ranks
    )
    # Every sample is trained: 5 samples x 2 trained tokens each.
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
    metric_value = next(metric.value for metric in batch.metrics if metric.key == key)
    assert isinstance(metric_value, m.NoReduce)
    return metric_value.value


def test_packing_uses_the_full_flat_rank_token_budget() -> None:
    batcher = Batcher.Config(batch=BatchConfig(local_batch_size=2, seq_len=4)).build(
        num_prompts_per_train_step=1, dp_degree=1, pad_id=0
    )
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_variable_length_group(0, lengths=[4, 4, 3])
    )

    assert batch is not None
    assert group_is_trainable
    assert len(batch.microbatches) == 1
    microbatch = batch.microbatches[0][0]
    assert microbatch.token_ids.shape == (8,)
    assert microbatch.positions.tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
    assert _metric_value(batch, "train_batch/padding_frac") == 0.0


def test_packing_balances_dp_rank_square_cost() -> None:
    # A few long sequences among many short ones. The longest-processing-time
    # assignment evens the per-rank full-attention cost.
    lengths = [63, 63, 33, 33, 33, 17, 17, 17, 17, 9, 9, 9, 9, 9, 9, 5, 5, 5]
    batcher = Batcher.Config(batch=BatchConfig(local_batch_size=2, seq_len=64)).build(
        num_prompts_per_train_step=1, dp_degree=2, pad_id=0
    )
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_variable_length_group(0, lengths=lengths)
    )
    assert batch is not None
    assert group_is_trainable
    # 1.0 is perfect DP-rank balance; count-based round-robin on this data is ~1.1.
    assert _metric_value(batch, "train_batch/cost_imbalance") <= 1.05


def test_packing_minimizes_microbatches_on_bad_arrival_order() -> None:
    # One rank has a 64-token budget. Longest-first packing pairs the two
    # effective length-29 samples, producing three microbatches instead of four.
    lengths = [40, 30, 40, 30]
    batcher = Batcher.Config(batch=BatchConfig(local_batch_size=1, seq_len=64)).build(
        num_prompts_per_train_step=1, dp_degree=1, pad_id=0
    )
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_variable_length_group(0, lengths=lengths)
    )
    assert batch is not None
    assert group_is_trainable
    assert _metric_value(batch, "train_batch/num_microbatches") == 3


def test_packing_cost_model_receives_1d_sequence_lengths() -> None:
    observed_inputs: list[torch.Tensor] = []

    def linear_cost(sequence_lengths: torch.Tensor) -> torch.Tensor:
        observed_inputs.append(sequence_lengths.clone())
        return sequence_lengths

    batcher = Batcher.Config(
        batch=BatchConfig(local_batch_size=1, seq_len=16),
        cost_model=linear_cost,
    ).build(num_prompts_per_train_step=1, dp_degree=2, pad_id=0)
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_variable_length_group(0, lengths=[9, 7, 5])
    )

    assert batch is not None
    assert group_is_trainable
    assert observed_inputs
    assert all(sequence_lengths.ndim == 1 for sequence_lengths in observed_inputs)
    assert observed_inputs[0].tolist() == [8, 6, 4]


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
        assert batcher.add_training_samples(
            training_sample_group=training_sample_group
        ) == (
            None,
            False,
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
