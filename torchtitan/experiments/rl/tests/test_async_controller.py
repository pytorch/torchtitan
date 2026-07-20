# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for async-controller pieces: batcher group-counting, the active-slot buffer backpressure,
the consume-time staleness invariant, the metrics timer drain, and RolloutTurnID."""

import asyncio

import pytest

from torchtitan.config import BatchConfig
from torchtitan.experiments.rl.components.batcher import Batcher
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
        buffer = RolloutGroupWorkBuffer.Config().build(
            max_active_rollout_groups=1, num_prompts_per_train_step=1
        )
        if not await buffer.wait_for_slot():
            raise RuntimeError("buffer closed unexpectedly")
        await buffer.add_work(RolloutGroupWork(group_id=0, sample=object()))
        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[]))
        await buffer.take_finalized(pending_trainable_count=0)

        waiter = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not waiter.done()

        await buffer.release_active_groups(1, reason="trained")
        assert await waiter

    asyncio.run(run())


def test_untrainable_group_releases_before_training() -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config().build(
            max_active_rollout_groups=1, num_prompts_per_train_step=1
        )
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
            max_offpolicy_steps=3,
        )


def _fifo_buffer(
    *,
    capacity: int,
    window_lookahead_steps: int = 0,
    num_prompts_per_train_step: int = 1,
) -> RolloutGroupWorkBuffer:
    return RolloutGroupWorkBuffer.Config(
        window_lookahead_steps=window_lookahead_steps
    ).build(
        max_active_rollout_groups=capacity,
        num_prompts_per_train_step=num_prompts_per_train_step,
    )


async def _admit(buffer: RolloutGroupWorkBuffer, group_id: int) -> None:
    if not await buffer.wait_for_slot():
        raise RuntimeError("buffer closed unexpectedly")
    await buffer.add_work(RolloutGroupWork(group_id=group_id, sample=object()))


async def _finalize(buffer: RolloutGroupWorkBuffer, group_id: int) -> None:
    await buffer.finalize_work(RolloutGroup(group_id=group_id, rollouts=[]))


def test_window_lookahead_steps_negative_raises() -> None:
    with pytest.raises(ValueError, match="window_lookahead_steps"):
        _fifo_buffer(capacity=10, window_lookahead_steps=-1)


def test_num_prompts_per_train_step_invalid_raises() -> None:
    with pytest.raises(ValueError, match="num_prompts_per_train_step"):
        _fifo_buffer(capacity=10, num_prompts_per_train_step=0)


def test_window_lookahead_steps_property() -> None:
    assert _fifo_buffer(capacity=10).window_lookahead_steps == 0
    assert (
        _fifo_buffer(capacity=10, window_lookahead_steps=2).window_lookahead_steps == 2
    )


def test_strict_fifo_blocks_on_unfinalized_head() -> None:
    async def run() -> None:
        buffer = _fifo_buffer(capacity=4)  # window_lookahead_steps=0 -> strict FIFO
        for group_id in range(2):
            await _admit(buffer, group_id)
        await buffer.claim_next()  # g0 -> INFLIGHT (head stuck)
        await _finalize(buffer, 1)  # only g1 finalized

        taker = asyncio.create_task(buffer.take_finalized(pending_trainable_count=0))
        await asyncio.sleep(0)
        assert not taker.done()  # strict FIFO: g1 finalized but head g0 is not -> stall

        await _finalize(buffer, 0)
        assert (await taker).group_id == 0  # head returned first

    asyncio.run(run())


def test_windowed_fifo_greedy_within_window_and_anchored_at_head() -> None:
    async def run() -> None:
        # s=1, P=2, head-phase r0=0 -> window_end = h + (s+1)*P - r0 - 1 = h + 3.
        # One extra train-step's worth (P=2 groups) is bypassable ahead of the stuck head.
        buffer = _fifo_buffer(
            capacity=10, window_lookahead_steps=1, num_prompts_per_train_step=2
        )
        for group_id in range(5):
            await _admit(buffer, group_id)
        await buffer.claim_next()  # g0 -> INFLIGHT (head stuck)
        for group_id in (1, 2, 3, 4):
            await _finalize(buffer, group_id)

        # Greedy within the window [g0, g3]: g1, g2, g3 fetched ahead of the stuck head.
        assert (await buffer.take_finalized(pending_trainable_count=0)).group_id == 1
        assert (await buffer.take_finalized(pending_trainable_count=0)).group_id == 2
        assert (await buffer.take_finalized(pending_trainable_count=0)).group_id == 3

        # Anchored: consuming g1/g2/g3 does NOT move the head, so g4 stays outside [g0, g3].
        taker = asyncio.create_task(buffer.take_finalized(pending_trainable_count=0))
        await asyncio.sleep(0)
        assert not taker.done()  # g4 finalized but beyond the window -> blocked

        # Consuming the head slides the window; g4 becomes eligible.
        await _finalize(buffer, 0)
        assert (await taker).group_id == 0
        assert (await buffer.take_finalized(pending_trainable_count=0)).group_id == 4

    asyncio.run(run())


def test_window_lookahead_is_phase_exact_across_r0() -> None:
    # Same s and P, different head-phase r0 at head-stall must bypass the SAME number of groups
    # (exactly (s+1)*P - r0 non-head entries fit) -- the property fixed-entry windows lacked.
    async def _bypassable_count(*, r0: int, s: int, p: int) -> int:
        # Big capacity so the active-slot budget never limits the window under test.
        buffer = _fifo_buffer(
            capacity=64, window_lookahead_steps=s, num_prompts_per_train_step=p
        )
        num_groups = (s + 2) * p + 1
        for group_id in range(num_groups):
            await _admit(buffer, group_id)
        await buffer.claim_next()  # g0 -> INFLIGHT (head stuck at phase r0)
        for group_id in range(1, num_groups):
            await _finalize(buffer, group_id)

        taken = 0
        while True:
            taker = asyncio.create_task(
                buffer.take_finalized(pending_trainable_count=r0)
            )
            await asyncio.sleep(0)
            if not taker.done():
                taker.cancel()
                break
            await taker
            taken += 1
        return taken

    async def run() -> None:
        p, s = 4, 1
        # window size W = (s+1)*P - r0; non-head bypassable entries = W - 1.
        assert await _bypassable_count(r0=0, s=s, p=p) == (s + 1) * p - 0 - 1
        assert await _bypassable_count(r0=p - 1, s=s, p=p) == (s + 1) * p - (p - 1) - 1
        # The EXTRA train-steps both incur is floor((r0 + (W-1)) / P) = s in both cases.
        for r0 in (0, p - 1):
            bypassed = await _bypassable_count(r0=r0, s=s, p=p)
            window_size = bypassed + 1  # + the head itself
            assert (r0 + (window_size - 1)) // p == s

    asyncio.run(run())


def test_window_anchored_until_head_consumed() -> None:
    # Consuming non-head groups must NOT slide the window (r0 held) until the head leaves.
    async def run() -> None:
        # s=1, P=2, r0=0 -> window_end = h + 3.
        buffer = _fifo_buffer(
            capacity=10, window_lookahead_steps=1, num_prompts_per_train_step=2
        )
        for group_id in range(6):
            await _admit(buffer, group_id)
        await buffer.claim_next()  # g0 stuck
        for group_id in (1, 2, 3, 4, 5):
            await _finalize(buffer, group_id)

        # g1..g3 consumable; g4, g5 outside [g0, g3].
        assert (await buffer.take_finalized(pending_trainable_count=0)).group_id == 1
        assert (await buffer.take_finalized(pending_trainable_count=0)).group_id == 2
        assert (await buffer.take_finalized(pending_trainable_count=0)).group_id == 3
        taker = asyncio.create_task(buffer.take_finalized(pending_trainable_count=0))
        await asyncio.sleep(0)
        assert not taker.done()  # window did not slide despite 3 holes

        # Now consume the head; window slides to g4 and g4/g5 become eligible.
        await _finalize(buffer, 0)
        assert (await taker).group_id == 0
        assert (await buffer.take_finalized(pending_trainable_count=0)).group_id == 4
        assert (await buffer.take_finalized(pending_trainable_count=0)).group_id == 5

    asyncio.run(run())


def test_policy_age_tolerance_accounts_for_window_lookahead() -> None:
    # Hard bound = max_offpolicy_steps + window_lookahead_steps = 3 + 2 = 5 (exact).
    # Age exactly at the bound is allowed; one past it raises.
    metrics = compute_policy_age_metrics(
        trainer_policy_version=5,
        min_policy_versions=[0],
        max_offpolicy_steps=3,
        window_lookahead_steps=2,
    )
    assert any(metric.key == "train_batch/policy_age_max" for metric in metrics)
    with pytest.raises(RuntimeError, match="admitted stale training data"):
        compute_policy_age_metrics(
            trainer_policy_version=6,
            min_policy_versions=[0],
            max_offpolicy_steps=3,
            window_lookahead_steps=2,
        )
