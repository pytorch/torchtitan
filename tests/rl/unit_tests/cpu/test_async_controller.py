# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for async-controller pieces: batcher group-counting, the active-slot buffer backpressure,
the consume-time staleness invariant, the metrics timer drain, and RolloutTurnID."""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch
from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.rl.controller.metrics import (
    compute_perf_ratio_metrics,
    compute_policy_age_metrics,
    compute_rollout_metrics,
    MetricsTimer,
)
from torchtitan.rl.controller.work_buffer import (
    RolloutGroupWork,
    RolloutGroupWorkBuffer,
)
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import RolloutGroup
from torchtitan.rl.rollout.types import (
    Rollout,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
)

from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleGroup,
)

from tests.rl.unit_tests.cpu.test_trainer_loop import fake_controller


def test_controller_config_maybe_log(tmp_path, caplog) -> None:
    from torchtitan_recipes.tests.rl.alphabet_sort import rl_grpo_qwen3_5_debug_varlen

    config = rl_grpo_qwen3_5_debug_varlen(seq_len=128)
    assert config.generator.max_num_batched_tokens == 128
    config.dump_folder = str(tmp_path)
    config.trainer.debug.print_config = True
    config.trainer.debug.save_config_file = "config.json"

    with caplog.at_level(logging.INFO, logger="torchtitan.rl.controller.controller"):
        config.maybe_log()

    assert "Running with configs:" in caplog.text
    with open(tmp_path / "config.json") as file:
        saved_config = json.load(file)
    assert saved_config["trainer"]["model"]["max_context_length"] == 128


def _training_sample(*, group_id: int, rollout_id: int) -> TrainingSample:
    return TrainingSample(
        min_policy_version=0,
        max_policy_version=0,
        rollout_id=RolloutTurnID(group_id=group_id, rollout_id=rollout_id, turn_id=0),
        token_ids=np.array([1, 2, 3]),
        loss_mask=np.array([False, True, True]),
        logprobs=np.array([0.0, 0.1, 0.2], dtype=np.float32),
        advantage=np.array([0.0, 1.0, 1.0], dtype=np.float32),
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


def _variable_length_group(
    group_id: int, *, token_lengths: list[int]
) -> TrainingSampleGroup:
    samples = []
    for rollout_id, token_length in enumerate(token_lengths):
        samples.append(
            TrainingSample(
                min_policy_version=0,
                max_policy_version=0,
                rollout_id=RolloutTurnID(
                    group_id=group_id,
                    rollout_id=rollout_id,
                    turn_id=0,
                ),
                token_ids=np.arange(token_length),
                loss_mask=np.array([False] + [True] * (token_length - 1)),
                logprobs=np.zeros(token_length, dtype=np.float32),
                advantage=np.array(
                    [0.0] + [1.0] * (token_length - 1), dtype=np.float32
                ),
            )
        )
    return TrainingSampleGroup(group_id=group_id, training_samples=samples, metrics=[])


def _metric_value(batch, key: str) -> float:
    metric = next(metric for metric in batch.metrics if metric.key == key)
    assert isinstance(metric.value, m.NoReduce)
    return metric.value.value


def _untrainable_group(group_id: int) -> TrainingSampleGroup:
    return TrainingSampleGroup(group_id=group_id, training_samples=[], metrics=[])


def _build_batcher(
    *, num_prompts_per_train_step: int, num_mtp_layers: int = 0
) -> Batcher:
    config = Batcher.Config()
    config.num_mtp_layers = num_mtp_layers
    return config.build(
        num_tokens_per_microbatch_per_dp_rank=16384,
        max_context_length=2048,
        num_prompts_per_train_step=num_prompts_per_train_step,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
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


def test_batcher_packs_groups_in_id_order_regardless_of_arrival() -> None:
    batcher = _build_batcher(num_prompts_per_train_step=2)
    late_group = _trainable_group(7, num_samples=1)
    early_group = _trainable_group(3, num_samples=1)
    late_group.training_samples[0].min_policy_version = 7
    early_group.training_samples[0].min_policy_version = 3

    # g7 finishes first; the packed batch still lists g3 before g7.
    batcher.add_training_samples(training_sample_group=late_group)
    batch, _ = batcher.add_training_samples(training_sample_group=early_group)

    assert batch is not None
    assert batch.min_policy_versions == [3, 7]
    assert batch.group_ids == [3, 7]


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
    assert batch.global_loss_token_counts[0] > 0
    assert batch.global_routing_token_counts.shape == (1,)
    assert batch.group_ids == [0, 1]


def test_batcher_prepares_per_depth_mtp_token_counts() -> None:
    batcher = _build_batcher(num_prompts_per_train_step=1, num_mtp_layers=2)
    batch, _ = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=2)
    )

    assert batch is not None
    assert batch.global_loss_token_counts.shape == (3,)
    assert batch.global_routing_token_counts.shape == (3,)


def test_batcher_warns_after_each_batch_of_untrainable_groups(
    caplog: pytest.LogCaptureFixture,
) -> None:
    batcher = _build_batcher(num_prompts_per_train_step=2)

    with caplog.at_level(logging.WARNING):
        batcher.add_training_samples(training_sample_group=_untrainable_group(0))
        batcher.add_training_samples(training_sample_group=_untrainable_group(1))

    assert (
        "Consecutive untrainable batches: 1/10 "
        "(2 rollout groups produced no trainable samples)."
    ) in caplog.text


def test_batcher_resets_no_progress_count_on_trainable_group(
    caplog: pytest.LogCaptureFixture,
) -> None:
    batcher = _build_batcher(num_prompts_per_train_step=2)

    with caplog.at_level(logging.WARNING):
        batcher.add_training_samples(training_sample_group=_untrainable_group(0))
        batcher.add_training_samples(training_sample_group=_untrainable_group(1))
        batcher.add_training_samples(
            training_sample_group=_trainable_group(2, num_samples=1)
        )
        caplog.clear()
        batcher.add_training_samples(training_sample_group=_untrainable_group(3))

    assert "zero-output batch equivalents" not in caplog.text


def test_batcher_raises_at_consecutive_untrainable_group_limit() -> None:
    batcher = _build_batcher(num_prompts_per_train_step=2)

    for group_id in range(19):
        batcher.add_training_samples(training_sample_group=_untrainable_group(group_id))

    with pytest.raises(RuntimeError, match="10 consecutive untrainable batches"):
        batcher.add_training_samples(training_sample_group=_untrainable_group(19))


def test_dp_assignment_avoids_all_padding_ranks_when_possible() -> None:
    # Five two-token samples need four rank inputs across two microbatches.
    # Redistributing one sample keeps every rank input trainable.
    batcher = Batcher.Config(max_num_documents=4).build(
        num_tokens_per_microbatch_per_dp_rank=4,
        max_context_length=2,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_trainable_group(0, num_samples=5)
    )
    assert batch is not None
    assert group_is_trainable
    cells = [
        microbatch
        for group in batch.microbatch_groups
        for ranks in group
        for microbatch in ranks
    ]
    assert len(cells) == 4  # 2 microbatches x 2 ranks
    for cell in cells:
        assert cell.loss_mask.any()
        assert cell.padding_mask.shape == cell.input.shape
        assert not cell.padding_mask[cell.loss_mask].any()


def test_batcher_uses_flat_rank_capacity_and_reports_padding() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_variable_length_group(
            0,
            token_lengths=[4, 4, 3],
        )
    )

    assert batch is not None
    assert group_is_trainable
    assert len(batch.microbatch_groups) == 1
    microbatch = batch.microbatch_groups[0][0][0]
    assert microbatch.positions.tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
    assert not microbatch.padding_mask.any()
    torch.testing.assert_close(microbatch.loss_token_counts, torch.tensor([8]))
    assert _metric_value(batch, "train_batch/padding_frac") == 0.0


def test_flat_rank_packing_preserves_padding_mask() -> None:
    batcher = Batcher.Config(per_sample_pad_multiple=4).build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    batch, _ = batcher.add_training_samples(
        training_sample_group=_variable_length_group(0, token_lengths=[4])
    )

    assert batch is not None
    microbatch = batch.microbatch_groups[0][0][0]
    assert microbatch.positions.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    torch.testing.assert_close(microbatch.loss_token_counts, torch.tensor([3]))
    assert microbatch.padding_mask.tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_batcher_balances_packing_across_dp_ranks() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_variable_length_group(
            0,
            token_lengths=[5, 4, 4, 3],
        )
    )

    assert batch is not None
    assert group_is_trainable
    assert [
        [int((~rank.padding_mask).sum().item()) for rank in microbatch]
        for (microbatch,) in batch.microbatch_groups
    ] == [[6, 6]]


def test_batcher_fills_new_bin_from_multiple_heaviest_bins() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=40,
        max_context_length=40,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    samples = _variable_length_group(
        0,
        token_lengths=[11] * 12,
    ).training_samples
    bins = [samples[:4], samples[4:8], samples[8:]]

    batcher._expand_bins_by_splitting(bins, target_num_bins=4)

    assert [len(bin_) for bin_ in bins] == [3, 3, 3, 3]
    assert [batcher._attention_workload(bin_) for bin_ in bins] == [400] * 4


def test_batcher_pads_when_no_bin_can_donate_a_sample() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=10,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    samples = _variable_length_group(0, token_lengths=[6, 6]).training_samples
    bins = [[samples[0]], [samples[1]]]

    batcher._expand_bins_by_splitting(bins, target_num_bins=4)

    assert bins == [[samples[0]], [samples[1]], [], []]


def test_batcher_splits_sorts_and_zigzags_by_attention_workload() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=10,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _variable_length_group(
        0,
        # Effective lengths are [6, 6, 6, 4, 4, 4]. FFD finds three bins,
        # then LPT repacks into four DP inputs.
        token_lengths=[7, 7, 7, 5, 5, 5],
    ).training_samples

    assignments = batcher._assign_training_samples_to_microbatches(samples)
    workloads = [
        [batcher._attention_workload(rank_samples) for rank_samples in microbatch]
        for microbatch in assignments
    ]

    # Global sorting puts similarly expensive bins in each concurrent DP group.
    # Reversing the second group pairs its lighter bin with the first rank.
    assert workloads == [[52, 52], [36, 52]]


def test_batcher_zigzags_workloads_across_dp_ranks() -> None:
    batcher = Batcher.Config(max_num_documents=1).build(
        num_tokens_per_microbatch_per_dp_rank=10,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _variable_length_group(
        0,
        # Effective lengths are [10, 8, 6, 4]. The document limit forces each
        # sample into its own bin, with padding-aware workloads [100, 68, 52, 52].
        token_lengths=[11, 9, 7, 5],
    ).training_samples

    assignments = batcher._assign_training_samples_to_microbatches(samples)
    workloads = [
        [batcher._attention_workload(rank_samples) for rank_samples in microbatch]
        for microbatch in assignments
    ]

    assert workloads == [[100, 68], [52, 52]]
    assert [sum(rank_workloads) for rank_workloads in zip(*workloads)] == [152, 120]


def test_batcher_reports_padding_when_document_limit_blocks_greedy_order() -> None:
    batcher = Batcher.Config(max_num_documents=3).build(
        num_tokens_per_microbatch_per_dp_rank=6,
        max_context_length=3,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    batch, _ = batcher.add_training_samples(
        training_sample_group=_variable_length_group(
            0,
            token_lengths=[2, 2, 4, 2, 2, 4],
        )
    )

    assert batch is not None
    assert len(batch.microbatch_groups) == 3
    assert all(
        microbatch.padding_mask.numel() == 6
        for ((microbatch,),) in batch.microbatch_groups
    )
    assert _metric_value(batch, "train_batch/padding_frac") == pytest.approx(4 / 9)


def test_document_limit_applies_to_each_local_microbatch() -> None:
    # A local microbatch is capped at three documents. The batcher must keep all
    # five documents and distribute them 2 + 3 across two microbatches.
    batcher = Batcher.Config(max_num_documents=3).build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_trainable_group(0, num_samples=5)
    )

    assert batch is not None
    assert group_is_trainable
    assert len(batch.microbatch_groups) == 2
    num_documents = []
    for ((microbatch,),) in batch.microbatch_groups:
        real_document_starts = (microbatch.positions == 0) & ~microbatch.padding_mask
        num_documents.append(int(real_document_starts.sum().item()))
    assert sorted(num_documents) == [2, 3]
    assert sum(num_documents) == 5


def test_document_limit_can_be_smaller_than_rows_per_microbatch() -> None:
    batcher = Batcher.Config(max_num_documents=1).build(
        num_tokens_per_microbatch_per_dp_rank=4,
        max_context_length=2,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    batch, _ = batcher.add_training_samples(
        training_sample_group=_trainable_group(0, num_samples=2)
    )

    assert batch is not None
    assert len(batch.microbatch_groups) == 2
    for ((microbatch,),) in batch.microbatch_groups:
        real_document_starts = (microbatch.positions == 0) & ~microbatch.padding_mask
        assert int(real_document_starts.sum().item()) == 1


def test_batcher_filters_training_samples_longer_than_context() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=4,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    sample = _training_sample(group_id=0, rollout_id=0)
    sample.token_ids = np.arange(6)
    sample.loss_mask = np.zeros(6, dtype=bool)
    sample.logprobs = np.zeros(6, dtype=np.float32)
    sample.advantage = np.zeros(6, dtype=np.float32)

    pending, group_is_trainable = batcher.add_training_samples(
        training_sample_group=TrainingSampleGroup(
            group_id=0,
            training_samples=[sample],
            metrics=[],
        )
    )
    assert pending is None
    assert not group_is_trainable

    batch, _ = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=1)
    )
    assert batch is not None
    assert batch.group_ids == [0, 1]
    dropped_metric = next(
        metric
        for metric in batch.metrics
        if metric.key == "batcher/num_samples_dropped_oversized"
    )
    assert dropped_metric.value.value == 1


def test_batcher_requires_whole_rows_per_microbatch() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        Batcher.Config().build(
            num_tokens_per_microbatch_per_dp_rank=5,
            max_context_length=3,
            num_prompts_per_train_step=1,
            dp_degree=1,
            pad_id=0,
            temperature=1.0,
        )


def test_compute_perf_ratio_metrics_reads_flushed_means() -> None:
    time_metrics = [
        m.Metric("timing/step/total", m.Mean.from_list([2.0])),
        m.Metric("timing/step/forward_backward", m.Mean.from_list([0.5])),
        m.Metric("timing/step/optimizer", m.Mean.from_list([0.5])),
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
            max_active_rollout_groups=1, window_size=None
        )
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
        buffer = RolloutGroupWorkBuffer.Config().build(
            max_active_rollout_groups=1, window_size=None
        )
        batcher = Batcher.Config().build(
            num_tokens_per_microbatch_per_dp_rank=16384,
            max_context_length=2048,
            num_prompts_per_train_step=1,
            dp_degree=1,
            pad_id=0,
            temperature=1.0,
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


def test_compute_policy_age_metrics_raises_beyond_cap() -> None:
    # cap 4 (S=3, windowed_fifo_batches=1): age 4 passes, age 5 raises
    metrics = compute_policy_age_metrics(
        trainer_policy_version=4,
        min_policy_versions=[0],
        target_offpolicy_steps=3,
        max_offpolicy_steps=4,
    )
    aggregated = m.MetricsProcessor._aggregate_metrics(metrics)
    assert aggregated["train_batch/policy_age_max"] == 4
    assert aggregated["train_batch/pct_samples_over_target_age"] == 100.0

    with pytest.raises(RuntimeError, match="admitted stale training data"):
        compute_policy_age_metrics(
            trainer_policy_version=5,
            min_policy_versions=[0],
            target_offpolicy_steps=3,
            max_offpolicy_steps=4,
        )


def test_compute_policy_age_metrics_uncapped_trains_over_target_age_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # no cap; one sample comes back at age S+3=6: counted and warned, not raised
    with caplog.at_level(logging.WARNING):
        metrics = compute_policy_age_metrics(
            trainer_policy_version=10,
            min_policy_versions=[4, 9, 8],
            target_offpolicy_steps=3,
            max_offpolicy_steps=None,
        )

    aggregated = m.MetricsProcessor._aggregate_metrics(metrics)
    assert aggregated["train_batch/policy_age/mean"] == 3
    assert aggregated["train_batch/policy_age_max"] == 6
    assert aggregated["train_batch/pct_samples_over_target_age"] == pytest.approx(
        100 / 3
    )
    assert "1 samples (33.3%) older than target_offpolicy_steps=3" in caplog.text


def test_compute_policy_age_metrics_uncapped_is_quiet_within_target(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        metrics = compute_policy_age_metrics(
            trainer_policy_version=10,
            min_policy_versions=[7, 9],
            target_offpolicy_steps=3,
            max_offpolicy_steps=None,
        )

    aggregated = m.MetricsProcessor._aggregate_metrics(metrics)
    assert aggregated["train_batch/policy_age/mean"] == 2
    assert aggregated["train_batch/pct_samples_over_target_age"] == 0.0
    assert caplog.text == ""


def _buffer(*, capacity: int, window_size: int | None) -> RolloutGroupWorkBuffer:
    return RolloutGroupWorkBuffer.Config().build(
        max_active_rollout_groups=capacity, window_size=window_size
    )


async def _admit(buffer: RolloutGroupWorkBuffer, group_id: int) -> None:
    if not await buffer.wait_for_slot():
        raise RuntimeError("buffer closed unexpectedly")
    await buffer.add_work(RolloutGroupWork(group_id=group_id, sample=object()))


async def _finalize(buffer: RolloutGroupWorkBuffer, group_id: int) -> None:
    await buffer.finalize_work(RolloutGroup(group_id=group_id, rollouts=[]))


def test_windowed_fifo_takes_within_anchored_window() -> None:
    async def run() -> None:
        # P=4, windowed_fifo_batches=1 -> window of 4 ids [g0, g3]: g1/g2/g3 may bypass stuck g0; g4 waits.
        buffer = _buffer(capacity=8, window_size=4)
        for group_id in range(5):
            await _admit(buffer, group_id)
        claimed_group_ids = [(await buffer.claim_next()).group_id for _ in range(5)]
        assert claimed_group_ids == [0, 1, 2, 3, 4]
        for group_id in (1, 2, 3, 4):
            await _finalize(buffer, group_id)

        assert (await buffer.take_finalized()).group_id == 1
        assert (await buffer.take_finalized()).group_id == 2
        assert (await buffer.take_finalized()).group_id == 3

        taker = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0)
        assert not taker.done()  # g4 is finalized but outside the anchored window

        await _finalize(buffer, 0)
        await asyncio.sleep(0)
        assert taker.done()
        assert taker.result().group_id == 0
        assert (await buffer.take_finalized()).group_id == 4

    asyncio.run(run())


def test_no_window_takes_oldest_ready_group_past_a_stuck_head() -> None:
    async def run() -> None:
        # S=1, P=4 -> 8 slots, no window. g0..g4 CLAIMED; g0 never finishes; g1..g4 finish out of id order.
        buffer = _buffer(capacity=8, window_size=None)
        for group_id in range(8):
            await _admit(buffer, group_id)
        claimed_group_ids = [(await buffer.claim_next()).group_id for _ in range(5)]
        assert claimed_group_ids == [0, 1, 2, 3, 4]
        for group_id in (3, 1, 4, 2):
            await _finalize(buffer, group_id)

        # A full batch of P=4 is taken oldest-ready first, without waiting on g0.
        for expected_group_id in (1, 2, 3, 4):
            taker = asyncio.create_task(buffer.take_finalized())
            await asyncio.sleep(0)
            assert taker.done()
            assert taker.result().group_id == expected_group_id

        # The remaining wait is for generation (nothing finalized), not for the head g0.
        taker = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0)
        assert not taker.done()
        await buffer.claim_next()  # g5 -> CLAIMED
        await buffer.claim_next()  # g6 -> CLAIMED
        await _finalize(buffer, 6)
        await asyncio.sleep(0)
        assert taker.done()
        assert taker.result().group_id == 6

    asyncio.run(run())


async def _block_forever(*args, **kwargs) -> None:
    await asyncio.Event().wait()


def _validation_rollout(versions: list[tuple[int, int]]) -> Rollout:
    """One rollout with a turn per `(min_policy_version, max_policy_version)`."""
    return Rollout(
        group_id=-1,
        rollout_id=0,
        status=RolloutStatus.COMPLETED,
        reward=1.0,
        turns=[
            RolloutTurn(
                rollout_id=RolloutTurnID(group_id=-1, rollout_id=0, turn_id=turn_id),
                prompt_prefix_len=0,
                prompt_delta_token_ids=np.array([1], dtype=np.int32),
                completion_token_ids=np.array([2], dtype=np.int32),
                completion_logprobs=np.array([-0.1], dtype=np.float32),
                min_policy_version=min_version,
                max_policy_version=max_version,
            )
            for turn_id, (min_version, max_version) in enumerate(versions)
        ],
    )


@pytest.mark.parametrize("step, group_ids", [(0, [-1, -2]), (25, [-51, -52])])
def test_validation_group_ids_and_sampled_policies(step, group_ids) -> None:
    """A pass's group ids come from its step and the prompt index, so passes running side by side
    never share one, and each pass releases only its own. A pass logs the policies it sampled: a
    weight sync during a rollout shows as a newer max version and a mixed-policy rollout."""
    controller = object.__new__(Controller)
    controller._validation_samples = ["prompt", "prompt"]
    controller.generator_router = SimpleNamespace(
        release_groups=SimpleNamespace(call_one=AsyncMock())
    )
    controller._rollouter = SimpleNamespace(
        run_group_rollouts=AsyncMock(
            side_effect=[
                # A sync to the next policy landed during the second turn.
                RolloutGroup(
                    group_id=group_ids[0],
                    rollouts=[_validation_rollout([(step, step), (step, step + 2)])],
                ),
                RolloutGroup(
                    group_id=group_ids[1],
                    rollouts=[_validation_rollout([(step, step)])],
                ),
            ]
        ),
    )

    _, metrics = asyncio.run(
        controller._collect_validation_rollouts(sampling=SamplingConfig(), step=step)
    )

    run_group_ids = [
        call.kwargs["group_id"]
        for call in controller._rollouter.run_group_rollouts.call_args_list
    ]
    assert run_group_ids == group_ids
    controller.generator_router.release_groups.call_one.assert_awaited_once_with(
        group_ids
    )
    reduced = m.MetricsProcessor._aggregate_metrics(metrics)
    assert reduced["validation/min_policy_version/min"] == step
    assert reduced["validation/max_policy_version/max"] == step + 2
    assert reduced["validation/mixed_policy_rollouts/mean"] == 0.5


def test_validation_rows_share_no_key_with_training_rows() -> None:
    """A validation row logs at its pass's start step, an older step than the training rows
    logged meanwhile. TensorBoard drops a tag's points at or after a step it sees again, so a
    shared key would lose training points. A timed-out rollout counts in `validation/error_rate`."""

    def rollout(*, group_id: int, status: RolloutStatus) -> Rollout:
        turn = RolloutTurn(
            rollout_id=RolloutTurnID(group_id=group_id, rollout_id=0, turn_id=0),
            prompt_prefix_len=0,
            prompt_delta_token_ids=np.array([1], dtype=np.int32),
            completion_token_ids=np.array([2], dtype=np.int32),
            completion_logprobs=np.array([-0.1], dtype=np.float32),
            min_policy_version=0,
            max_policy_version=0,
            # As the generator and the DAPO-Math rubric add them.
            metrics=[
                m.Metric(
                    "validation_generator/queue_time_ms"
                    if group_id < 0
                    else "generator/queue_time_ms",
                    m.Mean(1.0),
                ),
                m.Metric("math_verify/timeouts", m.Sum(1.0)),
            ],
        )
        return Rollout(
            group_id=group_id, rollout_id=0, status=status, reward=0.0, turns=[turn]
        )

    controller = object.__new__(Controller)
    controller._validation_samples = ["prompt"]
    controller.generator_router = SimpleNamespace(
        release_groups=SimpleNamespace(call_one=AsyncMock())
    )
    controller._rollouter = SimpleNamespace(
        run_group_rollouts=AsyncMock(
            return_value=RolloutGroup(
                group_id=-1,
                rollouts=[rollout(group_id=-1, status=RolloutStatus.ERROR_TIMEOUT)],
            )
        )
    )

    _, validation_metrics = asyncio.run(
        controller._collect_validation_rollouts(sampling=SamplingConfig(), step=0)
    )

    validation = m.MetricsProcessor._aggregate_metrics(validation_metrics)
    training = m.MetricsProcessor._aggregate_metrics(
        compute_rollout_metrics(
            prefix="rollout",
            rollouts=[rollout(group_id=3, status=RolloutStatus.COMPLETED)],
        )
    )
    assert validation.keys() & training.keys() == set()
    assert validation["validation/math_verify/timeouts/sum"] == 1.0
    assert validation["validation_generator/queue_time_ms/mean"] == 1.0
    assert validation["validation/error_rate/mean"] == 1.0
    assert training["rollout/error_rate/mean"] == 0.0


class _FakeValidation:
    """Stands in for `Controller.validate`: each pass runs until the test ends it."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []
        self._end_events: dict[int, asyncio.Event] = {}
        self._errors: dict[int, Exception] = {}

    @property
    def started_steps(self) -> list[int]:
        return [step for event, step in self.events if event == "start"]

    async def __call__(self, *, step: int) -> list[m.Metric]:
        self.events.append(("start", step))
        await self._end_event(step).wait()
        self.events.append(("end", step))
        if step in self._errors:
            raise self._errors[step]
        return [m.Metric("validation_reward", m.Mean(float(step)))]

    def end(self, step: int, error: Exception | None = None) -> None:
        if error is not None:
            self._errors[step] = error
        self._end_event(step).set()

    def _end_event(self, step: int) -> asyncio.Event:
        return self._end_events.setdefault(step, asyncio.Event())


def _controller_for_validation(
    validation: ValidationConfig, *, events: list[str] | None = None, on_pull=None
) -> tuple[Controller, _FakeValidation]:
    """`test_trainer_loop`'s fake controller, plus `validation`, a fake `validate` and a `Mock`
    metrics processor that records each `log`."""
    controller = fake_controller([] if events is None else events, on_pull=on_pull)
    controller.config.async_loop.validation = validation
    controller.metrics_processor = Mock()
    fake_validation = _FakeValidation()
    controller.validate = fake_validation
    return controller, fake_validation


def _training_batch(step: int) -> SimpleNamespace:
    """Stands in for the `TrainerStepBatch` that `step` trains on."""
    return SimpleNamespace(
        metrics=[],
        min_policy_versions=[step - 1],
        microbatch_groups=[],
        global_loss_token_counts=[1],
        global_routing_token_counts=None,
        group_ids=[step],
    )


def _logged_steps(controller: Controller, *, is_validation: bool) -> list[int]:
    return [
        call.kwargs["step"]
        for call in controller.metrics_processor.log.call_args_list
        if call.kwargs["is_validation"] == is_validation
    ]


async def _run_until(condition) -> None:
    """Let other tasks run until `condition()` holds."""
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never held")


async def _train(controller: Controller, queue: asyncio.Queue, step: int) -> None:
    """Hand `_trainer_loop` the batch for `step` and wait until it logs that step."""
    await queue.put(_training_batch(step))
    await _run_until(lambda: step in _logged_steps(controller, is_validation=False))


def test_passes_start_at_every_due_step() -> None:
    """Training keeps stepping while passes run, and every due step starts its own pass once that
    step's weight pull ends. A pass that ends before an older one waits for it: passes are logged
    oldest first, each at the step it started at."""
    events: list[str] = []
    pulls_may_end = asyncio.Event()
    controller, validation = _controller_for_validation(
        ValidationConfig(freq=2), events=events, on_pull=pulls_may_end.wait
    )

    async def run() -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        trainer = asyncio.create_task(
            controller._trainer_loop(queue, num_training_steps=6)
        )

        pulls_may_end.set()
        await _train(controller, queue, 1)
        # Step 1's pull has ended.
        await _run_until(lambda: "release 5 trained" in events)
        # Hold step 2's pull: pass 2 must not start before it ends, but step 3 trains meanwhile.
        pulls_may_end.clear()
        await _train(controller, queue, 2)
        await _run_until(lambda: "pull 2" in events)
        await queue.put(_training_batch(3))
        await _run_until(lambda: "fwd_bwd 3" in events)
        assert validation.started_steps == []
        pulls_may_end.set()
        await _run_until(lambda: validation.started_steps == [2])
        await _run_until(lambda: 3 in _logged_steps(controller, is_validation=False))

        # Step 4 starts its own pass while pass 2 runs.
        await _train(controller, queue, 4)
        await _run_until(lambda: validation.started_steps == [2, 4])

        # Pass 4 ends first, so it waits for pass 2.
        validation.end(4)
        await _run_until(lambda: controller._validation_tasks[4].done())
        await _train(controller, queue, 5)
        assert _logged_steps(controller, is_validation=True) == []

        validation.end(2)
        await _run_until(lambda: controller._validation_tasks[2].done())
        await _train(controller, queue, 6)
        await asyncio.wait_for(trainer, timeout=5)
        # The last step starts no pass; it logs both passes, oldest first.
        assert validation.started_steps == [2, 4]
        assert _logged_steps(controller, is_validation=True) == [2, 4]
        assert _logged_steps(controller, is_validation=False) == [1, 2, 3, 4, 5, 6]

    asyncio.run(run())


def test_one_pass_at_a_time_merges_due_steps() -> None:
    """With `one_pass_at_a_time`, due steps that come while a pass runs merge into one pass, which
    starts at the end of the step in which the running pass ends. The last step starts no pass."""
    controller, validation = _controller_for_validation(
        ValidationConfig(freq=1, one_pass_at_a_time=True)
    )

    async def run() -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        trainer = asyncio.create_task(
            controller._trainer_loop(queue, num_training_steps=5)
        )
        await _train(controller, queue, 1)
        await _run_until(lambda: validation.started_steps == [1])
        # Step 2 is due while pass 1 runs.
        await _train(controller, queue, 2)
        validation.end(1)
        await _run_until(lambda: controller._validation_tasks[1].done())
        # Pass 1 is logged in step 3, and one pass starts for steps 2 and 3.
        await _train(controller, queue, 3)
        await _run_until(lambda: validation.started_steps == [1, 3])
        assert _logged_steps(controller, is_validation=True) == [1]

        # Step 4 is due while pass 3 runs; step 5 is the last step.
        await _train(controller, queue, 4)
        validation.end(3)
        await _run_until(lambda: controller._validation_tasks[3].done())
        await _train(controller, queue, 5)
        await asyncio.wait_for(trainer, timeout=5)
        assert validation.started_steps == [1, 3]
        assert _logged_steps(controller, is_validation=True) == [1, 3]

    asyncio.run(run())


def test_validation_failure_fails_training() -> None:
    """A pass that raises fails training at the step it is logged: only pass-level errors (e.g.
    `release_groups`) reach it."""
    controller, validation = _controller_for_validation(ValidationConfig(freq=1))

    async def run() -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        trainer = asyncio.create_task(
            controller._trainer_loop(queue, num_training_steps=3)
        )
        await _train(controller, queue, 1)
        await _run_until(lambda: validation.started_steps == [1])
        validation.end(1, error=RuntimeError("release_groups failed"))
        await _run_until(lambda: controller._validation_tasks[1].done())
        await queue.put(_training_batch(2))
        await asyncio.wait_for(trainer, timeout=5)

    with pytest.raises(RuntimeError, match="release_groups failed"):
        asyncio.run(run())
    assert _logged_steps(controller, is_validation=False) == [1, 2]
    assert _logged_steps(controller, is_validation=True) == []


def _controller_for_run(
    *, watcher=_block_forever
) -> tuple[Controller, _FakeValidation]:
    """A `Controller` whose `run()` runs 4 steps on fakes. Every loop blocks; each test sets
    `_trainer_loop` or the engine-loop `watcher`."""
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=AsyncLoopConfig(num_training_steps=4),
        trainer=SimpleNamespace(
            training=SimpleNamespace(
                num_tokens_per_microbatch_per_dp_rank=64, max_context_length=64
            )
        ),
    )
    controller.start_step = 0
    controller.trainer = Mock()
    controller.generator_router = SimpleNamespace(
        wait_engine_loop=SimpleNamespace(call_one=watcher)
    )
    controller.trainer_dp_degree = 1
    controller.tokenizer = SimpleNamespace(eos_id=0)
    controller._sampling = SamplingConfig()
    controller.metrics_processor = Mock()
    controller._log_reward_delta = Mock()
    controller._validation_tasks = {}
    validation = _FakeValidation()
    controller.validate = validation
    controller._data_input_loop = _block_forever
    controller._rollout_loop = _block_forever
    controller._batcher_loop = _block_forever
    controller._trainer_loop = _block_forever
    return controller, validation


def test_run_fails_when_an_engine_loop_crashes() -> None:
    async def crashed() -> None:
        raise RuntimeError("engine loop crashed")

    controller, _ = _controller_for_run(watcher=crashed)
    with pytest.raises(RuntimeError, match="engine loop crashed"):
        asyncio.run(asyncio.wait_for(controller.run(), timeout=10))


def test_run_fails_when_the_engine_loops_exit_mid_run() -> None:
    async def exited() -> None:
        return None

    controller, _ = _controller_for_run(watcher=exited)
    with pytest.raises(
        RuntimeError, match="generator_wait_engine_loop exited unexpectedly"
    ):
        asyncio.run(asyncio.wait_for(controller.run(), timeout=10))


@pytest.mark.parametrize(
    "start_step, open_steps",
    [
        # Passes 0 and 3 still run when training ends.
        (0, [0, 3]),
        # Resumed at the last step: the first pass has the final pass's step, so its group ids.
        (4, [4]),
    ],
)
def test_run_waits_for_open_passes_then_runs_the_final_pass(
    start_step, open_steps
) -> None:
    """The first pass runs while training starts. Passes still running when training ends are
    awaited and logged at their start steps; only then does the final pass start."""
    controller, validation = _controller_for_run()
    controller.start_step = start_step

    # Stands in for `_trainer_loop`, which the tests above cover.
    async def trainer_loop(queue, *, num_training_steps: int) -> None:
        assert validation.started_steps == [start_step]
        for step in open_steps[1:]:
            controller._start_validation(step=step)
        await _run_until(lambda: validation.started_steps == open_steps)
        # Training ends before the open passes do; the final pass's end is set last.
        for delay, step in enumerate([*open_steps, 4], start=1):
            asyncio.get_running_loop().call_later(0.01 * delay, validation.end, step)

    controller._trainer_loop = trainer_loop

    asyncio.run(asyncio.wait_for(controller.run(), timeout=5))

    assert validation.events == [
        *[("start", step) for step in open_steps],
        *[("end", step) for step in open_steps],
        ("start", 4),
        ("end", 4),
    ]
    assert _logged_steps(controller, is_validation=True) == [*open_steps, 4]
    first_metrics, final_metrics = controller._log_reward_delta.call_args.args
    assert m.MetricsProcessor._aggregate_metrics(first_metrics) == {
        "validation_reward/mean": float(start_step)
    }
    assert m.MetricsProcessor._aggregate_metrics(final_metrics) == {
        "validation_reward/mean": 4.0
    }


def test_close_cancels_running_passes_before_closing_the_rollouter() -> None:
    """After a training crash, `close()` cancels the open passes and waits for them, then closes
    the rollouter they were using."""
    controller = object.__new__(Controller)
    controller.trainer = None
    controller.generator_router = None
    controller.metrics_processor = Mock()
    controller.actor_system = None
    controller._validation_tasks = {}
    validation = _FakeValidation()
    controller.validate = validation
    passes_done_at_rollouter_close: list[bool] = []

    async def close_rollouter() -> None:
        passes_done_at_rollouter_close.append(all(task.done() for task in passes))

    controller._rollouter = SimpleNamespace(close=close_rollouter)

    async def run() -> None:
        controller._start_validation(step=0)
        controller._start_validation(step=2)
        await _run_until(lambda: validation.started_steps == [0, 2])
        passes.extend(controller._validation_tasks.values())
        await asyncio.wait_for(controller.close(), timeout=5)

    passes: list[asyncio.Task] = []
    asyncio.run(run())
    assert all(task.cancelled() for task in passes)
    assert passes_done_at_rollouter_close == [True]
