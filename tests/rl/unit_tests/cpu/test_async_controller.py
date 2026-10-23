# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for async-controller pieces: batcher group-counting, the active-slot buffer backpressure,
the consume-time staleness invariant, the metrics timer drain, and RolloutTurnID."""

import asyncio
import gc
import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import torch
from torchtitan.rl.controller.adaptive_demand import (
    estimate_next_unavailable_upper_bound,
    mean_age_ceiling,
    prediction_multiplier,
    smooth_demand_toward_needed,
    StallDrivenDemand,
)
from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.rl.controller.metrics import (
    compute_perf_ratio_metrics,
    compute_policy_age_metrics,
    MetricsTimer,
)
from torchtitan.rl.controller.work_buffer import (
    AdaptiveRolloutGroupWorkBuffer,
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

from tests.rl.unit_tests.cpu.test_trainer_loop import _controller


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
    assert saved_config["trainer"]["training"]["max_context_length"] == 128


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


def test_compute_perf_ratio_metrics_reads_flushed_means() -> None:
    time_metrics = [
        m.Metric("timing/step/total", m.Mean.from_list([20.0])),
        m.Metric("timing/step/metrics_log", m.Mean.from_list([1.0])),
        m.Metric("timing/step/sync_log_step", m.Mean.from_list([1.0])),
        m.Metric("timing/step/wait_for_training_batch", m.Mean.from_list([4.0])),
        m.Metric("timing/step/forward_backward", m.Mean.from_list([8.0])),
        m.Metric(
            "timing/step/blocking_trainer_push_model_state_dict",
            m.Mean.from_list([1.0]),
        ),
        m.Metric("timing/step/optimizer", m.Mean.from_list([1.0])),
        m.Metric("timing/step/save_checkpoint", m.Mean.from_list([2.0])),
        m.Metric(
            "timing/step/blocking_generator_pull_model_state_dict",
            m.Mean.from_list([1.0]),
        ),
    ]
    ratios = {
        metric.key: metric.value.value
        for metric in compute_perf_ratio_metrics(
            num_global_valid_tokens=100,
            num_global_tokens=400,
            time_metrics=time_metrics,
        )
    }
    assert ratios == pytest.approx(
        {
            "perf/trainer/tokens_per_second_full_step": 5.0,
            "perf/trainer/tokens_per_second_forward_backward": 12.5,
            "perf/trainer/all_tokens_per_second_full_step": 20.0,
            "perf/trainer/all_tokens_per_second_forward_backward": 50.0,
            # One ratio per timing/step/<phase>; unaccounted is the rest of the step.
            "perf/trainer/step_time_ratio/metrics_log": 0.05,
            "perf/trainer/step_time_ratio/sync_log_step": 0.05,
            "perf/trainer/step_time_ratio/wait_for_training_batch": 0.2,
            "perf/trainer/step_time_ratio/forward_backward": 0.4,
            "perf/trainer/step_time_ratio/blocking_trainer_push_model_state_dict": 0.05,
            "perf/trainer/step_time_ratio/optimizer": 0.05,
            "perf/trainer/step_time_ratio/save_checkpoint": 0.1,
            "perf/trainer/step_time_ratio/blocking_generator_pull_model_state_dict": 0.05,
            "perf/trainer/step_time_ratio/unaccounted": 0.05,
        }
    )


def test_compute_perf_ratio_metrics_skips_missing_spans() -> None:
    # Only `total` recorded -> emit the full-step throughput, skip every ratio whose span is absent.
    time_metrics = [m.Metric("timing/step/total", m.Mean.from_list([2.0]))]
    keys = {
        metric.key
        for metric in compute_perf_ratio_metrics(
            num_global_valid_tokens=100,
            num_global_tokens=400,
            time_metrics=time_metrics,
        )
    }
    assert keys == {
        "perf/trainer/tokens_per_second_full_step",
        "perf/trainer/all_tokens_per_second_full_step",
    }


def test_compute_perf_ratio_metrics_returns_empty_without_total() -> None:
    assert (
        compute_perf_ratio_metrics(
            num_global_valid_tokens=100, num_global_tokens=400, time_metrics=[]
        )
        == []
    )


def test_gc_timer_times_collections_per_generation() -> None:
    gc_timer = m.GCTimer("controller", per_generation=True)
    gc.disable()  # only the collections below run
    try:
        gc.collect(0)
        cycle = []
        cycle.append(cycle)  # garbage only a collection can free
        del cycle
        gc.collect(2)
        values = {
            (metric.key, type(metric.value)): metric.value.value
            for metric in gc_timer.flush()
        }
    finally:
        gc_timer.close()
        gc.enable()
    collections = [
        values[(f"controller/gc/gen{g}/collections", m.Sum)] for g in range(3)
    ]
    seconds = [values[(f"controller/gc/gen{g}/seconds", m.Sum)] for g in range(3)]
    max_s = [values[(f"controller/gc/gen{g}/seconds", m.Max)] for g in range(3)]
    assert collections == [1, 0, 1]
    assert values[("controller/gc/gen2/collected", m.Sum)] >= 1
    assert seconds[1] == max_s[1] == 0.0
    assert values[("perf/controller/gc_seconds", m.Sum)] == sum(seconds)
    assert values[("perf/controller/gc_seconds", m.Max)] == max(max_s) > 0.0
    gc.collect(0)  # after close(): not timed, and the flush above reset the totals
    assert all(metric.value.value == 0.0 for metric in gc_timer.flush())
    gc_timer.close()  # a second close does nothing


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
        # S=1, P=4 -> 8 slots, no window. g0..g4 INFLIGHT; g0 never finishes; g1..g4 finish out of id order.
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
        await buffer.claim_next()  # g5 -> INFLIGHT
        await buffer.claim_next()  # g6 -> INFLIGHT
        await _finalize(buffer, 6)
        await asyncio.sleep(0)
        assert taker.done()
        assert taker.result().group_id == 6

    asyncio.run(run())


def test_buffer_gauges_count_every_active_slot() -> None:
    async def run() -> None:
        # 4 slots: g0 taken by the batcher, g1 finalized, g2 inflight, g3 waiting.
        buffer = _buffer(capacity=4, window_size=None)
        for group_id in range(4):
            await _admit(buffer, group_id)
        for _ in range(3):
            await buffer.claim_next()  # g0, g1, g2 -> INFLIGHT
        await _finalize(buffer, 0)
        await _finalize(buffer, 1)
        await buffer.take_finalized()  # g0 leaves the buffer but keeps its slot

        def gauges() -> dict[str, float]:
            return {
                metric.key: metric.value.value
                for metric in buffer.metrics()
                if metric.key != "rollout_buffer/active_slots_in_use_peak"
            }

        assert gauges() == {
            "rollout_buffer/num_groups_waiting": 1.0,
            "rollout_buffer/num_groups_inflight": 1.0,
            "rollout_buffer/num_groups_finalized": 1.0,
            "rollout_buffer/num_groups_taken": 1.0,
            "rollout_buffer/available_active_slots": 0.0,
        }
        # The weight pull after g0's optimizer step frees its slot.
        await buffer.release_active_groups(1, reason="trained")
        assert gauges()["rollout_buffer/num_groups_taken"] == 0.0
        assert gauges()["rollout_buffer/available_active_slots"] == 1.0

    asyncio.run(run())


async def _block_forever(*args, **kwargs) -> None:
    await asyncio.Event().wait()


def _run_with_generator_watcher(watcher) -> None:
    """Run Controller.run with every stage but the engine-loop watcher stubbed to block."""
    controller = object.__new__(Controller)
    controller._extra_validation_tasks = {}
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(
            num_training_steps=1,
            max_active_rollout_groups=1,
            window_size=None,
            target_offpolicy_steps=0,
            windowed_fifo_batches=None,
            max_offpolicy_steps=None,
            num_prompts_per_train_step=1,
            group_buffer=SimpleNamespace(
                build=lambda **kwargs: SimpleNamespace(close=AsyncMock())
            ),
            validation=ValidationConfig(),
            extra_validations=[],
        ),
        trainer=SimpleNamespace(
            training=SimpleNamespace(
                num_tokens_per_microbatch_per_dp_rank=8, max_context_length=8
            )
        ),
    )
    controller.start_step = 0
    controller.trainer = None
    controller.trainer_dp_degree = 1
    controller.tokenizer = SimpleNamespace(eos_id=0)
    controller._sampling = SimpleNamespace(temperature=1.0)
    controller.generator_router = SimpleNamespace(
        wait_engine_loop=SimpleNamespace(call_one=watcher)
    )
    with (
        patch.multiple(
            Controller,
            _validate_and_log=AsyncMock(return_value={}),
            _data_input_loop=_block_forever,
            _rollout_loop=_block_forever,
            _batcher_loop=_block_forever,
            _trainer_loop=_block_forever,
        ),
    ):
        asyncio.run(asyncio.wait_for(controller.run(), timeout=10))


def test_run_fails_when_an_engine_loop_crashes() -> None:
    async def crashed() -> None:
        raise RuntimeError("engine loop crashed")

    with pytest.raises(RuntimeError, match="engine loop crashed"):
        _run_with_generator_watcher(crashed)


def test_run_fails_when_the_engine_loops_exit_mid_run() -> None:
    async def exited() -> None:
        return None

    with pytest.raises(
        RuntimeError, match="generator_wait_engine_loop exited unexpectedly"
    ):
        _run_with_generator_watcher(exited)


def test_cancelled_rollout_loop_releases_its_group() -> None:
    """A cancel (shutdown) still releases the group: post-training validation reuses the generators."""

    async def run() -> None:
        started = asyncio.Event()
        events = []

        async def run_group_rollouts(**kwargs) -> RolloutGroup:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                events.append("group cancelled")
                raise

        async def release_groups(group_ids) -> None:
            events.append(("release_groups", group_ids))

        controller = object.__new__(Controller)
        controller.config = SimpleNamespace(
            async_loop=SimpleNamespace(num_samples_per_prompt=2, max_stragglers=None)
        )
        controller._sampling = None
        controller._rollouter = SimpleNamespace(run_group_rollouts=run_group_rollouts)
        controller.generator_router = SimpleNamespace(
            release_groups=SimpleNamespace(call_one=release_groups)
        )
        group_buffer = SimpleNamespace(
            claim_next=AsyncMock(
                return_value=RolloutGroupWork(group_id=3, sample=object())
            ),
            track_generation=lambda work, task: None,
            finalize_work=AsyncMock(),
            release_cancelled=AsyncMock(),
        )
        loop = asyncio.create_task(
            controller._rollout_loop(
                group_buffer=group_buffer, generate_fn=None, release_session_fn=None
            )
        )
        await started.wait()
        loop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop
        assert events == ["group cancelled", ("release_groups", [3])]
        group_buffer.finalize_work.assert_not_awaited()
        group_buffer.release_cancelled.assert_not_awaited()

    asyncio.run(run())


def test_fixed_buffer_ignores_the_adaptive_hooks() -> None:
    """The default buffer takes and meters the same with the adaptive buffer's hooks called."""
    config = AsyncLoopConfig(
        num_prompts_per_train_step=4, target_offpolicy_steps=1, windowed_fifo_batches=1
    )
    assert type(config.group_buffer) is RolloutGroupWorkBuffer.Config
    assert config.max_concurrent_rollout_groups == config.max_active_rollout_groups == 8
    assert config.max_offpolicy_steps == 2

    async def run(*, with_hooks: bool) -> list:
        buffer = _buffer(capacity=8, window_size=4)
        for group_id in range(8):
            await _admit(buffer, group_id)
        for _ in range(8):
            await buffer.claim_next()
        for group_id in (3, 1, 2, 5, 0, 7):
            await _finalize(buffer, group_id)
        log = []
        for version in range(2):
            if with_hooks:
                await buffer.record_step_start(trainer_policy_version=version)
            for _ in range(3):
                kwargs = {"consuming_policy_version": version} if with_hooks else {}
                log.append((await buffer.take_finalized(**kwargs)).group_id)
            if with_hooks:
                assert buffer.pop_dropped_group_ids() == []
            await buffer.release_active_groups(3, reason="trained")
            log.append({metric.key: metric.value.value for metric in buffer.metrics()})
        return log

    log = asyncio.run(run(with_hooks=True))
    assert log == asyncio.run(run(with_hooks=False))
    assert [entry for entry in log if isinstance(entry, int)] == [0, 1, 2, 3, 5, 7]


def _adaptive_buffer(
    *,
    num_prompts: int,
    max_offpolicy_steps: int | None = 4,
    generation_capacity: int = 64,
    **kwargs,
) -> AdaptiveRolloutGroupWorkBuffer:
    return AdaptiveRolloutGroupWorkBuffer.Config(
        max_offpolicy_steps=max_offpolicy_steps,
        generation_capacity=generation_capacity,
    ).build(num_prompts_per_train_step=num_prompts, **kwargs)


def _buffer_metric(buffer: RolloutGroupWorkBuffer, key: str) -> float:
    return next(metric.value.value for metric in buffer.metrics() if metric.key == key)


def test_stall_driven_demand_quantile_uses_every_point_of_the_lookback() -> None:
    history = [44, 46, 41, 48, 45, 47, 50, 43, 46, 45]
    # mean 45.5, sd 2.55, Student-t prediction multiplier for 10 samples at 95% = 1.92 -> 50.4 -> 51 (above the max seen)
    assert (
        estimate_next_unavailable_upper_bound(history=history, probability=0.95) == 51
    )
    assert (
        estimate_next_unavailable_upper_bound(history=[45, 45, 45], probability=0.95)
        == 45
    )
    assert estimate_next_unavailable_upper_bound(history=[24], probability=0.95) == 24
    assert round(prediction_multiplier(samples=10, probability=0.95), 3) == 1.923
    assert round(prediction_multiplier(samples=1000, probability=0.95), 2) == 1.65
    assert mean_age_ceiling(
        num_prompts_per_train_step=8,
        mean_age_limit=4,
        groups_generating=40,
        untrainable_share=0.36,
    ) == pytest.approx(54.4)
    # half the gap, the same up and down, rounded away from the current value
    assert (
        smooth_demand_toward_needed(
            current_demand=64, demand_needed=59, damping_factor=0.5
        )
        == 61
    )
    assert (
        smooth_demand_toward_needed(
            current_demand=64, demand_needed=73, damping_factor=0.5
        )
        == 69
    )
    assert (
        smooth_demand_toward_needed(
            current_demand=64, demand_needed=65, damping_factor=0.5
        )
        == 65
    )


def test_stall_driven_demand_starts_at_three_batches_and_moves_half_the_gap() -> None:
    demand = StallDrivenDemand(num_prompts_per_train_step=8, max_offpolicy_steps=10)
    assert demand.demand == 24
    # nothing ready: unavailable 24, one sample -> quantile 24, needed 8 + 24 + 1 = 33, half the gap up: 24 + 5
    assert (
        demand.observe(step=1, ready=0, generating=24, completed=0, trainable=0) == 29
    )
    assert demand.state == "ok"
    # a full shelf: unavailable max(0, 29 - 40) = 0; history [24, 0] -> mean 12, sd 17, two samples give a wide
    # margin (t = 7.5): quantile 128 -> needed 137. The ceiling at the max caps it: 10 x 8 + 8 + mean(24, 0) x
    # (1 - 40 / 40) = 88. Half the gap up from 29: 29 + 30
    assert (
        demand.observe(step=2, ready=40, generating=0, completed=40, trainable=40) == 59
    )
    assert demand.state == "age-limited"


def test_stall_driven_demand_moves_down_one_group_per_step() -> None:
    for damping_factor in (0.5, 1.0):
        demand = StallDrivenDemand(
            num_prompts_per_train_step=8, start_batches=8, damping_factor=damping_factor
        )
        # a full shelf: unavailable 0 -> needed 8 + 0 + 1 = 9; one group down from 64, whatever the damping
        assert (
            demand.observe(step=1, ready=64, generating=0, completed=64, trainable=64)
            == 63
        )
        assert demand.state == "ok"


def test_stall_driven_demand_above_the_ceiling_moves_down_half_the_gap() -> None:
    # ceiling with a max of 1: 1 x 8 + 8 + generating x untrainable share = 16 + 0 in both cases below
    # nothing ready: needed 8 + 64 + 1 = 73, capped at 16; half the gap down: 64 - 24
    demand = StallDrivenDemand(
        num_prompts_per_train_step=8, max_offpolicy_steps=1, start_batches=8
    )
    assert demand.observe(step=1, ready=0, generating=0, completed=0, trainable=0) == 40
    assert demand.state == "age-limited"
    # a full shelf: needed 8 + 4 + 1 = 13 is under the ceiling, but demand 64 is above it; half the gap down too
    demand = StallDrivenDemand(
        num_prompts_per_train_step=8, max_offpolicy_steps=1, start_batches=8
    )
    assert (
        demand.observe(step=1, ready=60, generating=4, completed=60, trainable=60) == 40
    )
    assert demand.state == "age-limited"


def test_stall_driven_demand_is_capped_by_the_mean_age_ceiling() -> None:
    # target 4 with 40 generating and 36% rejected: ceiling 4 x 8 + 8 + 40 x 0.36 = 54.4 -> 54
    demand = StallDrivenDemand(
        num_prompts_per_train_step=8, max_offpolicy_steps=10, target_offpolicy_steps=4
    )
    path = [
        demand.observe(step=step, ready=0, generating=40, completed=25, trainable=16)
        for step in range(1, 9)
    ]
    assert path == [29, 42, 48, 51, 53, 54, 54, 54]
    assert demand.state == "age-limited"
    # no target: the same ceiling is applied at max_offpolicy_steps
    demand = StallDrivenDemand(num_prompts_per_train_step=8, max_offpolicy_steps=4)
    path = [
        demand.observe(step=step, ready=0, generating=40, completed=25, trainable=16)
        for step in range(1, 9)
    ]
    assert path == [29, 42, 48, 51, 53, 54, 54, 54]
    assert demand.state == "age-limited"
    # a ceiling that stops binding is reported: once the lookback holds only step starts with a full shelf
    # (unavailable 0), the need falls to P + 0 + 1 = 9 and the state returns to "ok"
    for step in range(9, 21):
        demand.observe(step=step, ready=100, generating=0, completed=8, trainable=8)
    assert demand.state == "ok"


def test_stall_driven_demand_without_a_max_has_no_ceiling_and_drops_nothing() -> None:
    demand = StallDrivenDemand(num_prompts_per_train_step=2, max_offpolicy_steps=None)
    for step in range(1, 11):
        demand.observe(step=step, ready=0, generating=6, completed=0, trainable=0)
    # with a max of 1 the ceiling would have held it at 8
    assert demand.demand > 8 and demand.state == "ok"

    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=2, max_offpolicy_steps=None, generation_capacity=4
        )
        await _admit(buffer, 0)
        await buffer.claim_next()  # g0 generates under version 0
        await _finalize(buffer, 0)
        # 50 versions old: kept
        selected = await buffer.take_finalized(consuming_policy_version=50)
        assert selected is not None and selected.group_id == 0
        assert _buffer_metric(buffer, "rollout_buffer/dropped_too_old") == 0

    asyncio.run(run())
    with pytest.raises(ValueError, match="target_offpolicy_steps"):
        AdaptiveRolloutGroupWorkBuffer.Config(
            max_offpolicy_steps=None, generation_capacity=4, target_offpolicy_steps=0
        )
    # a target without a max is allowed: the ceiling holds the mean age, nothing is dropped
    config = AsyncLoopConfig(
        group_buffer=AdaptiveRolloutGroupWorkBuffer.Config(
            max_offpolicy_steps=None, generation_capacity=4, target_offpolicy_steps=3
        )
    )
    assert config.max_offpolicy_steps is None
    assert config.max_concurrent_rollout_groups == 4


def test_adaptive_buffer_takes_oldest_finalized_and_lets_slow_groups_keep_their_slot() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(num_prompts=2, generation_capacity=4)
        for group_id in range(4):
            await _admit(buffer, group_id)
        for _ in range(4):
            await buffer.claim_next()
        await _finalize(buffer, 2)
        await _finalize(buffer, 1)

        # g0 is still INFLIGHT: the batcher gets g1 then g2 without waiting for it
        assert (await buffer.take_finalized()).group_id == 1
        assert (await buffer.take_finalized()).group_id == 2
        taker = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0)
        assert not taker.done()

        await _finalize(buffer, 0)
        assert (await taker).group_id == 0
        # taking never frees a slot: 4 admitted -> 4 still active
        assert _buffer_metric(buffer, "rollout_buffer/active_slots_in_use_peak") == 4

    asyncio.run(run())


def test_adaptive_buffer_drops_groups_past_max_offpolicy_steps() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=2, max_offpolicy_steps=1, generation_capacity=4
        )
        for group_id in range(4):
            await _admit(buffer, group_id)
        for _ in range(4):
            await buffer.claim_next()  # g0..g3 generate under version 0

        # Step 1 trains g2 and g3 at version 0; the pull of version 1 releases their slots
        for group_id in (2, 3):
            await _finalize(buffer, group_id)
            await buffer.take_finalized(consuming_policy_version=0)
        await buffer.release_active_groups(2, reason="trained")
        await _admit(buffer, 4)
        await buffer.claim_next()  # g4 generates under version 1

        # The trainer starts step 2 (version 1); g0 and g1 finish
        await buffer.record_step_start(trainer_policy_version=1)
        await _finalize(buffer, 1)
        await _finalize(buffer, 0)
        # Finalization alone cannot know whether the batcher is assembling the
        # current or prefetched batch, so it must not guess the consume version.
        assert _buffer_metric(buffer, "rollout_buffer/num_groups_finalized") == 2

        # The batch being assembled trains at version 2: g0 and g1 would be 2 old -> dropped; g4 is 1 old
        await _finalize(buffer, 4)
        selected = await buffer.take_finalized(consuming_policy_version=2)
        assert selected is not None
        assert selected.group_id == 4
        metrics = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert metrics["rollout_buffer/dropped_too_old"] == 2
        # The controller acknowledges dropped ids once, so the dataloader does not replay them
        assert buffer.pop_dropped_group_ids() == [0, 1]
        assert buffer.pop_dropped_group_ids() == []
        # Dropped slots are free at once: 5 admitted, 2 trained, 2 dropped -> 1 active
        assert metrics["rollout_buffer/available_active_slots"] == (
            metrics["rollout_buffer/demand_target_groups"] - 1
        )

    asyncio.run(run())


def test_adaptive_buffer_cancels_generating_groups_past_max_offpolicy_steps() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=1, max_offpolicy_steps=1, generation_capacity=4
        )
        for group_id in range(3):
            await _admit(buffer, group_id)
        works = [await buffer.claim_next() for _ in range(3)]  # g0..g2 under version 0
        hung = asyncio.create_task(asyncio.sleep(3600))
        buffer.track_generation(works[0], hung)

        # Step 1 trains g2 at version 0; the pull of version 1 releases its slot
        await _finalize(buffer, 2)
        assert (await buffer.take_finalized(consuming_policy_version=0)).group_id == 2
        await buffer.release_active_groups(1, reason="trained")

        # The batch for version 1 makes g0/g1 1 old: still trainable, nothing cancelled
        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=1))
        await asyncio.sleep(0)
        assert not hung.cancelled()
        await _finalize(buffer, 1)
        assert (await taker).group_id == 1

        # The batch for version 2 makes g0 2 old: it can never train, so it is cancelled mid-generation
        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=2))
        with pytest.raises(asyncio.CancelledError):
            await hung
        metrics = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert metrics["rollout_buffer/cancelled_too_old"] == 1
        assert metrics["rollout_buffer/dropped_too_old"] == 0
        # Counted as dropped at once: the dataloader acknowledges it at the next optimizer step
        assert buffer.pop_dropped_group_ids() == [0]
        # Its slot stays charged (still generating) until its task ended and the rollout loop releases it
        assert metrics["rollout_buffer/num_groups_inflight"] == 1
        available = metrics["rollout_buffer/available_active_slots"]
        await buffer.release_cancelled(0)
        assert _buffer_metric(buffer, "rollout_buffer/num_groups_inflight") == 0
        assert (
            _buffer_metric(buffer, "rollout_buffer/available_active_slots")
            == available + 1
        )
        await buffer.close()
        assert await taker is None

    asyncio.run(run())


def test_adaptive_buffer_cancel_counts_like_a_finalized_too_old_drop() -> None:
    """A group cancelled while generating moves the demand counters exactly as the same group would if it
    finished and were dropped too old, so the demand ceiling's untrainable share does not change."""

    async def counters(*, finish_first: bool) -> tuple:
        buffer = _adaptive_buffer(
            num_prompts=1, max_offpolicy_steps=1, generation_capacity=4
        )
        for group_id in range(2):
            await _admit(buffer, group_id)
        work0, _ = [await buffer.claim_next() for _ in range(2)]  # under version 0
        buffer.track_generation(work0, asyncio.create_task(asyncio.sleep(3600)))
        await _finalize(buffer, 1)
        if finish_first:
            await _finalize(buffer, 0)
        # The batch for version 2: g0 is 2 old (dropped or cancelled); g1 is too, so nothing is returned
        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=2))
        await asyncio.sleep(0)
        if not finish_first:
            await asyncio.sleep(0)  # let the cancelled task finish
            await buffer.release_cancelled(0)
        result = (
            buffer._completed_since_step_start,
            buffer._dropped_since_step_start,
            buffer.pop_dropped_group_ids(),
            _buffer_metric(buffer, "rollout_buffer/available_active_slots"),
        )
        await buffer.close()
        assert await taker is None
        return result

    async def run() -> None:
        dropped = await counters(finish_first=True)
        cancelled = await counters(finish_first=False)
        # completed 2, dropped 2 (g0 and g1 both too old), the same acknowledged ids and free slots
        assert dropped == cancelled
        assert sorted(dropped[2]) == [0, 1]

    asyncio.run(run())


def test_adaptive_buffer_cancel_racing_the_last_rollout_releases_the_slot_once() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=1, max_offpolicy_steps=1, generation_capacity=4
        )
        await _admit(buffer, 0)
        await _admit(buffer, 1)
        work0 = await buffer.claim_next()  # g0 under version 0
        # g0's generation already ended, but its result is not stored yet: the cancel is a no-op
        finished = asyncio.create_task(asyncio.sleep(0))
        await finished
        buffer.track_generation(work0, finished)
        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=2))
        await asyncio.sleep(0)
        assert _buffer_metric(buffer, "rollout_buffer/cancelled_too_old") == 1
        available = _buffer_metric(buffer, "rollout_buffer/available_active_slots")

        # The late result is dropped and frees the slot, exactly once
        assert not await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[]))
        assert (
            _buffer_metric(buffer, "rollout_buffer/available_active_slots")
            == available + 1
        )
        await buffer.release_cancelled(0)  # entry already gone: no second release
        assert (
            _buffer_metric(buffer, "rollout_buffer/available_active_slots")
            == available + 1
        )
        # g1 was never claimed (WAITING): never cancelled
        assert _buffer_metric(buffer, "rollout_buffer/num_groups_waiting") == 1
        assert buffer.pop_dropped_group_ids() == [0]
        await buffer.close()
        assert await taker is None

    asyncio.run(run())


def test_rollout_worker_propagates_a_cancel_the_buffer_did_not_make() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=1, max_offpolicy_steps=1, generation_capacity=4
        )

        released = []

        async def run_group_rollouts(*, group_id: int, **kwargs) -> RolloutGroup:
            raise asyncio.CancelledError  # e.g. an awaited call cancelled from elsewhere

        async def release_groups(group_ids: list[int]) -> None:
            released.append(group_ids)

        controller = object.__new__(Controller)
        controller.config = SimpleNamespace(
            async_loop=SimpleNamespace(num_samples_per_prompt=2, max_stragglers=None)
        )
        controller._sampling = None
        controller.generator_router = SimpleNamespace(
            release_groups=SimpleNamespace(call_one=release_groups)
        )
        controller.rollout_recorder = SimpleNamespace(record=lambda **kwargs: None)
        controller._rollouter = SimpleNamespace(
            run_group_rollouts=run_group_rollouts,
            record_group_outcome=lambda group: None,
        )
        worker = asyncio.create_task(
            controller._rollout_loop(
                group_buffer=buffer, generate_fn=None, release_session_fn=None
            )
        )
        await _admit(buffer, 0)
        # The worker exits with the CancelledError instead of treating it as a buffer cancel.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(worker, timeout=5)
        assert released == [[0]]  # the group's sessions are still released

    asyncio.run(run())


def test_rollout_worker_survives_a_group_the_buffer_cancels() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=1, max_offpolicy_steps=1, generation_capacity=4
        )
        events = []
        cancel_drained = asyncio.Event()

        async def run_group_rollouts(*, group_id: int, **kwargs) -> RolloutGroup:
            if group_id == 0:  # a slow episode
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    # The real rollouter waits here for the worker's cancel_group.
                    events.append(("cancelled", group_id))
                    await cancel_drained.wait()
                    raise
            return RolloutGroup(group_id=group_id, rollouts=[])

        async def release_groups(group_ids: list[int]) -> None:
            events.append(("release_groups", group_ids))

        controller = object.__new__(Controller)
        controller.config = SimpleNamespace(
            async_loop=SimpleNamespace(num_samples_per_prompt=2, max_stragglers=None)
        )
        controller._sampling = None
        controller.generator_router = SimpleNamespace(
            release_groups=SimpleNamespace(call_one=release_groups)
        )
        controller.rollout_recorder = SimpleNamespace(record=lambda **kwargs: None)
        controller._rollouter = SimpleNamespace(
            run_group_rollouts=run_group_rollouts,
            record_group_outcome=lambda group: None,
        )
        workers = [
            asyncio.create_task(
                controller._rollout_loop(
                    group_buffer=buffer, generate_fn=None, release_session_fn=None
                )
            )
            for _ in range(2)
        ]
        await _admit(buffer, 0)
        await _admit(buffer, 1)
        # g1 finishes and trains at version 0; g0 is slow
        assert (await buffer.take_finalized(consuming_policy_version=0)).group_id == 1
        await buffer.release_active_groups(1, reason="trained")

        # The batch for version 2 cancels g0; its slot is held until its cancel drained
        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=2))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert ("cancelled", 0) in events
        assert _buffer_metric(buffer, "rollout_buffer/num_groups_inflight") == 1
        cancel_drained.set()
        for _ in range(5):
            await asyncio.sleep(0)
        assert _buffer_metric(buffer, "rollout_buffer/num_groups_inflight") == 0
        assert ("release_groups", [0]) in events

        # Both workers keep serving: g2 is generated and taken
        await _admit(buffer, 2)
        assert (await taker).group_id == 2
        assert not any(worker.done() for worker in workers)

        await buffer.close()
        await asyncio.gather(*workers)

    asyncio.run(run())


def test_adaptive_buffer_measures_the_shelf_and_moves_demand() -> None:
    async def run() -> None:
        # P=4: demand starts at three batches = 12; the shelf is what is finished and not held by the trainer.
        buffer = _adaptive_buffer(num_prompts=4, generation_capacity=40)
        assert _buffer_metric(buffer, "rollout_buffer/demand_target_groups") == 12
        for group_id in range(12):
            await _admit(buffer, group_id)
        for _ in range(12):
            await buffer.claim_next()

        # nothing finished: unavailable 12 -> needed 4 + 12 + 1 = 17 -> half the gap up: 12 + 3
        await buffer.record_step_start(trainer_policy_version=0)
        metrics = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert metrics["rollout_buffer/demand_target_groups"] == 15
        assert metrics["rollout_buffer/ready_at_step_start"] == 0
        assert metrics["rollout_buffer/generating_at_step_start"] == 12
        assert metrics["rollout_buffer/unavailable_at_step_start"] == 12
        assert metrics["rollout_buffer/demand_age_limited"] == 0

        # ten groups finish; the batcher selects four and the trainer trains them at version 0. The pull
        # of version 1 has not released them yet, so they leave the shelf
        for group_id in range(10):
            await _finalize(buffer, group_id)
        for _ in range(4):
            await buffer.take_finalized(consuming_policy_version=0)
        await buffer.record_step_start(trainer_policy_version=1)
        metrics = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert metrics["rollout_buffer/ready_at_step_start"] == 6
        assert metrics["rollout_buffer/generating_at_step_start"] == 2
        assert metrics["rollout_buffer/unavailable_at_step_start"] == 9
        # unavailable 15 - 6 = 9; history [12, 9] -> mean 10.5 + 4.46 x 2.12 -> 20 -> needed 25; ceiling
        # 4 x 4 + 4 + mean(12, 2) x (1 - 10 / 10) = 20 -> capped at 20 -> half the gap: 15 + 3
        assert metrics["rollout_buffer/demand_target_groups"] == 18
        assert metrics["rollout_buffer/demand_age_limited"] == 1

    asyncio.run(run())


def test_adaptive_buffer_target_offpolicy_steps_caps_demand() -> None:
    # P=4, target 5 of max 10: the ceiling is 5 x 4 + 4 + generating x untrainable share
    buffer = AdaptiveRolloutGroupWorkBuffer.Config(
        max_offpolicy_steps=10, generation_capacity=40, target_offpolicy_steps=5
    ).build(num_prompts_per_train_step=4)
    assert _buffer_metric(buffer, "rollout_buffer/demand_target_groups") == 12

    async def run() -> None:
        for _ in range(12):
            # nothing generating, nothing completed
            await buffer.record_step_start(trainer_policy_version=0)
        # nothing generating and no completions: ceiling 20 + 4 + 0 = 24, however short the shelf
        assert _buffer_metric(buffer, "rollout_buffer/demand_target_groups") == 24
        assert _buffer_metric(buffer, "rollout_buffer/demand_age_limited") == 1

    asyncio.run(run())
    with pytest.raises(ValueError, match="target_offpolicy_steps"):
        AdaptiveRolloutGroupWorkBuffer.Config(
            generation_capacity=40, target_offpolicy_steps=0
        )
    with pytest.raises(ValueError, match="target_offpolicy_steps"):
        AdaptiveRolloutGroupWorkBuffer.Config(
            max_offpolicy_steps=4, generation_capacity=40, target_offpolicy_steps=5
        )


def test_adaptive_buffer_demand_stays_under_the_mean_age_ceiling() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=2, max_offpolicy_steps=1, generation_capacity=4
        )
        for group_id in range(4):
            await _admit(buffer, group_id)
        for _ in range(4):
            await buffer.claim_next()
        for _ in range(10):
            await buffer.record_step_start(trainer_policy_version=0)
        # nothing has completed, so every generating group counts as untrainable: ceiling 1 x 2 + 2 + 4 x 1.0 = 8
        assert _buffer_metric(buffer, "rollout_buffer/demand_target_groups") == 8
        # demand has room, generation does not
        waiter = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not waiter.done()
        await buffer.close()
        assert await waiter is False

    asyncio.run(run())


def test_adaptive_buffer_does_not_queue_ahead_of_generation() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(num_prompts=2, generation_capacity=2)
        await _admit(buffer, 0)
        await _admit(buffer, 1)
        await buffer.claim_next()
        await buffer.claim_next()

        third_slot = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not third_slot.done()

        # Finalization releases a generation permit without releasing the
        # group's active demand slot.
        await _finalize(buffer, 0)
        assert await third_slot
        assert (
            _buffer_metric(buffer, "rollout_buffer/available_generation_permits") == 1
        )
        await buffer.close()

    asyncio.run(run())


def test_adaptive_buffer_admits_a_granted_slot_after_the_demand_drops() -> None:
    async def run() -> None:
        # P=2: demand 6. Five groups finished and none trained: a full shelf.
        buffer = _adaptive_buffer(num_prompts=2, generation_capacity=8)
        for group_id in range(5):
            await _admit(buffer, group_id)
            await buffer.claim_next()
            await _finalize(buffer, group_id)
        assert await buffer.wait_for_slot()

        # While the data input loop reads the sample, a step start lowers the demand:
        # unavailable 6 - 5 = 1 -> needed 2 + 1 + 1 = 4 -> one group down to 5, which the 5 groups fill.
        await buffer.record_step_start(trainer_policy_version=0)
        assert _buffer_metric(buffer, "rollout_buffer/demand_target_groups") == 5
        # The granted slot is still admitted; the next one waits.
        await buffer.add_work(RolloutGroupWork(group_id=5, sample=object()))
        assert _buffer_metric(buffer, "rollout_buffer/active_slots_in_use_peak") == 6
        waiter = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not waiter.done()
        await buffer.close()

    asyncio.run(run())


def test_batcher_passes_the_version_that_trains_each_group() -> None:
    """Batches train in order, one per version, so the batch a group joins sets its consuming version."""
    # Groups 0, 2, 3, 4 are trainable; group 1 is not and joins no batch's count.
    trainable = {0: True, 1: False, 2: True, 3: True, 4: True}
    consuming_policy_versions = []

    class _Buffer:
        async def take_finalized(self, *, consuming_policy_version):
            consuming_policy_versions.append(consuming_policy_version)
            group_id = len(consuming_policy_versions) - 1
            if group_id == len(trainable):
                return None
            return RolloutGroup(group_id=group_id, rollouts=[])

        async def release_active_groups(self, count, *, reason):
            assert reason == "untrainable_group"

    trainable_seen = []

    async def add_group(*, rollout_group):
        # A batch of 2 trainable groups, as the batcher actor returns it.
        if not trainable[rollout_group.group_id]:
            return None, False
        trainable_seen.append(rollout_group.group_id)
        return ("batch" if len(trainable_seen) % 2 == 0 else None), True

    controller = SimpleNamespace(
        config=SimpleNamespace(
            async_loop=SimpleNamespace(batcher=Batcher.Config()),
            generator=SimpleNamespace(
                engine=SimpleNamespace(return_routed_experts=False)
            ),
        ),
        start_step=5,
        _group_buffer=_Buffer(),
        batcher=SimpleNamespace(add_group=SimpleNamespace(call_one=add_group)),
    )
    queue: asyncio.Queue = asyncio.Queue()
    asyncio.run(Controller._batcher_loop(controller, training_batch_queue=queue))
    # [0, 1, 2] trains at version 5, [3, 4] at 6; the last call returns None
    assert consuming_policy_versions == [5, 5, 5, 6, 6, 7]
    assert queue.qsize() == 3  # two batches and the None sentinel


def test_generation_capacity_default_follows_target_offpolicy_steps() -> None:
    config = AdaptiveRolloutGroupWorkBuffer.Config(
        max_offpolicy_steps=10, target_offpolicy_steps=6
    )
    # 2 x (6 + 1) versions x 48 prompts: room for up to half the groups being untrainable.
    assert config.resolve_generation_capacity(48) == 672
    loop = AsyncLoopConfig(num_prompts_per_train_step=48, group_buffer=config)
    assert loop.max_concurrent_rollout_groups == 672
    buffer = config.build(num_prompts_per_train_step=48)
    assert _buffer_metric(buffer, "rollout_buffer/generation_capacity_groups") == 672


def test_generation_capacity_default_without_target_fills_vllm() -> None:
    config = AdaptiveRolloutGroupWorkBuffer.Config(max_offpolicy_steps=10)
    # 512 sequences x 24 generator DP shards / 16 samples per prompt
    assert (
        config.resolve_generation_capacity(
            48, num_generator_dp_shards=24, num_samples_per_prompt=16
        )
        == 768
    )
    loop = AsyncLoopConfig(
        num_prompts_per_train_step=48, num_samples_per_prompt=16, group_buffer=config
    )
    assert loop.resolve_max_concurrent_rollout_groups(num_generator_dp_shards=12) == 384
    # Without the generator layout the default can't be computed.
    with pytest.raises(ValueError, match="needs num_generator_dp_shards"):
        loop.max_concurrent_rollout_groups


def test_generation_capacity_explicit_overrides_default() -> None:
    explicit = AdaptiveRolloutGroupWorkBuffer.Config(
        max_offpolicy_steps=12, target_offpolicy_steps=6, generation_capacity=96
    )
    assert explicit.resolve_generation_capacity(16) == 96
    assert (
        explicit.resolve_generation_capacity(
            16, num_generator_dp_shards=24, num_samples_per_prompt=16
        )
        == 96
    )


def test_adaptive_buffer_warns_once_when_generation_capacity_caps_demand(
    caplog,
) -> None:
    # Demand starts at 3 batches (24 groups), above the capacity of 10.
    buffer = _adaptive_buffer(num_prompts=8, generation_capacity=10)

    async def run() -> None:
        with caplog.at_level(logging.WARNING):
            await buffer.record_step_start(trainer_policy_version=0)
            await buffer.record_step_start(trainer_policy_version=1)

    asyncio.run(run())
    warnings = [r for r in caplog.records if "> generation_capacity 10" in r.message]
    assert len(warnings) == 1


@pytest.mark.parametrize(
    "num_samples_per_prompt, fraction, max_stragglers",
    [
        (16, 0.05, 1),  # 0.8 rounds down to 0, but one is always allowed
        (64, 0.05, 3),
        (100, 0.29, 29),  # 0.29 * 100 is 28.999... in floats
        (2, 0.9, 1),  # at most n - 1: one rollout always trains
        (1, 0.5, None),  # a group of 1 has nothing to keep
    ],
)
def test_max_stragglers_rounds_down_but_allows_at_least_one(
    num_samples_per_prompt, fraction, max_stragglers
) -> None:
    loop = AsyncLoopConfig(
        num_samples_per_prompt=num_samples_per_prompt,
        group_buffer=AdaptiveRolloutGroupWorkBuffer.Config(
            max_offpolicy_steps=10, max_straggler_fraction=fraction
        ),
    )
    assert loop.max_stragglers == max_stragglers


@pytest.mark.parametrize(
    "fraction, max_offpolicy_steps, grace, match",
    [
        (0.0, 10, 2, r"in \(0, 1\)"),
        (1.0, 10, 2, r"in \(0, 1\)"),
        (0.05, None, 2, "needs max_offpolicy_steps"),
        (0.05, 10, 0, "straggler_grace_steps must be >= 1"),
    ],
)
def test_adaptive_buffer_config_rejects_a_bad_straggler_cut(
    fraction, max_offpolicy_steps, grace, match
) -> None:
    with pytest.raises(ValueError, match=match):
        AdaptiveRolloutGroupWorkBuffer.Config(
            max_offpolicy_steps=max_offpolicy_steps,
            max_straggler_fraction=fraction,
            straggler_grace_steps=grace,
        )


def _cut_buffer(
    *, max_offpolicy_steps: int, fraction: float | None = 0.05
) -> AdaptiveRolloutGroupWorkBuffer:
    return AdaptiveRolloutGroupWorkBuffer.Config(
        max_offpolicy_steps=max_offpolicy_steps,
        generation_capacity=4,
        max_straggler_fraction=fraction,
        straggler_grace_steps=2,
    ).build(num_prompts_per_train_step=1)


def _cut_requests(work: RolloutGroupWork) -> list[str]:
    queue = work.straggler_cut_requests
    return [queue.get_nowait() for _ in range(queue.qsize())]


def _cut_rollout(group_id: int) -> Rollout:
    return Rollout(
        group_id=group_id,
        rollout_id=1,
        status=RolloutStatus.ERROR_SYSTEM,
        logs={"straggler_cut": "cap"},
    )


@pytest.mark.parametrize(
    "fraction, gate_version, requests",
    [
        # Gate at version 1: the grace ends at batch 3; the cap requests still follow (they cut nothing
        # once the cut went through)
        (0.05, 1, {3: "grace", 5: "cap_minus_1", 6: "cap"}),
        # Gate never reached: only the cap requests, which the worker declines with more than k unfinished
        (0.05, None, {5: "cap_minus_1", 6: "cap"}),
        # Gate at version 4: the step before the last batch comes first; the grace ends at the last batch
        (0.05, 4, {5: "cap_minus_1", 6: "grace"}),
        (None, 1, {}),
    ],
)
def test_adaptive_buffer_requests_a_straggler_cut_at_each_trigger(
    fraction, gate_version, requests
) -> None:
    async def run() -> None:
        buffer = _cut_buffer(max_offpolicy_steps=6, fraction=fraction)
        generating, waiting = (
            RolloutGroupWork(group_id=group_id, sample=object()) for group_id in (0, 1)
        )
        for work in (generating, waiting):
            await buffer.add_work(work)
        await buffer.claim_next()  # g0 generates under version 0; g1 waits

        takers, seen = [], {}
        for version in range(1, 7):
            takers.append(
                asyncio.create_task(
                    buffer.take_finalized(consuming_policy_version=version)
                )
            )
            await asyncio.sleep(0)
            if version == gate_version:
                # g0's worker reports the gate while batch `version` is assembled; a repeat is ignored
                buffer.record_straggler_gate(generating)
                buffer.record_straggler_gate(generating)
                assert generating.straggler_gate_version == version
            # At most one request per batch
            for reason in _cut_requests(generating):
                assert version not in seen
                seen[version] = reason
        assert seen == requests
        # A waiting group has no claim version and no worker
        assert _cut_requests(waiting) == []
        if fraction is not None:
            metrics = {metric.key: metric.value.value for metric in buffer.metrics()}
            assert metrics["rollout_buffer/straggler_gates"] == (
                gate_version is not None
            )
            for reason in ("grace", "cap_minus_1", "cap"):
                key = f"rollout_buffer/straggler_cut_requests/{reason}"
                assert metrics[key] == (reason in requests.values())
        await buffer.close()
        assert await asyncio.gather(*takers) == [None] * 6

    asyncio.run(run())


def test_adaptive_buffer_trains_a_cut_group_in_its_last_batch_once() -> None:
    async def run() -> None:
        buffer = _cut_buffer(max_offpolicy_steps=2)
        await _admit(buffer, 0)
        work = await buffer.claim_next()  # under version 0

        # The batch for version 2 is g0's last: the cut is requested, then the group finalizes with its cut
        takers = [
            asyncio.create_task(buffer.take_finalized(consuming_policy_version=1))
        ]
        await asyncio.sleep(0)
        takers.append(
            asyncio.create_task(buffer.take_finalized(consuming_policy_version=2))
        )
        await asyncio.sleep(0)
        assert _cut_requests(work) == ["cap_minus_1", "cap"]
        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[_cut_rollout(0)]))
        (taken,), _ = await asyncio.wait(takers, return_when=asyncio.FIRST_COMPLETED)
        assert taken.result().group_id == 0
        await buffer.release_active_groups(1, reason="trained")

        # Trained once: not dropped, not cancelled, its slot released once, nothing left in the buffer
        assert buffer.pop_dropped_group_ids() == []
        assert not work.cancelled
        metrics = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert metrics["rollout_buffer/straggler_cut_groups"] == 1
        assert metrics["rollout_buffer/dropped_too_old"] == 0
        assert metrics["rollout_buffer/cancelled_too_old"] == 0
        assert metrics["rollout_buffer/num_groups_inflight"] == 0
        assert buffer._active_rollout_groups == 0
        await buffer.close()
        await asyncio.gather(*takers)  # the other taker returns None

    asyncio.run(run())


def test_adaptive_buffer_counts_a_cut_group_still_generating_at_the_next_batch() -> None:
    async def run() -> None:
        buffer = _cut_buffer(max_offpolicy_steps=1)
        await _admit(buffer, 0)
        work = await buffer.claim_next()  # under version 0
        generation_task = asyncio.create_task(asyncio.sleep(3600))
        buffer.track_generation(work, generation_task)

        # The batch for version 1 requests the cut; the group still generates (e.g. slow scoring after it)
        takers = [
            asyncio.create_task(buffer.take_finalized(consuming_policy_version=1))
        ]
        await asyncio.sleep(0)
        assert _cut_requests(work) == ["cap"]
        # The batch for version 2 cancels it as too old, as without a cut, and requests nothing more
        takers.append(
            asyncio.create_task(buffer.take_finalized(consuming_policy_version=2))
        )
        await asyncio.sleep(0)
        assert work.cancelled and _cut_requests(work) == []
        with pytest.raises(asyncio.CancelledError):
            await generation_task
        assert buffer.pop_dropped_group_ids() == [0]
        metrics = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert metrics["rollout_buffer/cancelled_too_old"] == 1
        assert metrics["rollout_buffer/cancelled_too_old_after_cut_request"] == 1
        await buffer.close()
        assert await asyncio.gather(*takers) == [None, None]

    asyncio.run(run())


@pytest.mark.parametrize("max_stragglers", [1, None])
def test_rollout_loop_passes_the_cut_signal_only_when_the_cut_is_on(
    max_stragglers,
) -> None:
    async def run() -> None:
        buffer = _cut_buffer(max_offpolicy_steps=4)
        calls = []

        async def run_group_rollouts(*, group_id: int, **kwargs) -> RolloutGroup:
            calls.append(kwargs)
            if max_stragglers is not None:
                kwargs["on_straggler_gate"]()  # the worker reports the gate
            return RolloutGroup(group_id=group_id, rollouts=[])

        async def release_groups(group_ids: list[int]) -> None:
            pass

        controller = object.__new__(Controller)
        controller.config = SimpleNamespace(
            async_loop=SimpleNamespace(
                num_samples_per_prompt=2, max_stragglers=max_stragglers
            )
        )
        controller._sampling = None
        controller.generator_router = SimpleNamespace(
            release_groups=SimpleNamespace(call_one=release_groups)
        )
        controller.rollout_recorder = SimpleNamespace(record=lambda **kwargs: None)
        controller._rollouter = SimpleNamespace(
            run_group_rollouts=run_group_rollouts,
            record_group_outcome=lambda group: None,
        )
        worker = asyncio.create_task(
            controller._rollout_loop(
                group_buffer=buffer, generate_fn=None, release_session_fn=None
            )
        )
        work = RolloutGroupWork(group_id=0, sample=object())
        await buffer.wait_for_slot()
        await buffer.add_work(work)
        assert (await buffer.take_finalized(consuming_policy_version=0)).group_id == 0
        await buffer.close()
        await worker

        (kwargs,) = calls
        if max_stragglers is None:  # off: the call is the one before the cut existed
            assert not {"max_stragglers", "cut_requests", "on_straggler_gate"} & set(
                kwargs
            )
        else:
            assert kwargs["max_stragglers"] == max_stragglers
            assert kwargs["cut_requests"] is work.straggler_cut_requests
            assert work.straggler_gate_version == 0

    asyncio.run(run())


def test_adaptive_buffer_logs_cut_metrics_only_when_the_cut_is_on() -> None:
    keys = {metric.key for metric in _cut_buffer(max_offpolicy_steps=1).metrics()}
    assert {
        "rollout_buffer/straggler_gates",
        "rollout_buffer/straggler_cut_requests/grace",
        "rollout_buffer/straggler_cut_requests/cap_minus_1",
        "rollout_buffer/straggler_cut_requests/cap",
        "rollout_buffer/straggler_cut_groups",
        "rollout_buffer/dropped_too_old_after_cut",
        "rollout_buffer/cancelled_too_old_after_cut_request",
    } <= keys
    off = _cut_buffer(max_offpolicy_steps=1, fraction=None)
    assert not any(
        "straggler" in key or "after_cut" in key
        for key in {metric.key for metric in off.metrics()}
    )


def test_adaptive_buffer_counts_cut_groups_that_missed_their_last_batch() -> None:
    async def run() -> None:
        buffer = _cut_buffer(max_offpolicy_steps=1)
        for group_id in range(2):
            await _admit(buffer, group_id)
            await buffer.claim_next()  # under version 0
        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[_cut_rollout(0)]))
        await _finalize(buffer, 1)

        # The batch for version 2: both groups finished too late and are dropped; only g0 had a cut
        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=2))
        await asyncio.sleep(0)
        metrics = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert metrics["rollout_buffer/dropped_too_old"] == 2
        assert metrics["rollout_buffer/dropped_too_old_after_cut"] == 1
        await buffer.close()
        assert await taker is None

    asyncio.run(run())


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


def test_validation_logs_the_policies_its_rollouts_sampled() -> None:
    """A weight sync during a rollout shows as a newer max version and a mixed-policy rollout."""
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(validation=ValidationConfig())
    )
    controller.generator_router = SimpleNamespace(
        release_groups=SimpleNamespace(call_one=AsyncMock())
    )
    controller._rollouter = SimpleNamespace(
        run_group_rollouts=AsyncMock(
            side_effect=[
                # A sync to policy 27 landed during the second turn.
                RolloutGroup(
                    group_id=-1,
                    rollouts=[_validation_rollout([(25, 25), (25, 27)])],
                ),
                RolloutGroup(group_id=-2, rollouts=[_validation_rollout([(25, 25)])]),
            ]
        ),
    )

    _, metrics = asyncio.run(
        controller._collect_validation_rollouts(
            samples=["prompt", "prompt"], sampling=SamplingConfig(), step=25
        )
    )

    reduced = m.MetricsProcessor._aggregate_metrics(metrics)
    assert reduced["validation/min_policy_version/min"] == 25
    assert reduced["validation/max_policy_version/max"] == 27
    assert reduced["validation/mixed_policy_rollouts/mean"] == 0.5


@pytest.mark.parametrize(
    "overlapped, group_ids", [(False, [-1, -2]), (True, [-1_000_000, -1_000_001])]
)
def test_overlapped_validation_uses_lowest_priority_group_ids(
    overlapped, group_ids
) -> None:
    """A pass beside training runs and releases ids at or below -1_000_000, which `vllm_priority`
    puts after training; a pass training waits for keeps -1, -2, ..."""
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(validation=ValidationConfig())
    )
    controller.generator_router = SimpleNamespace(
        release_groups=SimpleNamespace(call_one=AsyncMock())
    )
    controller._rollouter = SimpleNamespace(
        run_group_rollouts=AsyncMock(
            side_effect=lambda *, group_id, **kwargs: RolloutGroup(
                group_id=group_id, rollouts=[]
            )
        ),
    )

    asyncio.run(
        controller._collect_validation_rollouts(
            samples=["prompt", "prompt"],
            sampling=SamplingConfig(),
            step=3,
            overlapped=overlapped,
        )
    )

    run_group_ids = [
        call.kwargs["group_id"]
        for call in controller._rollouter.run_group_rollouts.call_args_list
    ]
    assert run_group_ids == group_ids
    controller.generator_router.release_groups.call_one.assert_awaited_once_with(
        group_ids
    )


class _FakeValidation:
    """Stands in for `Controller.validate`: each pass runs until the test ends it."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []
        self.overlapped: dict[int, bool] = {}
        self._end_events: dict[int, asyncio.Event] = {}
        self._errors: dict[int, Exception] = {}

    @property
    def started_steps(self) -> list[int]:
        return [step for event, step in self.events if event == "start"]

    async def __call__(self, *, step: int, overlapped: bool = False) -> list[m.Metric]:
        self.events.append(("start", step))
        self.overlapped[step] = overlapped
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
    metrics processor that records each `log` and `commit`."""
    controller = _controller([] if events is None else events, on_pull=on_pull)
    controller.config.async_loop.validation = validation
    controller.metrics_processor = Mock()
    controller._validation_task = None
    controller._extra_validation_tasks = {}
    controller._extra_validation_requested = set()
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
        global_routing_token_counts=[1],
        group_ids=[step],
    )


def _logged_steps(controller: Controller, *, is_validation: bool) -> list[int]:
    return [
        call.kwargs["step"]
        for call in controller.metrics_processor.log.call_args_list
        if call.kwargs["is_validation"] == is_validation
    ]


def _validation_logs(controller: Controller) -> list[tuple[int, list[str]]]:
    """`(step, metric keys)` of each validation log."""
    return [
        (call.kwargs["step"], [metric.key for metric in call.kwargs["metrics"]])
        for call in controller.metrics_processor.log.call_args_list
        if call.kwargs["is_validation"]
    ]


async def _run_until(condition) -> None:
    """Let other tasks run until `condition()` holds (up to ~5 s: the trainer loop also awaits
    worker threads, so a fixed count of event-loop ticks is not enough)."""
    for _ in range(5000):
        if condition():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition never held")


async def _train(controller: Controller, queue: asyncio.Queue, step: int) -> None:
    """Hand `_trainer_loop` the batch for `step` and wait until it logs that step."""
    await queue.put(_training_batch(step))
    await _run_until(lambda: step in _logged_steps(controller, is_validation=False))


def test_overlapped_validation_runs_beside_training() -> None:
    """Training keeps stepping while a pass runs; the pass is logged at the step it ends, and a
    validation step that comes meanwhile starts the next pass once it ends."""
    events: list[str] = []
    pulls_may_end = asyncio.Event()
    controller, validation = _controller_for_validation(
        ValidationConfig(freq=2, overlap_training=True),
        events=events,
        on_pull=pulls_may_end.wait,
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
        # Hold step 2's pull: the pass must not start before it ends.
        pulls_may_end.clear()
        await _train(controller, queue, 2)
        await _run_until(lambda: "pull 2" in events)
        for _ in range(10):
            await asyncio.sleep(0)
        assert validation.started_steps == []
        pulls_may_end.set()
        await _run_until(lambda: validation.started_steps == [2])

        # Steps 3 and 4 train while the step-2 pass runs; step 4's pass waits for it.
        await _train(controller, queue, 3)
        await _train(controller, queue, 4)
        assert validation.started_steps == [2]
        assert _logged_steps(controller, is_validation=True) == []

        validation.end(2)
        await _run_until(lambda: controller._validation_task.done())
        await _train(controller, queue, 5)
        assert _logged_steps(controller, is_validation=True) == [5]
        await _run_until(lambda: validation.started_steps == [2, 5])

        await _train(controller, queue, 6)
        await asyncio.wait_for(trainer, timeout=5)
        # Step 6 is the last step: no pass starts there. run() awaits the running one.
        assert validation.started_steps == [2, 5]
        assert not controller._validation_task.done()
        validation.end(5)
        await controller._validation_task

    asyncio.run(run())
    # The pass logged at step 5 joins row 5: the commit that pushes row 5 comes after it.
    calls = [
        (name, kwargs.get("step"), kwargs.get("is_validation"))
        for name, _, kwargs in controller.metrics_processor.mock_calls
        if name in ("log", "commit")
    ]
    validation_log = calls.index(("log", 5, True))
    assert ("commit", None, None) not in calls[
        calls.index(("log", 5, False)) : validation_log
    ]
    assert calls[validation_log + 1][0] == "commit"


def test_overlapped_validation_failure_does_not_stop_training(caplog) -> None:
    """A pass that raises logs its error and `validation/pass_failures` at the step it ends; training
    goes on, and so do the next passes."""
    controller, validation = _controller_for_validation(
        ValidationConfig(freq=1, overlap_training=True)
    )

    async def run() -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        trainer = asyncio.create_task(
            controller._trainer_loop(queue, num_training_steps=3)
        )
        await _train(controller, queue, 1)
        await _run_until(lambda: validation.started_steps == [1])
        validation.end(1, error=RuntimeError("generator died"))
        await _run_until(lambda: controller._validation_task.done())
        await _train(controller, queue, 2)
        await queue.put(_training_batch(3))
        await asyncio.wait_for(trainer, timeout=5)

    with caplog.at_level(logging.ERROR):
        asyncio.run(run())
    assert _logged_steps(controller, is_validation=False) == [1, 2, 3]
    assert validation.started_steps == [1, 2]
    assert _validation_logs(controller) == [(2, ["validation/pass_failures"])]
    assert "validation_step_1 failed" in caplog.text


def test_blocking_validation_failure_stops_training() -> None:
    """Unlike a pass beside training, a pass that training waits for fails the run."""
    controller, validation = _controller_for_validation(ValidationConfig(freq=1))
    validation.end(1, error=RuntimeError("generator died"))

    async def run() -> None:
        queue: asyncio.Queue = asyncio.Queue()
        for step in (1, 2):
            queue.put_nowait(_training_batch(step))
        await controller._trainer_loop(queue, num_training_steps=2)

    with pytest.raises(RuntimeError, match="generator died"):
        asyncio.run(run())


def test_overlapped_validation_starts_no_pass_at_the_last_step() -> None:
    """A pass asked for while another runs does not start at the last step; the final pass
    after training covers it. The last step leaves a finished pass for run() to log."""
    controller, validation = _controller_for_validation(
        ValidationConfig(freq=1, overlap_training=True)
    )

    async def run() -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        trainer = asyncio.create_task(
            controller._trainer_loop(queue, num_training_steps=3)
        )
        await _train(controller, queue, 1)
        await _run_until(lambda: validation.started_steps == [1])
        # Step 2 asks for a pass while step 1's runs.
        await _train(controller, queue, 2)
        validation.end(1)
        await _run_until(lambda: controller._validation_task.done())
        await queue.put(_training_batch(3))
        await asyncio.wait_for(trainer, timeout=5)

    asyncio.run(run())
    # Step 3 started no pass and left step 1's for run().
    assert validation.started_steps == [1]
    assert _logged_steps(controller, is_validation=True) == []
    assert controller._validation_task.done()


def test_validation_pauses_training_by_default() -> None:
    controller, validation = _controller_for_validation(ValidationConfig(freq=2))

    async def run() -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        trainer = asyncio.create_task(
            controller._trainer_loop(queue, num_training_steps=3)
        )
        await queue.put(_training_batch(1))
        await queue.put(_training_batch(2))
        await _run_until(lambda: validation.started_steps == [2])

        await queue.put(_training_batch(3))
        for _ in range(20):
            await asyncio.sleep(0)
        assert _logged_steps(controller, is_validation=False) == [1, 2]

        validation.end(2)
        await asyncio.wait_for(trainer, timeout=5)
        assert _logged_steps(controller, is_validation=False) == [1, 2, 3]
        assert _logged_steps(controller, is_validation=True) == [2]

    asyncio.run(run())


def _controller_for_run() -> tuple[Controller, _FakeValidation]:
    """A `Controller` whose `run()` runs 4 steps with overlapped validation on fakes. Every loop but
    `_trainer_loop` blocks; each test sets `_trainer_loop`."""
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=AsyncLoopConfig(
            num_training_steps=4,
            validation=ValidationConfig(overlap_training=True),
        ),
        trainer=SimpleNamespace(
            training=SimpleNamespace(
                num_tokens_per_microbatch_per_dp_rank=64, max_context_length=64
            )
        ),
    )
    controller.start_step = 0
    controller.trainer = Mock()
    controller.generator_router = SimpleNamespace(
        wait_engine_loop=SimpleNamespace(call_one=_block_forever)
    )
    controller.trainer_dp_degree = 1
    controller.tokenizer = SimpleNamespace(eos_id=0)
    controller._sampling = SamplingConfig()
    controller.metrics_processor = Mock()
    controller._log_reward_delta = Mock()
    controller._validation_task = None
    controller._extra_validation_tasks = {}
    controller._extra_validation_requested = set()
    validation = _FakeValidation()
    controller.validate = validation
    controller._data_input_loop = _block_forever
    controller._rollout_loop = _block_forever
    controller._batcher_loop = _block_forever
    return controller, validation


@pytest.mark.parametrize(
    "pass_3_ends_in_training, logged_steps", [(True, [2, 3, 4]), (False, [2, 4, 5])]
)
def test_overlapped_validation_before_and_after_training(
    pass_3_ends_in_training, logged_steps
) -> None:
    """The pre-training pass runs while training starts. A pass still running when training
    ends is awaited and logged at the last step; the final pass then logs one step later."""
    controller, validation = _controller_for_run()

    # Stands in for `_trainer_loop`'s overlap branch, which the tests above cover.
    async def trainer_loop(queue, *, num_training_steps: int) -> None:
        # Training starts while the pre-training pass runs.
        assert validation.started_steps == [0]
        assert not controller._validation_task.done()
        validation.end(0)
        await _run_until(lambda: controller._validation_task.done())
        controller._log_finished_validation(step=2)
        controller._start_validation(step=3)
        await _run_until(lambda: validation.started_steps == [0, 3])
        if pass_3_ends_in_training:
            validation.end(3)
            await _run_until(lambda: controller._validation_task.done())
            controller._log_finished_validation(step=3)
        else:
            # Training ends with the step-3 pass still running.
            asyncio.get_running_loop().call_later(0.05, validation.end, 3)

    controller._trainer_loop = trainer_loop
    validation.end(4)
    validation.end(5)

    asyncio.run(asyncio.wait_for(controller.run(), timeout=5))

    final_pass_step = logged_steps[-1]
    assert validation.events == [
        ("start", 0),
        ("end", 0),
        ("start", 3),
        ("end", 3),
        ("start", final_pass_step),
        ("end", final_pass_step),
    ]
    assert _logged_steps(controller, is_validation=True) == logged_steps
    # Only the post-training pass blocks.
    assert validation.overlapped == {0: True, 3: True, final_pass_step: False}
    controller._log_reward_delta.assert_called_once_with(
        {"validation_reward/mean": 0.0},
        {"validation_reward/mean": float(final_pass_step)},
    )


def test_overlapped_validation_failures_do_not_fail_the_run() -> None:
    """A failed pre-training pass leaves no pre score; a pass that fails after training logs
    `validation/pass_failures` at the last step, and the final pass still runs one step later."""
    controller, validation = _controller_for_run()

    async def trainer_loop(queue, *, num_training_steps: int) -> None:
        validation.end(0, error=RuntimeError("sandbox died"))
        await _run_until(lambda: controller._validation_task.done())
        controller._log_finished_validation(step=2)
        controller._start_validation(step=3)
        await _run_until(lambda: validation.started_steps == [0, 3])
        # Training ends with the step-3 pass still running; it fails afterwards.
        asyncio.get_running_loop().call_later(
            0.05, validation.end, 3, RuntimeError("sandbox died")
        )

    controller._trainer_loop = trainer_loop
    validation.end(5)

    asyncio.run(asyncio.wait_for(controller.run(), timeout=5))

    assert _validation_logs(controller) == [
        (2, ["validation/pass_failures"]),
        (4, ["validation/pass_failures"]),
        (5, ["validation_reward"]),
    ]
    controller._log_reward_delta.assert_called_once_with(
        {}, {"validation_reward/mean": 5.0}
    )


def test_close_cancels_a_running_pass_before_closing_the_rollouter() -> None:
    """After a training crash, `close()` cancels the pass and waits for it, then closes the
    rollouter the pass was using."""
    controller = object.__new__(Controller)
    controller.trainer = None
    controller.generator_router = None
    controller.metrics_processor = Mock()
    controller.actor_system = None
    controller._extra_validation_tasks = {}
    validation = _FakeValidation()
    controller.validate = validation
    pass_done_at_rollouter_close: list[bool] = []

    async def close_rollouter() -> None:
        pass_done_at_rollouter_close.append(controller._validation_task.done())

    controller._rollouter = SimpleNamespace(close=close_rollouter)

    async def run() -> None:
        controller._start_validation(step=0)
        await _run_until(lambda: validation.started_steps == [0])
        await asyncio.wait_for(controller.close(), timeout=5)

    asyncio.run(run())
    assert controller._validation_task.cancelled()
    assert pass_done_at_rollouter_close == [True]


@pytest.mark.parametrize(
    "overlapped, key",
    [
        (False, "timing/validation/blocking"),
        (True, "timing/validation/overlapped"),
    ],
)
def test_validation_wall_time_says_whether_the_pass_blocked(overlapped, key) -> None:
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(validation=ValidationConfig(steps=1))
    )
    controller._rollouter = SimpleNamespace(
        get_validation_samples=lambda steps: ["prompt"]
    )
    controller._sampling = SamplingConfig()
    controller.rollout_recorder = Mock()

    async def collect(*, samples, sampling, step, overlapped):
        return [], []

    controller._collect_validation_rollouts = collect
    metrics = asyncio.run(controller.validate(step=5, overlapped=overlapped))
    assert [metric.key for metric in metrics] == [key]


def _scored_group(group_id: int, rewards: list[float]) -> RolloutGroup:
    return RolloutGroup(
        group_id=group_id,
        rollouts=[
            Rollout(
                group_id=group_id,
                rollout_id=rollout_id,
                status=RolloutStatus.COMPLETED,
                reward=reward,
            )
            for rollout_id, reward in enumerate(rewards)
        ],
    )


@pytest.mark.parametrize(
    "num_samples_per_prompt, rewards, expected_mean, expected_best_of",
    [
        (1, [[1.0], [0.0]], 0.5, None),
        (2, [[1.0, 0.0], [0.0, 0.0]], 0.25, 0.5),
    ],
)
def test_validate_rolls_out_each_prompt_n_times(
    num_samples_per_prompt: int,
    rewards: list[list[float]],
    expected_mean: float,
    expected_best_of: float | None,
) -> None:
    """2 prompts x n rollouts at T=1: the mean covers all rollouts; best_of_n, logged only for
    n > 1, the best rollout per prompt."""
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(
            validation=ValidationConfig(
                steps=2, num_samples_per_prompt=num_samples_per_prompt
            )
        )
    )
    controller._sampling = SamplingConfig(temperature=0.8, seed=7, stop_token_ids=[1])
    controller.generator_router = SimpleNamespace(
        release_groups=SimpleNamespace(call_one=AsyncMock())
    )
    controller.rollout_recorder = SimpleNamespace(record=lambda **kwargs: None)
    run_group_rollouts = AsyncMock(
        side_effect=[
            _scored_group(-(i + 1), group_rewards)
            for i, group_rewards in enumerate(rewards)
        ]
    )
    controller._rollouter = SimpleNamespace(
        get_validation_samples=lambda steps: ["a", "b"],
        run_group_rollouts=run_group_rollouts,
    )

    metrics = asyncio.run(controller.validate(step=0))

    calls = run_group_rollouts.call_args_list
    assert [call.kwargs["group_size"] for call in calls] == [num_samples_per_prompt] * 2
    expected_sampling = SamplingConfig(temperature=1.0, seed=7, stop_token_ids=[1])
    assert [call.kwargs["sampling"] for call in calls] == [expected_sampling] * 2
    aggregated = m.MetricsProcessor._aggregate_metrics(metrics)
    assert aggregated["validation_reward/_mean"] == expected_mean
    best_of_key = f"validation_reward/best_of_{num_samples_per_prompt}/mean"
    assert aggregated.get(best_of_key) == expected_best_of


def test_validation_config_rejects_bad_sampling() -> None:
    with pytest.raises(ValueError, match="temperature"):
        ValidationConfig(temperature=-1.0)
    with pytest.raises(ValueError, match="num_samples_per_prompt"):
        ValidationConfig(num_samples_per_prompt=0)
    with pytest.raises(ValueError, match="greedy"):
        ValidationConfig(temperature=0.0, num_samples_per_prompt=4)


def test_event_loop_lag_records_a_blocking_call() -> None:
    async def run() -> list[m.Metric]:
        lag = m.EventLoopLag("controller", interval_s=0.01, threshold_s=0.05)
        await asyncio.sleep(0.02)  # let the lag task start its first check
        time.sleep(0.3)  # blocks the loop
        await asyncio.sleep(0.05)  # the lag task wakes late and records it
        metrics = lag.flush()
        lag.close()
        return metrics

    values = {type(metric.value): metric.value.value for metric in asyncio.run(run())}
    assert values[m.Max] >= 250.0
    assert values[m.Sum] >= values[m.Max]
