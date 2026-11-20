# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for async-controller pieces: batcher group-counting, the active-slot buffer backpressure,
the consume-time staleness invariant, the metrics timer drain, and RolloutTurnID."""

import asyncio
import itertools
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import torch
from torchtitan.rl.controller.controller import Controller, ValidationConfig
from torchtitan.rl.controller.metrics import (
    compute_perf_ratio_metrics,
    compute_policy_age_metrics,
    MetricsTimer,
)
from torchtitan.rl.controller.work_buffer import (
    RolloutGroupWork,
    RolloutGroupWorkBuffer,
)
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import RolloutGroup
from torchtitan.rl.rollout.finished_group import FinishedRolloutGroup
from torchtitan.rl.rollout.packed_group import pack_rollout_group, PackedRolloutGroup
from torchtitan.rl.rollout.types import (
    Prompt,
    Rollout,
    rollout_turn_lengths,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
)

from torchtitan.rl.trainer.batcher import (
    Batcher,
    MicrobatchPacker,
    StepPlan,
    TrainingMicrobatch,
)
from torchtitan.rl.trainer.microbatch_group_reader import MicrobatchGroupReader
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleSpec


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


def _rollout(*, group_id: int, rollout_id: int, token_ids: list[int]) -> Rollout:
    """A one-turn rollout whose training sample is `token_ids`: its first token is the prompt,
    the rest the completion (advantage 1). Rewards differ across a group, so no group is
    zero-std."""
    return Rollout(
        group_id=group_id,
        rollout_id=rollout_id,
        status=RolloutStatus.COMPLETED,
        turns=[
            RolloutTurn(
                rollout_id=RolloutTurnID(
                    group_id=group_id, rollout_id=rollout_id, turn_id=0
                ),
                prompt_prefix_len=0,
                prompt_delta_token_ids=np.array(token_ids[:1], dtype=np.int32),
                completion_token_ids=np.array(token_ids[1:], dtype=np.int32),
                completion_logprobs=np.zeros(len(token_ids) - 1, dtype=np.float32),
                min_policy_version=0,
                max_policy_version=0,
            )
        ],
        reward=float(rollout_id),
        advantage=1.0,
    )


def _trainable_group(group_id: int, *, num_samples: int) -> RolloutGroup:
    return RolloutGroup(
        group_id=group_id,
        rollouts=[
            _rollout(group_id=group_id, rollout_id=i, token_ids=[1, 2, 3])
            for i in range(num_samples)
        ],
    )


def _variable_length_group(group_id: int, *, token_lengths: list[int]) -> RolloutGroup:
    return RolloutGroup(
        group_id=group_id,
        rollouts=[
            _rollout(
                group_id=group_id,
                rollout_id=rollout_id,
                token_ids=list(range(token_length)),
            )
            for rollout_id, token_length in enumerate(token_lengths)
        ],
    )


def _add_group(batcher: Batcher, group: RolloutGroup) -> tuple[StepPlan | None, bool]:
    """`Batcher.add_group` with `group`'s summary, as the rollout worker packs it."""
    summary, _ = pack_rollout_group(group)
    return batcher.add_group(summary=summary)


def _leader_microbatch_groups(
    plan: StepPlan,
    groups: list[RolloutGroup],
    *,
    config: Batcher.Config,
    dp_degree: int,
    num_tokens_per_microbatch_per_dp_rank: int,
    max_context_length: int,
) -> list[list[list[TrainingMicrobatch]]]:
    """What the trainer DP replicas train on for `plan`: the controller's columns
    (`Controller._replica_columns`) of the packed `groups`, each packed by its leader's
    `MicrobatchGroupReader` with no peers. [num_groups][num_microbatches_per_group][dp_degree]."""
    controller = SimpleNamespace(
        trainer_dp_degree=dp_degree,
        _packed_groups_by_id={
            group.group_id: pack_rollout_group(group)[1] for group in groups
        },
    )
    packer = config.microbatch_packer.build(
        num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
        max_context_length=max_context_length,
        pad_id=0,
        temperature=1.0,
        vision_processor=None,
    )
    columns = [
        list(
            MicrobatchGroupReader(
                num_microbatch_groups=len(column.microbatch_groups),
                replica_group=None,
                column=column,
                training_sample_builder=config.training_sample_builder.build(),
                microbatch_packer=packer,
            )
        )
        for column in Controller._replica_columns(controller, plan)
    ]
    return [
        [
            [column[group_index][row_index] for column in columns]
            for row_index in range(len(group))
        ]
        for group_index, group in enumerate(plan.microbatch_groups)
    ]


def _num_input_tokens(specs: list[TrainingSampleSpec]) -> int:
    """The non-padding positions the specs' samples fill in a packed microbatch: each sample's
    tokens but the last."""
    return sum(spec.num_tokens - 1 for spec in specs)


def _specs(batcher: Batcher, group: RolloutGroup) -> list[TrainingSampleSpec]:
    """The training sample specs the batcher plans `group` with."""
    return batcher._training_sample_builder.decide(
        rollout_group=group,
        turn_lengths=tuple(map(rollout_turn_lengths, group.rollouts)),
        num_mtp_layers=0,
    ).training_sample_specs


def _metric_value(batch, key: str) -> float:
    metric = next(metric for metric in batch.metrics if metric.key == key)
    assert isinstance(metric.value, m.NoReduce)
    return metric.value.value


def _untrainable_group(group_id: int) -> RolloutGroup:
    """A failed group: no rollouts, no training samples."""
    return RolloutGroup(group_id=group_id, rollouts=[])


def _build_batcher(
    *, num_prompts_per_train_step: int, num_mtp_layers: int = 0
) -> Batcher:
    config = Batcher.Config(
        microbatch_packer=MicrobatchPacker.Config(num_mtp_layers=num_mtp_layers)
    )
    return config.build(
        num_tokens_per_microbatch_per_dp_rank=16384,
        max_context_length=2048,
        num_prompts_per_train_step=num_prompts_per_train_step,
        dp_degree=1,
    )


def test_batcher_counts_trainable_groups_not_rollouts() -> None:
    # Target is 2 GROUPS. A single group with many rollouts is not a full batch; two groups are,
    # regardless of how many rollouts each contributes.
    batcher = _build_batcher(num_prompts_per_train_step=2)
    batch, group_is_trainable = _add_group(batcher, _trainable_group(0, num_samples=8))
    assert batch is None
    assert group_is_trainable
    batch, group_is_trainable = _add_group(batcher, _trainable_group(1, num_samples=1))
    assert batch is not None
    assert group_is_trainable


def test_batcher_packs_groups_in_id_order_regardless_of_arrival() -> None:
    batcher = _build_batcher(num_prompts_per_train_step=2)
    late_group = _trainable_group(7, num_samples=1)
    early_group = _trainable_group(3, num_samples=1)
    late_group.rollouts[0].turns[0].min_policy_version = 7
    early_group.rollouts[0].turns[0].min_policy_version = 3

    # g7 finishes first; the packed batch still lists g3 before g7.
    _add_group(batcher, late_group)
    batch, _ = _add_group(batcher, early_group)

    assert batch is not None
    assert batch.min_policy_versions == [3, 7]
    assert batch.group_ids == [3, 7]


def test_batcher_carries_metric_only_groups_until_trainable_batch() -> None:
    # Metric-only (empty) groups do not count toward the target and cannot form a zero-token batch;
    # they ride along until a trainable group completes the batch.
    batcher = _build_batcher(num_prompts_per_train_step=1)
    metric_only = _untrainable_group(0)
    assert _add_group(batcher, metric_only) == (
        None,
        False,
    )
    batch, group_is_trainable = _add_group(batcher, _trainable_group(1, num_samples=2))
    assert batch is not None
    assert group_is_trainable
    assert batch.global_loss_token_counts[0] > 0
    assert batch.global_routing_token_counts.shape == (1,)
    assert batch.group_ids == [0, 1]
    # Only group 1 trains; the controller released group 0 when the batcher reported it untrainable.
    assert batch.trainable_group_ids == [1]


def test_batcher_prepares_per_depth_mtp_token_counts() -> None:
    batcher = _build_batcher(num_prompts_per_train_step=1, num_mtp_layers=2)
    batch, _ = _add_group(batcher, _trainable_group(1, num_samples=2))

    assert batch is not None
    assert batch.global_loss_token_counts.shape == (3,)
    assert batch.global_routing_token_counts.shape == (3,)


def test_batcher_warns_after_each_batch_of_untrainable_groups(
    caplog: pytest.LogCaptureFixture,
) -> None:
    batcher = _build_batcher(num_prompts_per_train_step=2)

    with caplog.at_level(logging.WARNING):
        _add_group(batcher, _untrainable_group(0))
        _add_group(batcher, _untrainable_group(1))

    assert (
        "Consecutive untrainable batches: 1/10 "
        "(2 rollout groups produced no trainable samples)."
    ) in caplog.text


def test_batcher_resets_no_progress_count_on_trainable_group(
    caplog: pytest.LogCaptureFixture,
) -> None:
    batcher = _build_batcher(num_prompts_per_train_step=2)

    with caplog.at_level(logging.WARNING):
        _add_group(batcher, _untrainable_group(0))
        _add_group(batcher, _untrainable_group(1))
        _add_group(batcher, _trainable_group(2, num_samples=1))
        caplog.clear()
        _add_group(batcher, _untrainable_group(3))

    assert "zero-output batch equivalents" not in caplog.text


def test_batcher_raises_at_consecutive_untrainable_group_limit() -> None:
    batcher = _build_batcher(num_prompts_per_train_step=2)

    for group_id in range(19):
        _add_group(batcher, _untrainable_group(group_id))

    with pytest.raises(RuntimeError, match="10 consecutive untrainable batches"):
        _add_group(batcher, _untrainable_group(19))


def test_dp_assignment_avoids_all_padding_ranks_when_possible() -> None:
    # Five two-token samples need four rank inputs across two microbatches.
    # Redistributing one sample keeps every rank input trainable.
    batcher = Batcher.Config(max_num_documents=4).build(
        num_tokens_per_microbatch_per_dp_rank=4,
        max_context_length=2,
        num_prompts_per_train_step=1,
        dp_degree=2,
    )
    plan, group_is_trainable = _add_group(batcher, _trainable_group(0, num_samples=5))
    assert plan is not None
    assert group_is_trainable
    cells = [
        specs for group in plan.microbatch_groups for row in group for specs in row
    ]
    assert len(cells) == 4  # 2 microbatches x 2 ranks
    assert sum(map(len, cells)) == 5
    for specs in cells:
        assert sum(spec.loss_token_counts[0] for spec in specs) > 0


def test_batcher_uses_flat_rank_capacity_and_reports_padding() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=1,
    )
    plan, group_is_trainable = _add_group(
        batcher, _variable_length_group(0, token_lengths=[4, 4, 3])
    )

    assert plan is not None
    assert group_is_trainable
    # 3 + 3 + 2 input tokens fill the rank's 8 in one microbatch; rows of 4 would need two.
    [[[specs]]] = plan.microbatch_groups
    assert sorted(spec.num_tokens for spec in specs) == [3, 4, 4]
    assert _num_input_tokens(specs) == 8
    torch.testing.assert_close(plan.global_loss_token_counts, torch.tensor([8]))
    assert _metric_value(plan, "train_batch/padding_frac") == 0.0


def test_flat_rank_packing_preserves_padding_mask() -> None:
    config = Batcher.Config(
        microbatch_packer=MicrobatchPacker.Config(per_sample_pad_multiple=4)
    )
    batcher_kwargs = dict(
        num_tokens_per_microbatch_per_dp_rank=8, max_context_length=4, dp_degree=1
    )
    batcher = config.build(num_prompts_per_train_step=1, **batcher_kwargs)
    group = _variable_length_group(0, token_lengths=[4])
    plan, _ = _add_group(batcher, group)

    assert plan is not None
    [[[microbatch]]] = _leader_microbatch_groups(
        plan, [group], config=config, **batcher_kwargs
    )
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
    )
    plan, group_is_trainable = _add_group(
        batcher, _variable_length_group(0, token_lengths=[5, 4, 4, 3])
    )

    assert plan is not None
    assert group_is_trainable
    assert [
        [_num_input_tokens(specs) for specs in row] for (row,) in plan.microbatch_groups
    ] == [[6, 6]]


def test_batcher_fills_new_bin_from_multiple_heaviest_bins() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=40,
        max_context_length=40,
        num_prompts_per_train_step=1,
        dp_degree=1,
    )
    samples = _specs(batcher, _variable_length_group(0, token_lengths=[11] * 12))
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
    )
    samples = _specs(batcher, _variable_length_group(0, token_lengths=[6, 6]))
    bins = [[samples[0]], [samples[1]]]

    batcher._expand_bins_by_splitting(bins, target_num_bins=4)

    assert bins == [[samples[0]], [samples[1]], [], []]


def test_batcher_splits_sorts_and_zigzags_by_attention_workload() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=10,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=2,
    )
    samples = _specs(
        batcher,
        _variable_length_group(
            0,
            # Effective lengths are [6, 6, 6, 4, 4, 4]. FFD finds three bins,
            # then LPT repacks into four DP inputs.
            token_lengths=[7, 7, 7, 5, 5, 5],
        ),
    )

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
    )
    samples = _specs(
        batcher,
        _variable_length_group(
            0,
            # Effective lengths are [10, 8, 6, 4]. The document limit forces each
            # sample into its own bin, with padding-aware workloads [100, 68, 52, 52].
            token_lengths=[11, 9, 7, 5],
        ),
    )

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
    )
    plan, _ = _add_group(
        batcher, _variable_length_group(0, token_lengths=[2, 2, 4, 2, 2, 4])
    )

    assert plan is not None
    assert len(plan.microbatch_groups) == 3
    assert all(len(specs) <= 3 for ((specs,),) in plan.microbatch_groups)
    # 10 input tokens in 3 microbatches of 6.
    assert _metric_value(plan, "train_batch/padding_frac") == pytest.approx(4 / 9)


def test_document_limit_applies_to_each_local_microbatch() -> None:
    # A local microbatch is capped at three documents. The batcher must keep all
    # five documents and distribute them 2 + 3 across two microbatches.
    batcher = Batcher.Config(max_num_documents=3).build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=1,
    )
    plan, group_is_trainable = _add_group(batcher, _trainable_group(0, num_samples=5))

    assert plan is not None
    assert group_is_trainable
    assert sorted(len(specs) for ((specs,),) in plan.microbatch_groups) == [2, 3]


def test_document_limit_can_be_smaller_than_rows_per_microbatch() -> None:
    batcher = Batcher.Config(max_num_documents=1).build(
        num_tokens_per_microbatch_per_dp_rank=4,
        max_context_length=2,
        num_prompts_per_train_step=1,
        dp_degree=1,
    )
    plan, _ = _add_group(batcher, _trainable_group(0, num_samples=2))

    assert plan is not None
    assert [len(specs) for ((specs,),) in plan.microbatch_groups] == [1, 1]


def test_batcher_filters_training_samples_longer_than_context() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=4,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=1,
    )
    pending, group_is_trainable = _add_group(
        batcher, _variable_length_group(0, token_lengths=[6])
    )
    assert pending is None
    assert not group_is_trainable

    batch, _ = _add_group(batcher, _trainable_group(1, num_samples=1))
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
        )


def test_compute_perf_ratio_metrics_reads_flushed_means() -> None:
    time_metrics = [
        m.Metric("timing/step/total", m.Mean.from_list([2.0])),
        m.Metric("timing/step/wait_for_training_batch", m.Mean.from_list([0.25])),
        m.Metric("timing/step/send_replica_columns", m.Mean.from_list([0.25])),
        m.Metric("timing/step/forward_backward", m.Mean.from_list([0.5])),
        m.Metric("timing/step/optimizer", m.Mean.from_list([0.5])),
        m.Metric(
            "timing/step/blocking_trainer_push_model_state_dict",
            m.Mean.from_list([0.0]),
        ),
        m.Metric(
            "timing/step/blocking_generator_pull_model_state_dict",
            m.Mean.from_list([0.25]),
        ),
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
    assert ratios["perf/trainer/step_time_ratio/send_replica_columns"] == 0.125
    # 2.0 s - (0.25 + 0.25 + 0.5 + 0.5 + 0.0 + 0.25) s: the column send is accounted for.
    assert ratios["perf/trainer/step_time_ratio/unaccounted"] == 0.125


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
        await buffer.finalize_work(_finished_group(0))
        await buffer.take_finalized()

        waiter = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not waiter.done()

        await buffer.release_slots([0], reason="trained")
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
        )

        if not await buffer.wait_for_slot():
            raise RuntimeError("buffer closed unexpectedly")
        await buffer.add_work(RolloutGroupWork(group_id=0, sample=object()))

        await buffer.release_slots([0], reason="untrainable_group")
        assert batcher.add_group(summary=_finished_group(0).summary) == (
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


def _finished_group(group_id: int) -> FinishedRolloutGroup:
    return FinishedRolloutGroup.from_group(RolloutGroup(group_id=group_id, rollouts=[]))


async def _finalize(buffer: RolloutGroupWorkBuffer, group_id: int) -> None:
    await buffer.finalize_work(_finished_group(group_id))


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


def _finalize_one_group_with_rollout_loop(
    run_group_rollouts, recorder
) -> FinishedRolloutGroup:
    """Run `_rollout_loop` on one admitted group and return the group it finalizes."""

    async def run() -> FinishedRolloutGroup:
        buffer = _buffer(capacity=1, window_size=None)
        await _admit(buffer, 0)
        controller = object.__new__(Controller)
        controller.config = SimpleNamespace(
            async_loop=SimpleNamespace(num_samples_per_prompt=1)
        )
        controller._sampling = SamplingConfig()
        controller._rollouter = SimpleNamespace(run_group_rollouts=run_group_rollouts)
        controller.rollout_recorder = recorder
        controller.generator_router = SimpleNamespace(
            release_groups=SimpleNamespace(call_one=AsyncMock())
        )
        loop = asyncio.create_task(
            controller._rollout_loop(
                group_buffer=buffer, generate_fn=None, release_session_fn=None
            )
        )
        group = await buffer.take_finalized()
        await buffer.close()
        await loop
        return group

    return asyncio.run(asyncio.wait_for(run(), timeout=10))


def test_rollout_loop_appends_the_worker_lines_and_keeps_the_group_metrics() -> None:
    """The rollout worker computes the group's rollout metrics and recorder lines; the
    controller appends the lines and passes the group, with its metrics, to the batcher."""
    run_group_kwargs = []

    async def run_group_rollouts(*, group_id: int, **kwargs) -> FinishedRolloutGroup:
        run_group_kwargs.append(kwargs)
        group = RolloutGroup(
            group_id=group_id,
            rollouts=[],
            metrics=[m.Metric("timing/hop/run_group/wall", m.Mean(0.1))],
        )
        return FinishedRolloutGroup.from_group(
            group, recorder_lines='{"group_id": 0}\n'
        )

    recorder = Mock()
    finished = _finalize_one_group_with_rollout_loop(run_group_rollouts, recorder)

    assert [metric.key for metric in finished.summary.group.metrics] == [
        "timing/hop/run_group/wall"
    ]
    # The buffer does not keep the lines the recorder already has.
    assert finished.recorder_lines == ""
    assert [kwargs["for_training"] for kwargs in run_group_kwargs] == [True]
    recorder.append_lines.assert_called_once_with('{"group_id": 0}\n')
    recorder.record.assert_not_called()


def test_rollout_loop_turns_a_failed_group_into_one_failure_metric() -> None:
    async def run_group_rollouts(**kwargs) -> FinishedRolloutGroup:
        raise RuntimeError("scoring failed")

    recorder = Mock()
    finished = _finalize_one_group_with_rollout_loop(run_group_rollouts, recorder)
    group = finished.summary.group

    assert (group.group_id, group.rollouts) == (0, [])
    assert m.MetricsProcessor._aggregate_metrics(group.metrics) == {
        "rollout/group_failures/sum": 1.0
    }
    recorder.append_lines.assert_not_called()


def test_batcher_loop_times_each_add_group_call() -> None:
    """Each add_group call's wall and handler times ride with the next batch."""
    batch = SimpleNamespace(metrics=[])

    async def add_group(*, summary):
        await asyncio.sleep(0.03)
        # The batcher spends 10 ms of each call in its handler; the second call fills a batch.
        return (batch if summary.group.group_id == 1 else None), True, 0.01

    async def run() -> object:
        buffer = _buffer(capacity=2, window_size=None)
        for group_id in (0, 1):
            await _admit(buffer, group_id)
            await buffer.claim_next()
            await _finalize(buffer, group_id)
        controller = object.__new__(Controller)
        controller._group_buffer = buffer
        controller._packed_groups_by_id = {}
        controller.batcher = SimpleNamespace(
            add_group=SimpleNamespace(call_one=add_group)
        )
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.create_task(controller._batcher_loop(training_batch_queue=queue))
        packed = await queue.get()
        await buffer.close()
        await loop
        return packed

    packed = asyncio.run(asyncio.wait_for(run(), timeout=10))

    wall = next(
        metric.value
        for metric in packed.metrics
        if metric.key == "timing/hop/add_group/wall"
    )
    assert wall.count == 2  # both calls, not only the one that returned the batch
    metrics = m.MetricsProcessor._aggregate_metrics(packed.metrics)
    assert metrics["timing/hop/add_group/handler/mean"] == pytest.approx(0.01)
    assert metrics["timing/hop/add_group/wall/mean"] >= 0.03
    assert metrics["timing/hop/add_group/transfer/mean"] == pytest.approx(
        metrics["timing/hop/add_group/wall/mean"] - 0.01
    )
    assert (
        metrics["timing/hop/add_group/transfer/max"]
        >= metrics["timing/hop/add_group/transfer/mean"]
    )


def test_batcher_loop_keeps_only_the_packed_bytes_of_trainable_groups() -> None:
    """Until its step, the controller keeps a trainable group's packed bytes, not its summary
    (the per-turn Python objects the batcher plans from). Untrainable groups are not kept."""
    finished_groups = [_finished_group(group_id) for group_id in (0, 1)]

    async def add_group(*, summary):
        return None, summary.group.group_id == 0, 0.0

    async def run() -> dict[int, PackedRolloutGroup]:
        buffer = _buffer(capacity=2, window_size=None)
        for finished in finished_groups:
            await _admit(buffer, finished.group_id)
            await buffer.claim_next()
            await buffer.finalize_work(finished)
        controller = object.__new__(Controller)
        controller._group_buffer = buffer
        controller._packed_groups_by_id = {}
        controller.batcher = SimpleNamespace(
            add_group=SimpleNamespace(call_one=add_group)
        )
        loop = asyncio.create_task(
            controller._batcher_loop(training_batch_queue=asyncio.Queue())
        )
        await _settle()
        await buffer.close()
        await loop
        return controller._packed_groups_by_id

    packed_groups_by_id = asyncio.run(asyncio.wait_for(run(), timeout=10))

    assert packed_groups_by_id == {0: finished_groups[0].packed}


async def _settle() -> None:
    """Let every ready task run until it blocks again."""
    for _ in range(10):
        await asyncio.sleep(0)


def test_release_slots_frees_only_the_named_groups_slots() -> None:
    async def run() -> list[m.Metric]:
        buffer = _buffer(capacity=3, window_size=None)
        await _admit(buffer, 0)
        await _admit(buffer, 1)
        # Each failure is a caller bug that would otherwise free or keep the wrong slots.
        with pytest.raises(RuntimeError, match="already holds an active slot"):
            await buffer.add_work(RolloutGroupWork(group_id=0, sample=object()))
        with pytest.raises(RuntimeError, match=r"without a slot: \[5\]"):
            await buffer.release_slots([5], reason="trained")
        with pytest.raises(RuntimeError, match=r"repeated: \[1\]"):
            await buffer.release_slots([1, 1], reason="trained")
        await buffer.release_slots([1], reason="untrainable_group")
        with pytest.raises(RuntimeError, match=r"without a slot: \[1\]"):
            await buffer.release_slots([1], reason="trained")
        return buffer.metrics()

    metrics = {metric.key: metric.value.value for metric in asyncio.run(run())}
    # Group 0 still holds its slot.
    assert metrics["rollout_buffer/available_active_slots"] == 2


def test_add_work_wakes_one_claim_next_waiter(monkeypatch) -> None:
    """A new group wakes one idle rollout loop. Waking all (S+1) x P loops, each of which then
    goes back to wait, saturated the controller at P=1024."""
    num_waits = 0
    wait = asyncio.Condition.wait

    async def counting_wait(self) -> bool:
        nonlocal num_waits
        num_waits += 1
        return await wait(self)

    monkeypatch.setattr(asyncio.Condition, "wait", counting_wait)

    async def run() -> None:
        buffer = _buffer(capacity=8, window_size=None)
        claimers = [asyncio.create_task(buffer.claim_next()) for _ in range(5)]
        await _settle()
        assert num_waits == 5
        await _admit(buffer, 0)
        await _settle()
        [claimed] = [claimer.result() for claimer in claimers if claimer.done()]
        assert claimed.group_id == 0
        # The four other loops never woke, so none of them waited again.
        assert num_waits == 5
        await buffer.close()
        await asyncio.gather(*claimers)

    asyncio.run(asyncio.wait_for(run(), timeout=10))


def test_close_unblocks_every_waiter() -> None:
    async def run() -> list:
        buffer = _buffer(capacity=1, window_size=None)
        await _admit(buffer, 0)
        await buffer.claim_next()  # g0 CLAIMED: nothing to claim or take, and no free slot
        waiters = [
            *(asyncio.create_task(buffer.claim_next()) for _ in range(3)),
            asyncio.create_task(buffer.wait_for_slot()),
            asyncio.create_task(buffer.take_finalized()),
        ]
        await _settle()
        assert not any(waiter.done() for waiter in waiters)
        await buffer.close()
        return await asyncio.gather(*waiters)

    assert asyncio.run(asyncio.wait_for(run(), timeout=10)) == [
        None,
        None,
        None,
        False,
        None,
    ]


async def _block_forever(*args, **kwargs) -> None:
    await asyncio.Event().wait()


async def _train_one_step(*args, **kwargs) -> None:
    await asyncio.sleep(0)


def _run_with_generator_watcher(
    watcher,
    *,
    trainer_loop=_block_forever,
    validate=None,
    validation=None,
    on_close=None,
) -> Controller:
    """Run Controller.run with stub loops: `watcher` waits on the engine loops, `trainer_loop` trains, the rest block."""
    controller = object.__new__(Controller)
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
                build=lambda **kwargs: SimpleNamespace(
                    close=AsyncMock(side_effect=on_close)
                )
            ),
            validation=validation or ValidationConfig(),
        ),
        trainer=SimpleNamespace(
            training=SimpleNamespace(num_tokens_per_microbatch_per_dp_rank=8)
        ),
    )
    controller.start_step = 0
    controller._validation_tasks = []
    controller._packed_groups_by_id = {}
    controller._has_pending_validation = False
    controller.metrics_processor = Mock()
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
            validate=validate or AsyncMock(return_value=[]),
            _make_generate_fn=lambda self: None,
            _data_input_loop=_block_forever,
            _rollout_loop=_block_forever,
            _batcher_loop=_block_forever,
            _trainer_loop=trainer_loop,
        ),
    ):
        asyncio.run(asyncio.wait_for(controller.run(), timeout=10))
    return controller


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


@pytest.mark.parametrize(
    "step_0_validation, validate_after_training, started_steps, logged_steps",
    [
        # Ends before training does: the call after training logs it at step 1.
        ("ends", False, [0], [1]),
        # Ends while run()'s finally awaits group_buffer.close(): also logged at step 1.
        ("ends_during_close", False, [0], [1]),
        # Still running: run() cancels it (else run() hangs) and logs nothing.
        ("runs", False, [0], []),
        # Cancelled, then the validation after training runs and is logged at step 1.
        ("runs", True, [0, 1], [1]),
        # Failed: the call after training raises its error.
        ("fails", False, [0], None),
    ],
)
def test_run_ends_validation(
    step_0_validation, validate_after_training, started_steps, logged_steps
) -> None:
    """1 training step: what run() does with the step-0 validation and the validation after training."""
    started = []
    close_started = asyncio.Event()

    async def validate(self, *, step: int) -> list[m.Metric]:
        started.append(step)
        if step == 0 and step_0_validation == "ends_during_close":
            await close_started.wait()
        if step == 0 and step_0_validation == "runs":
            await asyncio.Event().wait()
        if step == 0 and step_0_validation == "fails":
            raise RuntimeError("validation failed")
        return [m.Metric(f"started_{step}", m.Mean(1.0))]

    async def close() -> None:
        close_started.set()
        await asyncio.sleep(0)  # the validation returns

    def run() -> Controller:
        return _run_with_generator_watcher(
            _block_forever,
            trainer_loop=_train_one_step,
            validate=validate,
            validation=ValidationConfig(
                validate_after_training=validate_after_training
            ),
            on_close=close,
        )

    if step_0_validation == "fails":
        with pytest.raises(RuntimeError, match="validation failed"):
            run()
    else:
        controller = run()
        assert [
            call.kwargs["step"]
            for call in controller.metrics_processor.log.call_args_list
        ] == logged_steps
    assert started == started_steps


@pytest.mark.parametrize(
    "start_step, validate_at_step_0, defer_new_if_not_finished, started_steps, logged",
    [
        # The validation started at 3 came due at 2; the one started at 6 covers steps 4 and 6.
        (0, True, True, [0, 3, 6], [(3, "started_0"), (6, "started_3")]),
        (
            0,
            True,
            False,
            [0, 2, 4, 6],
            [(3, "started_0"), (5, "started_2"), (7, "started_4")],
        ),
        (0, False, True, [2, 5], [(5, "started_2"), (8, "started_5")]),
        # Resumed at step 2, a multiple of freq: a validation starts there. validate_at_step_0 is for step 0 only.
        (2, False, True, [2, 5], [(5, "started_2"), (8, "started_5")]),
        # Resumed at step 3: the first validation starts at 4, also with validate_at_step_0.
        (3, True, True, [4, 7], [(7, "started_4")]),
    ],
)
def test_validation_schedule(
    start_step, validate_at_step_0, defer_new_if_not_finished, started_steps, logged
) -> None:
    """freq=2, 8 steps, each validation ends 3 steps after it starts and is logged at that step.

    No validation starts at step 8, the last step.
    """
    end_events: dict[int, asyncio.Event] = {}

    async def validate(*, step: int) -> list[m.Metric]:
        end_events[step] = asyncio.Event()
        await end_events[step].wait()
        return [m.Metric(f"started_{step}", m.Mean(1.0))]

    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(
            num_training_steps=8,
            validation=ValidationConfig(
                freq=2,
                validate_at_step_0=validate_at_step_0,
                defer_new_if_not_finished=defer_new_if_not_finished,
            ),
        )
    )
    controller._validation_tasks = []
    controller._has_pending_validation = False
    controller.validate = validate
    controller.metrics_processor = Mock()

    async def run_steps() -> None:
        for step in range(start_step, 9):
            if step - 3 in end_events:
                end_events[step - 3].set()
                await asyncio.sleep(0)  # the validation returns
            controller._maybe_start_validation(step=step)
            await asyncio.sleep(0)  # a new validation starts

    asyncio.run(run_steps())

    assert list(end_events) == started_steps
    assert [
        (call.kwargs["step"], call.kwargs["metrics"][0].key)
        for call in controller.metrics_processor.log.call_args_list
    ] == logged


def test_generate_fn_forwards_the_session_and_releases_on_failure() -> None:
    """A failed call releases its reserved load, or the router would keep counting it."""

    async def generate_call(*args, **kwargs) -> None:
        # The load stays reserved while the generator runs the call.
        release.assert_not_called()
        raise RuntimeError("generator failed")

    generate_call = AsyncMock(side_effect=generate_call)
    reserve = AsyncMock(
        return_value=SimpleNamespace(generate=SimpleNamespace(call_one=generate_call))
    )
    release = Mock()
    controller = object.__new__(Controller)
    controller.generator_router = SimpleNamespace(
        reserve=SimpleNamespace(call_one=reserve),
        release_reservation=SimpleNamespace(broadcast=release),
    )
    generate = controller._make_generate_fn()
    prompt = Prompt(token_ids=[1, 2])

    with pytest.raises(RuntimeError, match="generator failed"):
        asyncio.run(
            generate(prompt, request_id="r0", group_id=0, routing_session_id="s0")
        )

    # The session reaches both routing layers: the router and the generator's DP routing.
    reserve.assert_awaited_once_with("r0", routing_session_id="s0")
    generate_call.assert_awaited_once_with(
        prompt,
        request_id="r0",
        group_id=0,
        routing_session_id="s0",
        sampling_config=None,
    )
    release.assert_called_once_with("r0")


def test_validation_renames_turn_metrics_and_uses_fresh_group_ids() -> None:
    """Validation prefixes its turns' metrics with `validation_` (`generator/x` -> `validation_generator/x`,
    `math_verify/x` -> `validation_math_verify/x`). Two validations never share a group id."""

    async def run_group_rollouts(
        *, group_id: int, for_training: bool, **kwargs
    ) -> FinishedRolloutGroup:
        # Validation gets the full group and logs it in the controller.
        assert not for_training
        turn = RolloutTurn(
            rollout_id=RolloutTurnID(group_id=group_id, rollout_id=0, turn_id=0),
            prompt_prefix_len=0,
            prompt_delta_token_ids=np.array([1], dtype=np.int32),
            completion_token_ids=np.array([2], dtype=np.int32),
            completion_logprobs=np.array([-0.1], dtype=np.float32),
            min_policy_version=0,
            max_policy_version=0,
            metrics=[
                m.Metric("generator/queue_time_ms", m.Mean(2.0)),
                m.Metric("math_verify/timeouts", m.Sum(1.0)),
            ],
        )
        rollout = Rollout(
            group_id=group_id,
            rollout_id=0,
            status=RolloutStatus.COMPLETED,
            reward=0.0,
            turns=[turn],
        )
        group = RolloutGroup(group_id=group_id, rollouts=[rollout])
        return FinishedRolloutGroup.from_group(group)

    controller = object.__new__(Controller)
    controller._validation_group_ids = itertools.count(start=-1, step=-1)
    controller._make_generate_fn = lambda: None
    controller._make_release_session_fn = lambda: None
    controller._rollouter = SimpleNamespace(run_group_rollouts=run_group_rollouts)
    release_groups = AsyncMock()
    controller.generator_router = SimpleNamespace(
        release_groups=SimpleNamespace(call_one=release_groups)
    )

    async def run_two_validations() -> list[m.Metric]:
        for step in (0, 2):
            _, metrics = await controller._collect_validation_rollouts(
                samples=["prompt 1", "prompt 2"], sampling=SamplingConfig(), step=step
            )
        return metrics

    keys = m.MetricsProcessor._aggregate_metrics(asyncio.run(run_two_validations()))
    assert [call.args[0] for call in release_groups.call_args_list] == [
        [-1, -2],
        [-3, -4],
    ]
    assert keys["validation_generator/queue_time_ms/mean"] == 2.0
    assert keys["validation_math_verify/timeouts/sum"] == 2.0
    assert "generator/queue_time_ms/mean" not in keys
    assert "math_verify/timeouts/sum" not in keys
