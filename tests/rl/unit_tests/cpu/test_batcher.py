# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for fixed-bin LPT packing and its FFD fallback."""

import random

import numpy as np
import pytest
import torch
from torchtitan.rl.rollout.types import RolloutTurnID

from torchtitan.rl.trainer.batcher import Batcher, TrainerStepBatch
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleGroup,
)


def _make_samples(lengths: list[int]) -> list[TrainingSample]:
    return [
        TrainingSample(
            min_policy_version=0,
            max_policy_version=0,
            rollout_id=RolloutTurnID(group_id=0, rollout_id=index, turn_id=0),
            token_ids=np.arange(length + 1),
            loss_mask=np.array([False] + [True] * length),
            logprobs=np.zeros(length + 1, dtype=np.float32),
            advantage=np.zeros(length + 1, dtype=np.float32),
        )
        for index, length in enumerate(lengths)
    ]


def test_padding_workload_matches_existing_full_length_segments() -> None:
    batcher = Batcher.Config(max_num_documents=20).build(
        num_tokens_per_microbatch_per_dp_rank=60,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    long_samples = _make_samples([8] * 6)
    short_samples = _make_samples([1] * 20)

    assert batcher._attention_workload(long_samples) == 6 * 8**2 + 10**2 + 2**2
    assert batcher._attention_workload(short_samples) == 20 + 4 * 10**2
    assert batcher._attention_workload([]) == 6 * 10**2

    for samples in (long_samples, short_samples, []):
        microbatch = batcher._pack_training_samples(samples)
        num_padding_segments = int(
            ((microbatch.positions == 0) & microbatch.padding_mask).sum().item()
        )
        assert num_padding_segments <= 6


def test_lpt_uses_padding_workload_for_full_length_documents() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=48,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([8] * 3 + [2] * 16)

    assignments = batcher._assign_training_samples_to_microbatches(samples)

    assert [
        batcher._attention_workload(rank_samples)
        for row in assignments
        for rank_samples in row
    ] == [288, 264]
    assert sorted(
        sum(map(batcher.num_tokens_to_pack, rank_samples))
        for row in assignments
        for rank_samples in row
    ) == [26, 30]


def test_lpt_does_not_leave_a_full_length_document_in_an_expensive_bin() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=20,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([10, 5, 5, 5, 5])

    assignments = batcher._assign_training_samples_to_microbatches(samples)

    assert [
        batcher._attention_workload(rank_samples)
        for row in assignments
        for rank_samples in row
    ] == [150, 100]


def test_lpt_rebalances_ffd_bins_with_document_limit() -> None:
    batcher = Batcher.Config(max_num_documents=3).build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([4, 4, 3, 3, 3, 3, 2, 2])

    assignments = batcher._assign_training_samples_to_microbatches(samples)

    assert [
        [batcher._attention_workload(rank_samples) for rank_samples in microbatch]
        for microbatch in assignments
    ] == [[26, 26], [22, 22]]
    assert all(
        len(rank_samples) <= 3
        and sum(map(batcher.num_tokens_to_pack, rank_samples)) <= 8
        for microbatch in assignments
        for rank_samples in microbatch
    )
    assert sorted(
        sample.rollout_id.rollout_id
        for microbatch in assignments
        for rank_samples in microbatch
        for sample in rank_samples
    ) == list(range(len(samples)))


def test_lpt_falls_back_when_ffd_fits_but_lpt_cannot() -> None:
    batcher = Batcher.Config(max_num_documents=3).build(
        num_tokens_per_microbatch_per_dp_rank=6,
        max_context_length=3,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([3] * 8 + [2] * 9)
    assert batcher._pack_with_lpt(samples, target_num_bins=7) is None

    assignments = batcher._assign_training_samples_to_microbatches(samples)

    assert len(assignments) == 7
    assert [
        sum(batcher.num_tokens_to_pack(sample) for sample in rank_samples)
        for (rank_samples,) in assignments
    ] == [6] * 7
    assert (
        sorted(len(rank_samples) for (rank_samples,) in assignments)
        == [2] * 4 + [3] * 3
    )


def test_three_pp_microbatches_balance_rank_costs_within_step() -> None:
    batcher = Batcher.Config(max_num_documents=1).build(
        num_tokens_per_microbatch_per_dp_rank=10,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([10, 9, 8, 7, 6, 5])

    assignments = batcher._assign_training_samples_to_microbatches(
        samples, num_pp_microbatches=3
    )

    assert len(assignments) == 3
    workloads = [
        [batcher._attention_workload(rank_samples) for rank_samples in microbatch]
        for microbatch in assignments
    ]
    assert workloads == [[100, 82], [58, 68], [50, 52]]
    assert [sum(rank_workloads) for rank_workloads in zip(*workloads)] == [208, 202]


def test_packing_preserves_all_samples_and_capacity_constraints() -> None:
    rng = random.Random(42)
    for _ in range(40):
        seq_len = rng.randint(3, 12)
        num_rows = rng.randint(1, 3)
        dp_degree = rng.randint(1, 4)
        num_pp_microbatches = rng.choice([1, 3])
        max_num_documents = rng.choice([None, 1, 3, 6])
        batcher = Batcher.Config(max_num_documents=max_num_documents).build(
            num_tokens_per_microbatch_per_dp_rank=num_rows * seq_len,
            max_context_length=seq_len,
            num_prompts_per_train_step=1,
            dp_degree=dp_degree,
            pad_id=0,
            temperature=1.0,
        )
        samples = _make_samples(
            [rng.randint(1, seq_len) for _ in range(rng.randint(1, 30))]
        )

        assignments = batcher._assign_training_samples_to_microbatches(
            samples, num_pp_microbatches=num_pp_microbatches
        )

        assert len(assignments) % num_pp_microbatches == 0
        assert all(len(row) == dp_degree for row in assignments)
        for row in assignments:
            for rank_samples in row:
                assert sum(map(batcher.num_tokens_to_pack, rank_samples)) <= (
                    num_rows * seq_len
                )
                if max_num_documents is not None:
                    assert len(rank_samples) <= max_num_documents
        assert sorted(
            sample.rollout_id.rollout_id
            for row in assignments
            for rank_samples in row
            for sample in rank_samples
        ) == list(range(len(samples)))


def test_microbatch_carries_the_sampling_temperature_into_the_loss() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=0.5,
    )
    microbatch = batcher._pack_training_samples(_make_samples([3]))

    temperature = microbatch.as_input_dict()["loss_kwargs"]["temperature"]
    assert temperature.dtype == torch.float32
    assert temperature.tolist() == [0.5] * 8


def test_microbatch_moves_loss_kwargs_with_the_model_inputs() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    microbatch = batcher._pack_training_samples(_make_samples([3]))

    input_dict = microbatch.to_input_dict("meta")

    assert input_dict["input"].device.type == "meta"
    assert set(input_dict["loss_kwargs"]) == {
        "generator_logprobs",
        "temperature",
        "loss_mask",
        "advantages",
    }
    for value in input_dict["loss_kwargs"].values():
        assert value.device.type == "meta"


class _SizedSample:
    """Stands in for a TrainingSample in packing-only tests: packs `num_tokens` tokens."""

    __slots__ = ("num_tokens",)

    def __init__(self, num_tokens: int) -> None:
        self.num_tokens = num_tokens


def _record_streamed_step(
    batcher: Batcher, groups: list[list[_SizedSample]]
) -> tuple[list[list[list[_SizedSample]]], list[list[list[_SizedSample]]]]:
    """Add one step's groups to a streaming batcher of `_SizedSample`s; return the streamed batches'
    rows and the step's last batch's rows, each `[num_rows][dp_degree]` bins, as packing assigns them.
    """
    num_pp_microbatches = batcher._num_pp_microbatches
    streamed_rows: list[list[list[_SizedSample]]] = []
    last_rows: list[list[list[_SizedSample]]] = []

    def record_pack(*, streamed_training_samples=None):
        if streamed_training_samples is None:
            batcher._take_groups()
            last_rows.extend(
                batcher._assign_training_samples_to_microbatches(
                    batcher._unsent_training_samples,
                    num_pp_microbatches=num_pp_microbatches,
                )
            )
            batcher._unsent_training_samples = []
            return TrainerStepBatch(
                microbatch_groups=[],
                num_loss_tokens=0,
                metrics=[],
                group_ids=[],
                min_policy_versions=[],
            )
        streamed_rows.extend(
            batcher._assign_training_samples_to_microbatches(
                streamed_training_samples, num_pp_microbatches=num_pp_microbatches
            )
        )
        return TrainerStepBatch(
            microbatch_groups=[],
            num_loss_tokens=0,
            metrics=[],
            group_ids=[],
            min_policy_versions=[],
            ends_step=False,
        )

    batcher.num_tokens_to_pack = lambda sample: sample.num_tokens
    batcher._pack_one_training_batch = record_pack
    for group_id, samples in enumerate(groups):
        batch, _ = batcher.add_training_samples(
            training_sample_group=TrainingSampleGroup(
                group_id=group_id, training_samples=samples, metrics=[]
            )
        )
        is_last_group = group_id == len(groups) - 1
        assert batch is None or batch.ends_step == is_last_group
        assert (batch is not None) or not is_last_group
    return streamed_rows, last_rows


def _streaming_batcher(
    *,
    num_tokens_per_microbatch: int,
    max_context_length: int,
    num_groups: int,
    dp_degree: int,
    num_pp_microbatches: int = 1,
    max_num_documents: int | None = None,
) -> Batcher:
    return Batcher.Config(
        stream_microbatches=True, max_num_documents=max_num_documents
    ).build(
        num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch,
        max_context_length=max_context_length,
        num_prompts_per_train_step=num_groups,
        dp_degree=dp_degree,
        num_pp_microbatches=num_pp_microbatches,
        pad_id=0,
        temperature=1.0,
    )


@pytest.mark.parametrize("num_pp_microbatches", [1, 2])
def test_streamed_packing_sends_each_sample_once_per_step(
    num_pp_microbatches: int,
) -> None:
    rng = random.Random(0)
    for _ in range(30):
        seq_len = rng.randint(4, 12)
        num_tokens_per_microbatch = seq_len * rng.randint(1, 3)
        dp_degree = rng.randint(1, 4)
        max_num_documents = rng.choice([None, 2, 4])
        num_groups = rng.randint(1, 6)
        batcher = _streaming_batcher(
            num_tokens_per_microbatch=num_tokens_per_microbatch,
            max_context_length=seq_len,
            num_groups=num_groups,
            dp_degree=dp_degree,
            num_pp_microbatches=num_pp_microbatches,
            max_num_documents=max_num_documents,
        )
        # Two steps on one batcher: each ends at its own last group.
        for _step in range(2):
            groups = [
                [
                    _SizedSample(rng.randint(1, seq_len))
                    for _ in range(rng.randint(1, 8))
                ]
                for _ in range(num_groups)
            ]
            streamed_rows, last_rows = _record_streamed_step(batcher, groups)
            rows = streamed_rows + last_rows
            assert len(streamed_rows) % num_pp_microbatches == 0
            assert len(last_rows) % num_pp_microbatches == 0
            assert all(len(row) == dp_degree for row in rows)
            for row in rows:
                for bin_ in row:
                    assert sum(s.num_tokens for s in bin_) <= num_tokens_per_microbatch
                    if max_num_documents is not None:
                        assert len(bin_) <= max_num_documents
            packed = [id(s) for row in rows for bin_ in row for s in bin_]
            assert sorted(packed) == sorted(id(s) for group in groups for s in group)
            assert batcher._groups_for_next_batch == []
            # Streamed batches carry no empty bin, and the step's last batch keeps one sample per bin
            # of a microbatch group (or every sample, if the step has fewer).
            assert all(bin_ for row in streamed_rows for bin_ in row)
            num_last_batch_samples = sum(len(bin_) for row in last_rows for bin_ in row)
            assert num_last_batch_samples >= min(
                num_pp_microbatches * dp_degree, len(packed)
            )


@pytest.mark.parametrize(
    ("lengths", "sent", "unsent"),
    [
        # The `_pack_ready_microbatches` docstring cases: dp_degree=2, 10 tokens per bin.
        ([6, 6, 4, 4, 5, 3], [4, 4, 6, 6], [3, 5]),
        # Sending both full bins would leave no sample for the step's last batch.
        ([6, 6, 4, 4], None, [4, 4, 6, 6]),
        # Sending 4 of the 5 bins would leave 1 sample; one group fewer leaves 3.
        ([9, 9, 9, 9, 9, 1], [1, 9, 9], [9, 9, 9]),
    ],
)
def test_streaming_sends_full_microbatch_groups_and_holds_back_a_sample_per_bin(
    lengths: list[int], sent: list[int] | None, unsent: list[int]
) -> None:
    batcher = _streaming_batcher(
        num_tokens_per_microbatch=10, max_context_length=10, num_groups=1, dp_degree=2
    )
    batcher.num_tokens_to_pack = lambda sample: sample.num_tokens
    batcher._unsent_training_samples = [_SizedSample(n) for n in lengths]
    sent_samples = []
    batcher._pack_one_training_batch = (
        lambda *, streamed_training_samples: sent_samples.append(
            streamed_training_samples
        )
    )

    batcher._pack_ready_microbatches()

    if sent is None:
        assert sent_samples == []
    else:
        assert sorted(s.num_tokens for s in sent_samples[0]) == sent
    assert sorted(s.num_tokens for s in batcher._unsent_training_samples) == unsent
