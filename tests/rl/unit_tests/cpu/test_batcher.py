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

from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.training_sample_builder import TrainingSample


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

    temperature = microbatch.loss_kwargs()["temperature"]
    assert temperature.dtype == torch.float32
    assert temperature.tolist() == [0.5] * 8


def test_routed_expert_ids_follow_input_ids_with_zero_padding() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([2, 3])
    for sample, first_id in zip(samples, (10, 20), strict=True):
        # One row per input token; uint16, as vLLM returns above 256 experts.
        rows = first_id + np.arange(len(sample.token_ids) - 1, dtype=np.uint16)
        sample.routed_expert_ids = rows.reshape(-1, 1, 1)

    microbatch = batcher._pack_training_samples(samples)

    routed_expert_ids = microbatch.model_kwargs["routed_expert_ids"]
    # Torch can't unpickle uint16 tensors, so the batcher ships int16.
    assert routed_expert_ids.dtype == torch.int16
    assert routed_expert_ids.flatten().tolist() == [10, 11, 20, 21, 22, 0, 0, 0]


def test_microbatch_ships_topk_rows_for_loss_tokens_only() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
        num_topk_logprobs=2,
    )
    first, second = _make_samples([3, 3])
    # Prompt, completion, env reply, completion: two loss tokens, so two top-k rows.
    first.loss_mask = np.array([False, True, False, True])
    first.topk_token_ids = np.array([[5, 6], [9, 4]], dtype=np.int32)
    first.topk_logprobs = np.array([[-0.1, -2.0], [-0.3, -4.0]], dtype=np.float32)
    second.loss_mask = np.array([False, False, True, True])
    second.topk_token_ids = np.array([[1, 2], [3, 7]], dtype=np.int32)
    second.topk_logprobs = np.array([[-0.5, -1.0], [-0.6, -1.1]], dtype=np.float32)

    microbatch = batcher._pack_training_samples([first, second])
    loss_kwargs = microbatch.to_loss_kwargs("cpu")

    assert microbatch.generator_topk_token_ids.shape == (4, 2)
    # Shifted like generator_logprobs: row t holds the top-k that sampled labels[t].
    assert loss_kwargs["generator_topk_token_ids"].tolist() == (
        [[5, 6], [0, 0], [9, 4]] + [[0, 0], [1, 2], [3, 7]] + [[0, 0]] * 2
    )
    assert torch.equal(
        loss_kwargs["generator_topk_logprobs"],
        torch.tensor(
            [[-0.1, -2.0], [0.0, 0.0], [-0.3, -4.0]]
            + [[0.0, 0.0], [-0.5, -1.0], [-0.6, -1.1]]
            + [[0.0, 0.0]] * 2
        ),
    )
    # An empty bin ships no rows but still gives the loss [T, k], so every rank calls it alike.
    empty_microbatch = batcher._pack_training_samples([])
    assert empty_microbatch.generator_topk_token_ids.shape == (0, 2)
    empty_loss_kwargs = empty_microbatch.to_loss_kwargs("cpu")
    assert empty_loss_kwargs["generator_topk_token_ids"].tolist() == [[0, 0]] * 8
    # A wrong row count shifts later rows, so packing checks each sample: 3 + 1 rows for
    # 2 + 2 loss tokens has the right total but is refused.
    first.topk_token_ids = np.array([[5, 6], [9, 4], [8, 8]], dtype=np.int32)
    first.topk_logprobs = np.zeros((3, 2), dtype=np.float32)
    second.topk_token_ids = second.topk_token_ids[:1]
    second.topk_logprobs = second.topk_logprobs[:1]
    with pytest.raises(ValueError, match="3 top-k rows for 2 loss tokens"):
        batcher._pack_training_samples([first, second])


def test_microbatch_has_no_topk_kwargs_by_default() -> None:
    # DAPOLoss and GRPOLoss take no top-k arguments.
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    loss_kwargs = batcher._pack_training_samples(_make_samples([3])).loss_kwargs()
    assert "generator_topk_token_ids" not in loss_kwargs


def test_token_budget_need_not_be_a_multiple_of_the_context() -> None:
    """A 2.5-context budget packs whole samples; padding positions wrap at the context."""
    from torchtitan.models.common.attention.attention import (
        create_varlen_metadata_for_document,
    )

    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=25,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([10, 10, 4])

    assignments = batcher._assign_training_samples_to_microbatches(samples)
    microbatch = batcher._pack_training_samples(samples)
    metadata = create_varlen_metadata_for_document(
        microbatch.positions, padding_mask=microbatch.padding_mask
    )

    assert [len(rank_samples) for row in assignments for rank_samples in row] == [3]
    assert microbatch.input.shape == (25,)
    assert microbatch.positions.tolist()[20:] == [0, 1, 2, 3, 0]
    assert microbatch.padding_mask.tolist()[24] is True
    assert metadata.cu_seq_q.tolist() == [0, 10, 20, 24, 25]


def test_token_budget_must_fit_a_full_context_sample() -> None:
    with pytest.raises(ValueError, match="must be at least max_context_length"):
        Batcher.Config().build(
            num_tokens_per_microbatch_per_dp_rank=9,
            max_context_length=10,
            num_prompts_per_train_step=1,
            dp_degree=1,
            pad_id=0,
            temperature=1.0,
        )
