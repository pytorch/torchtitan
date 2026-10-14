# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `MonarchRDMAMicrobatch`, with Monarch's RDMA classes faked."""

import asyncio
import dataclasses

import numpy as np
import pytest
import torch
from torch.utils import _pytree as pytree
from torchtitan.rl.components.batcher import Batcher, TrainingMicrobatch
from torchtitan.rl.components.training_sample_builder import (
    TrainingSample,
    TrainingSampleGroup,
)

from torchtitan.rl.distributed import rdma_microbatch
from torchtitan.rl.distributed.rdma_microbatch import _TensorRef, MonarchRDMAMicrobatch
from torchtitan.rl.rollout.types import RolloutTurnID


class _FakeRDMABuffer:
    """Stands in for `monarch.rdma.RDMABuffer`: keeps the owner's byte view."""

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data
        self.dropped = False

    async def drop(self) -> None:
        self.dropped = True


class _FakeRDMAAction:
    """Stands in for `monarch.rdma.RDMAAction`: copies every queued read on `submit()`."""

    def __init__(self) -> None:
        self._reads: list[tuple[torch.Tensor, _FakeRDMABuffer]] = []

    def read_remote(self, dst: torch.Tensor, src: _FakeRDMABuffer) -> "_FakeRDMAAction":
        self._reads.append((dst, src))
        return self

    async def submit(self) -> None:
        for dst, src in self._reads:
            dst.copy_(src.data)


def _microbatch(num_tokens: int) -> TrainingMicrobatch:
    tokens = torch.arange(num_tokens)
    return TrainingMicrobatch(
        input=tokens + 3,
        labels=tokens + 5,
        positions=tokens % 7,
        padding_mask=tokens % 3 == 0,
        loss_token_counts=torch.tensor(2),
        routing_token_counts=torch.tensor([2, 2]),
        generator_logprobs=-tokens.float() / num_tokens,
        temperature=torch.ones(num_tokens),
        loss_mask=tokens % 2 == 0,
        advantages=tokens.float() / num_tokens,
        model_kwargs={
            "routed_expert_ids": torch.arange(num_tokens * 4, dtype=torch.uint8).view(
                num_tokens, 2, 2
            ),
            "topk_logprobs": torch.zeros(0, 4),
        },
    )


def test_rdma_microbatch_round_trips_every_field_and_drops_its_buffers(
    monkeypatch,
) -> None:
    monkeypatch.setattr(rdma_microbatch, "RDMABuffer", _FakeRDMABuffer)
    monkeypatch.setattr(rdma_microbatch, "RDMAAction", _FakeRDMAAction)
    # At 512 tokens the int64 (4 KiB), float (2 KiB) and routed-id (2 KiB, in model_kwargs)
    # tensors go by ref; the bool masks (512 B), token counts and empty tensor go by value.
    microbatch = _microbatch(num_tokens=512)

    min_bytes = 1024
    registered = MonarchRDMAMicrobatch.register(microbatch, min_bytes=min_bytes)
    read_microbatch = asyncio.run(registered.read())

    names = [field.name for field in dataclasses.fields(microbatch)]
    leaves = pytree.tree_leaves({name: getattr(microbatch, name) for name in names})
    rdma_leaves = pytree.tree_leaves(registered.fields)
    read_leaves = pytree.tree_leaves(
        {name: getattr(read_microbatch, name) for name in names}
    )
    for leaf, rdma_leaf, read_leaf in zip(
        leaves, rdma_leaves, read_leaves, strict=True
    ):
        assert isinstance(rdma_leaf, _TensorRef) == (leaf.nbytes >= min_bytes)
        assert torch.equal(leaf, read_leaf) and leaf.dtype == read_leaf.dtype

    asyncio.run(registered.drop())
    tensor_refs = [leaf for leaf in rdma_leaves if isinstance(leaf, _TensorRef)]
    assert tensor_refs and all(tensor_ref.buffer.dropped for tensor_ref in tensor_refs)


def _build_batcher(rdma_min_tensor_bytes: int | None) -> Batcher:
    # One group per batch, one 256-token microbatch: `input` and `labels` (int64, 2 KiB) pass 1 KiB.
    return Batcher.Config(rdma_min_tensor_bytes=rdma_min_tensor_bytes).build(
        num_tokens_per_microbatch_per_dp_rank=256,
        max_context_length=256,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )


def _training_sample_group(group_id: int) -> TrainingSampleGroup:
    sample = TrainingSample(
        min_policy_version=0,
        max_policy_version=0,
        rollout_id=RolloutTurnID(group_id=group_id, rollout_id=0, turn_id=0),
        token_ids=np.arange(9),
        loss_mask=np.array([False] + [True] * 8),
        logprobs=np.zeros(9, dtype=np.float32),
        advantage=np.ones(9, dtype=np.float32),
    )
    return TrainingSampleGroup(group_id=group_id, training_samples=[sample], metrics=[])


def test_batcher_without_rdma_min_tensor_bytes_never_registers(monkeypatch) -> None:
    def fail_to_register(tensor: torch.Tensor) -> None:
        pytest.fail("registered an RDMA buffer with rdma_min_tensor_bytes unset")

    monkeypatch.setattr(rdma_microbatch, "RDMABuffer", fail_to_register)
    batcher = _build_batcher(rdma_min_tensor_bytes=None)

    batch, _ = batcher.add_training_samples(
        training_sample_group=_training_sample_group(0)
    )
    asyncio.run(batcher.maybe_release_oldest_batch())

    assert isinstance(batch.microbatches[0][0], TrainingMicrobatch)


def test_batcher_registers_each_batch_and_releases_the_oldest(monkeypatch) -> None:
    monkeypatch.setattr(rdma_microbatch, "RDMABuffer", _FakeRDMABuffer)
    batcher = _build_batcher(rdma_min_tensor_bytes=1024)

    batches = [
        batcher.add_training_samples(
            training_sample_group=_training_sample_group(group_id)
        )[0]
        for group_id in (0, 1)
    ]
    asyncio.run(batcher.maybe_release_oldest_batch())

    buffers = [
        [
            leaf.buffer
            for row in batch.microbatches
            for microbatch in row
            for leaf in pytree.tree_leaves(microbatch.fields)
            if isinstance(leaf, _TensorRef)
        ]
        for batch in batches
    ]
    assert buffers[0] and all(buffer.dropped for buffer in buffers[0])
    assert buffers[1] and not any(buffer.dropped for buffer in buffers[1])
