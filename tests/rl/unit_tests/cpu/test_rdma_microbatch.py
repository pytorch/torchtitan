# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `MonarchRDMAMicrobatch` and `RDMAMicrobatchPrefetcher`, with Monarch's RDMA
classes faked except in one test over real Monarch (TCP)."""

import asyncio
import contextvars
import dataclasses
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from monarch.actor import Actor, current_rank, endpoint, this_host
from torch.utils import _pytree as pytree
from torchtitan.rl.rollout.types import RolloutTurnID

from torchtitan.rl.trainer import rdma_microbatch
from torchtitan.rl.trainer.batcher import Batcher, TrainingMicrobatch
from torchtitan.rl.trainer.rdma_microbatch import (
    _NUM_MICROBATCHES_READ_AHEAD,
    _TensorRef,
    MonarchRDMAMicrobatch,
    RDMAMicrobatchPrefetcher,
)
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleGroup,
)


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
    registered = MonarchRDMAMicrobatch.register(
        microbatch, min_bytes=min_bytes, batch_index=0, dp_rank=0, microbatch_index=0
    )
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
    # One group per batch, one 256-token microbatch: the int64 (2 KiB) and float32 (1 KiB) fields
    # reach 1 KiB; the bool masks and token counts stay in the message.
    return Batcher.Config(rdma_min_tensor_bytes=rdma_min_tensor_bytes).build(
        num_tokens_per_microbatch_per_dp_rank=256,
        max_context_length=256,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )


def _training_sample_group(group_id: int, num_samples: int = 1) -> TrainingSampleGroup:
    samples = [
        TrainingSample(
            min_policy_version=0,
            max_policy_version=0,
            rollout_id=RolloutTurnID(
                group_id=group_id, rollout_id=rollout_id, turn_id=0
            ),
            # Distinct token ids per group and sample, so a swapped column shows.
            token_ids=np.arange(9) + 1000 * group_id + 100 * rollout_id,
            loss_mask=np.array([False] + [True] * 8),
            logprobs=np.zeros(9, dtype=np.float32),
            advantage=np.ones(9, dtype=np.float32),
        )
        for rollout_id in range(num_samples)
    ]
    return TrainingSampleGroup(group_id=group_id, training_samples=samples, metrics=[])


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
    assert [batch.microbatches[0][0].batch_index for batch in batches] == [0, 1]


def _fake_rdma_microbatch(index: int, started: list[int]) -> SimpleNamespace:
    """Stands in for a `MonarchRDMAMicrobatch`: `read()` records its index."""

    async def read() -> int:
        started.append(index)
        return index

    return SimpleNamespace(read=read)


def test_prefetcher_yields_in_order_and_reads_two_ahead() -> None:
    started: list[int] = []
    prefetcher = RDMAMicrobatchPrefetcher(
        [_fake_rdma_microbatch(index, started) for index in range(6)]
    )

    assert len(prefetcher) == 6
    for [index] in prefetcher:
        # Wait for the expected read, then 20 ms more, so a read too far ahead shows.
        expected = min(index + _NUM_MICROBATCHES_READ_AHEAD, 5)
        deadline = time.monotonic() + 5
        while max(started) < expected and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.02)
        assert max(started) == expected
    assert started == list(range(6))


def test_prefetcher_read_thread_sees_caller_context() -> None:
    step = contextvars.ContextVar("step")
    step.set(7)

    async def read() -> int:
        return step.get()

    prefetcher = RDMAMicrobatchPrefetcher([SimpleNamespace(read=read)])

    assert list(prefetcher) == [[7]]


def test_prefetcher_skips_queued_reads_when_training_stops() -> None:
    started: list[int] = []
    read_1_started, read_1_done = threading.Event(), threading.Event()

    async def read_1() -> int:
        read_1_started.set()
        await asyncio.sleep(0.2)
        read_1_done.set()
        return 1

    prefetcher = RDMAMicrobatchPrefetcher(
        [
            _fake_rdma_microbatch(0, started),
            SimpleNamespace(read=read_1),
            _fake_rdma_microbatch(2, started),
        ]
    )
    iterator = iter(prefetcher)

    assert next(iterator) == [0]
    assert read_1_started.wait(timeout=5)
    # Read 1 is running and read 2 is queued: closing waits for read 1 and cancels read 2.
    iterator.close()

    assert read_1_done.is_set() and 2 not in started


class _TrainerRank(Actor):
    """Reads its DP column of a batch, as a trainer rank does."""

    @endpoint
    async def read_column(
        self, rdma_microbatches: list[list[MonarchRDMAMicrobatch]]
    ) -> list[TrainingMicrobatch]:
        dp_rank = current_rank().rank
        return [await row[dp_rank].read() for row in rdma_microbatches]


def test_four_ranks_read_their_columns_exactly_across_released_batches(
    monkeypatch,
) -> None:
    # Real Monarch RDMA over TCP (works on any CI host; the rank processes inherit the env var).
    monkeypatch.setenv("MONARCH_RDMA_DISABLE_IBVERBS", "1")
    batcher_kwargs = dict(
        num_tokens_per_microbatch_per_dp_rank=64,
        max_context_length=16,
        num_prompts_per_train_step=1,
        dp_degree=4,
        pad_id=0,
        temperature=1.0,
    )
    rdma_batcher = Batcher.Config(rdma_min_tensor_bytes=256).build(**batcher_kwargs)
    by_value_batcher = Batcher.Config().build(**batcher_kwargs)

    async def run() -> None:
        mesh = this_host().spawn_procs(per_host={"procs": 4})
        try:
            ranks = mesh.spawn("rank", _TrainerRank)
            for batch_index in range(3):
                group = _training_sample_group(batch_index, num_samples=8)
                batch, _ = rdma_batcher.add_training_samples(
                    training_sample_group=group
                )
                expected, _ = by_value_batcher.add_training_samples(
                    training_sample_group=group
                )
                # All 4 ranks read at once, while the previous batch is still registered.
                columns = list(
                    (await ranks.read_column.call(batch.microbatches)).values()
                )
                for dp_rank, column in enumerate(columns):
                    torch.testing.assert_close(
                        [dataclasses.asdict(microbatch) for microbatch in column],
                        [
                            dataclasses.asdict(row[dp_rank])
                            for row in expected.microbatches
                        ],
                        rtol=0,
                        atol=0,
                    )
                if batch_index > 0:
                    # Free the previous batch; the next one registers fresh buffers.
                    await rdma_batcher.maybe_release_oldest_batch()
            await rdma_batcher.maybe_release_oldest_batch()
        finally:
            await mesh.stop()

    asyncio.run(run())
