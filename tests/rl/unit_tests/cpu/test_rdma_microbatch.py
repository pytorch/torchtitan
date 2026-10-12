# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `RDMAMicrobatch` and `MicrobatchStream`, with Monarch's RDMA classes faked."""

import asyncio
import contextvars
import dataclasses
import threading
import time
from types import SimpleNamespace

import torch
from torch.utils import _pytree as pytree
from torchtitan.rl.components.batcher import TrainingMicrobatch

from torchtitan.rl.distributed import rdma_microbatch
from torchtitan.rl.distributed.rdma_microbatch import (
    _NUM_MICROBATCHES_READ_AHEAD,
    _TensorRef,
    MicrobatchStream,
    RDMAMicrobatch,
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
    registered = RDMAMicrobatch.register(microbatch, min_bytes=min_bytes)
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


def _fake_rdma_microbatch(index: int, started: list[int]) -> SimpleNamespace:
    """Stands in for an `RDMAMicrobatch`: `read()` records its index."""

    async def read() -> int:
        started.append(index)
        return index

    return SimpleNamespace(read=read)


def test_microbatch_stream_yields_in_order_and_reads_two_ahead() -> None:
    started: list[int] = []
    stream = MicrobatchStream(
        [_fake_rdma_microbatch(index, started) for index in range(6)]
    )

    assert len(stream) == 6
    for [index] in stream:
        # Wait for the expected read, then 20 ms more, so a read too far ahead shows.
        expected = min(index + _NUM_MICROBATCHES_READ_AHEAD, 5)
        deadline = time.monotonic() + 5
        while max(started) < expected and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.02)
        assert max(started) == expected
    assert started == list(range(6))


def test_microbatch_stream_read_thread_sees_caller_context() -> None:
    step = contextvars.ContextVar("step")
    step.set(7)

    async def read() -> int:
        return step.get()

    stream = MicrobatchStream([SimpleNamespace(read=read)])

    assert list(stream) == [[7]]


def test_microbatch_stream_skips_queued_reads_when_training_stops() -> None:
    started: list[int] = []
    read_1_started, read_1_done = threading.Event(), threading.Event()

    async def read_1() -> int:
        read_1_started.set()
        await asyncio.sleep(0.2)
        read_1_done.set()
        return 1

    stream = MicrobatchStream(
        [
            _fake_rdma_microbatch(0, started),
            SimpleNamespace(read=read_1),
            _fake_rdma_microbatch(2, started),
        ]
    )
    iterator = iter(stream)

    assert next(iterator) == [0]
    assert read_1_started.wait(timeout=5)
    # Read 1 is running and read 2 is queued: closing waits for read 1 and cancels read 2.
    iterator.close()

    assert read_1_done.is_set() and 2 not in started
