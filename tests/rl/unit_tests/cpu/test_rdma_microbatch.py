# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `TransportRDMAMicrobatch` and `RDMAMicrobatchPrefetcher`, with the transport
faked except in one test over real NIXL across processes."""

import asyncio
import contextvars
import dataclasses
import pickle
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from monarch.actor import Actor, current_rank, endpoint, this_host
from torch.utils import _pytree as pytree

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.models.qwen3_5 import QWEN3_5_VISION_PROCESSOR_KWARGS
from torchtitan.rl.rollout.types import RolloutTurnID
from torchtitan.rl.trainer import rdma_microbatch
from torchtitan.rl.trainer.batcher import Batcher, TrainingMicrobatch
from torchtitan.rl.trainer.rdma_microbatch import (
    _NUM_MICROBATCHES_READ_AHEAD,
    _TensorRef,
    RDMAMicrobatch,
    RDMAMicrobatchPrefetcher,
    TransportRDMAMicrobatch,
)
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleGroup,
)


# Fake remote buffers pickle as an id, so a reader sees the owner's live memory, like a real transport.
_fake_memories: dict[int, "_FakeMemory"] = {}


def _fake_remote_buffer(memory_id: int) -> "_FakeRemoteBuffer":
    return _FakeRemoteBuffer(_fake_memories[memory_id])


class _FakeRemoteBuffer:
    __slots__ = ("memory",)

    def __init__(self, memory: "_FakeMemory") -> None:
        self.memory = memory

    def __reduce__(self) -> tuple[object, ...]:
        return _fake_remote_buffer, (id(self.memory),)


class _FakeMemory:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.data = tensor
        self.unregistered = False
        _fake_memories[id(self)] = self

    def to_remote_buffer(self) -> _FakeRemoteBuffer:
        return _FakeRemoteBuffer(self)

    def to_mutable_view(self, offset: int, nbytes: int) -> torch.Tensor:
        assert offset + nbytes <= self.data.numel()
        return self.data[offset : offset + nbytes]


class _FakeWork:
    """Copies when waited on, so a reader that skips `wait_all` sees no data."""

    def __init__(self, local: torch.Tensor, remote: _FakeRemoteBuffer) -> None:
        self.local = local
        self.remote = remote

    def run(self) -> None:
        assert not self.remote.memory.unregistered
        self.local.copy_(self.remote.memory.data)


class _FakeTransport:
    """Stands in for a `torch.distributed._transport` transport in one process."""

    def __init__(self, backend: str) -> None:
        self.url = backend.encode()
        self.connected: set[bytes] = set()
        self.memories: list[_FakeMemory] = []

    def bind(self) -> bytes:
        return self.url

    def connect(self, url: bytes) -> None:
        self.connected.add(url)

    def register_memory(self, tensor: torch.Tensor) -> _FakeMemory:
        assert tensor.dtype == torch.uint8 and tensor.numel() > 0
        assert tensor.is_contiguous()
        memory = _FakeMemory(tensor)
        self.memories.append(memory)
        return memory

    def unregister_memory(self, memory: _FakeMemory) -> None:
        memory.unregistered = True

    def read(
        self, local: torch.Tensor, remote: _FakeRemoteBuffer, *, async_op: bool
    ) -> _FakeWork:
        assert async_op and self.url in self.connected
        assert not remote.memory.unregistered
        return _FakeWork(local, remote)


async def _wait_all(works, *, timeout: float) -> None:
    assert timeout == rdma_microbatch._READ_TIMEOUT_S
    for work in works:
        work.run()


@pytest.fixture
def fake_transport(monkeypatch) -> _FakeTransport:
    transport = _FakeTransport("nixl")
    monkeypatch.setattr(
        rdma_microbatch, "_transports", {"nixl": (transport, transport.bind())}
    )
    monkeypatch.setattr(rdma_microbatch, "wait_all", _wait_all)
    return transport


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
        vision_processor=VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS),
    )


def test_rdma_microbatch_round_trips_every_field_and_drops_its_buffers(
    fake_transport,
) -> None:
    # At 512 tokens the int64 (4 KiB), float (2 KiB) and routed-id (2 KiB, in model_kwargs)
    # tensors go by ref; the bool masks (512 B), token counts and empty tensor go by value.
    microbatch = _microbatch(num_tokens=512)

    min_bytes = 1024
    registered = TransportRDMAMicrobatch.register(
        microbatch,
        backend="nixl",
        min_bytes=min_bytes,
        batch_index=0,
        dp_rank=0,
        microbatch_index=0,
    )
    # A trainer rank gets a pickled copy, without the owner's registrations.
    received = pickle.loads(pickle.dumps(registered))
    assert registered.registered and not received.registered
    read_microbatch = asyncio.run(received.read())

    names = [field.name for field in dataclasses.fields(microbatch)]
    leaves = pytree.tree_leaves({name: getattr(microbatch, name) for name in names})
    rdma_leaves = pytree.tree_leaves(registered.fields)
    read_leaves = pytree.tree_leaves(
        {name: getattr(read_microbatch, name) for name in names}
    )
    for leaf, rdma_leaf, read_leaf in zip(
        leaves, rdma_leaves, read_leaves, strict=True
    ):
        if not isinstance(leaf, torch.Tensor):  # e.g. vision_processor, by value
            assert rdma_leaf == leaf == read_leaf
            continue
        assert isinstance(rdma_leaf, _TensorRef) == (leaf.nbytes >= min_bytes)
        assert torch.equal(leaf, read_leaf) and leaf.dtype == read_leaf.dtype

    tensor_refs = [leaf for leaf in rdma_leaves if isinstance(leaf, _TensorRef)]
    # The reader unregisters its buffer once the reads finish.
    assert len(fake_transport.memories) == len(tensor_refs) + 1
    assert fake_transport.memories[-1].unregistered
    asyncio.run(registered.drop())
    assert tensor_refs and all(
        tensor_ref.buffer.memory.unregistered for tensor_ref in tensor_refs
    )


@pytest.mark.parametrize(
    ("error", "unregistered"), [(RuntimeError, True), (TimeoutError, False)]
)
def test_read_failure_unregisters_only_when_reads_are_done(
    fake_transport, monkeypatch, error: type[Exception], unregistered: bool
) -> None:
    """A read error drains the reads, so the reader frees its buffer; a timeout leaves them in flight."""

    async def failing_wait_all(works, *, timeout: float) -> None:
        raise error

    monkeypatch.setattr(rdma_microbatch, "wait_all", failing_wait_all)
    registered = TransportRDMAMicrobatch.register(
        _microbatch(num_tokens=512),
        backend="nixl",
        min_bytes=1024,
        batch_index=0,
        dp_rank=0,
        microbatch_index=0,
    )
    num_owner_memories = len(fake_transport.memories)

    with pytest.raises(error):
        asyncio.run(registered.read())

    assert len(fake_transport.memories) == num_owner_memories + 1
    assert fake_transport.memories[-1].unregistered == unregistered


def test_only_registered_microbatches_are_rdma_microbatches(fake_transport) -> None:
    """The trainer tells the two kinds apart with `isinstance(..., RDMAMicrobatch)`."""
    microbatch = _microbatch(num_tokens=512)
    registered = TransportRDMAMicrobatch.register(
        microbatch,
        backend="nixl",
        min_bytes=1024,
        batch_index=0,
        dp_rank=0,
        microbatch_index=0,
    )

    assert isinstance(registered, RDMAMicrobatch)
    assert not isinstance(microbatch, RDMAMicrobatch)


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


def test_batcher_without_rdma_min_tensor_bytes_never_registers(fake_transport) -> None:
    batcher = _build_batcher(rdma_min_tensor_bytes=None)

    batch, _ = batcher.add_training_samples(
        training_sample_group=_training_sample_group(0)
    )
    asyncio.run(batcher.maybe_release_oldest_batch())

    assert isinstance(batch.microbatch_groups[0][0][0], TrainingMicrobatch)
    assert not fake_transport.memories


def test_batcher_registers_each_batch_and_releases_the_oldest(fake_transport) -> None:
    batcher = _build_batcher(rdma_min_tensor_bytes=1024)

    batches = [
        batcher.add_training_samples(
            training_sample_group=_training_sample_group(group_id)
        )[0]
        for group_id in (0, 1)
    ]
    asyncio.run(batcher.maybe_release_oldest_batch())

    memories = [
        [
            leaf.buffer.memory
            for group in batch.microbatch_groups
            for row in group
            for microbatch in row
            for leaf in pytree.tree_leaves(microbatch.fields)
            if isinstance(leaf, _TensorRef)
        ]
        for batch in batches
    ]
    assert memories[0] and all(memory.unregistered for memory in memories[0])
    assert memories[1] and not any(memory.unregistered for memory in memories[1])
    assert [batch.microbatch_groups[0][0][0].batch_index for batch in batches] == [
        0,
        1,
    ]


class _ReadMicrobatch(int):
    """Stands in for a read `TrainingMicrobatch` without images: an index."""

    def preprocess_images(self) -> "_ReadMicrobatch":
        return self


def _fake_rdma_microbatch(index: int, started: list[int]) -> SimpleNamespace:
    """Stands in for a `TransportRDMAMicrobatch`: `read()` records its index."""

    async def read() -> _ReadMicrobatch:
        started.append(index)
        return _ReadMicrobatch(index)

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

    async def read() -> _ReadMicrobatch:
        return _ReadMicrobatch(step.get())

    prefetcher = RDMAMicrobatchPrefetcher([SimpleNamespace(read=read)])

    assert list(prefetcher) == [[7]]


def test_prefetcher_preprocesses_images_on_the_read_thread() -> None:
    preprocessed = SimpleNamespace()

    class _ImageMicrobatch:
        def preprocess_images(self) -> SimpleNamespace:
            preprocessed.thread = threading.current_thread()
            return preprocessed

    async def read() -> _ImageMicrobatch:
        return _ImageMicrobatch()

    prefetcher = RDMAMicrobatchPrefetcher([SimpleNamespace(read=read)])

    assert list(prefetcher) == [[preprocessed]]
    assert preprocessed.thread is not threading.main_thread()


def test_prefetcher_skips_queued_reads_when_training_stops() -> None:
    started: list[int] = []
    read_1_started, read_1_done = threading.Event(), threading.Event()

    async def read_1() -> _ReadMicrobatch:
        read_1_started.set()
        await asyncio.sleep(0.2)
        read_1_done.set()
        return _ReadMicrobatch(1)

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
        self, rdma_microbatches: list[list[TransportRDMAMicrobatch]]
    ) -> list[TrainingMicrobatch]:
        dp_rank = current_rank().rank
        return [await row[dp_rank].read() for row in rdma_microbatches]


def test_four_ranks_read_their_columns_exactly_across_released_batches() -> None:
    # Real NIXL: UCX uses TCP or shared memory on hosts without RDMA devices.
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
                rows = [row for group in batch.microbatch_groups for row in group]
                columns = list((await ranks.read_column.call(rows)).values())
                for dp_rank, column in enumerate(columns):
                    torch.testing.assert_close(
                        [dataclasses.asdict(microbatch) for microbatch in column],
                        [
                            dataclasses.asdict(row[dp_rank])
                            for group in expected.microbatch_groups
                            for row in group
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
