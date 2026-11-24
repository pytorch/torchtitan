# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.rl.torch_transport_rdma`: the registered-buffer pool, range reads,
and the setup probe.

Most tests use `LoopbackTransport`, an in-process stand-in for torch transport. The tests marked
`requires_nixl` use the real NIXL backend (two transports in this process) and skip without the
`nixl` package (`scripts/install_rl_env.sh` installs it; see
`torchtitan/rl/MODEL_STATE_DICT_TRANSPORT.md`).
"""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from torchtitan.rl.torch_transport_rdma import (
    buffer_size_class,
    check_rdma_probes,
    make_rdma_probe,
    RDMAByteBuffers,
    RegisteredBufferPool,
)

MiB = 1 << 20


@dataclass(frozen=True)
class _LoopbackRemoteBuffer:
    key: int
    offset: int
    length: int

    def slice(self, offset: int, length: int) -> _LoopbackRemoteBuffer:
        assert 0 <= offset and offset + length <= self.length
        return replace(self, offset=self.offset + offset, length=length)


@dataclass(frozen=True)
class _LoopbackView:
    tensor: torch.Tensor
    offset: int
    length: int


class _LoopbackMemory:
    def __init__(self, key: int, tensor: torch.Tensor) -> None:
        self.key = key
        self.tensor = tensor

    def to_remote_buffer(self) -> _LoopbackRemoteBuffer:
        return _LoopbackRemoteBuffer(self.key, 0, self.tensor.numel())

    def to_mutable_view(self, offset: int, length: int) -> _LoopbackView:
        return _LoopbackView(self.tensor, offset, length)


class _DoneWork:
    def __init__(self, transport: LoopbackTransport) -> None:
        self._transport = transport

    def wait(self) -> None:
        self._transport.num_waits += 1


class LoopbackTransport:
    """In-process stand-in for a torch transport: registrations of every `LoopbackTransport`
    sharing `registry` are readable by the others, and each call is counted.

    `fail_reads` makes every read raise, as a host whose RDMA transfers fail;
    `fail_after_num_reads` makes every read after that many raise."""

    def __init__(self, registry: dict[int, torch.Tensor], name: str) -> None:
        self.registry = registry
        self.name = name
        self.num_registrations = 0
        self.num_unregistrations = 0
        self.read_ranges: list[tuple[int, int]] = []
        self.num_waits = 0
        self.connected: list[bytes] = []
        self.fail_reads = False
        self.fail_after_num_reads: int | None = None

    def bind(self) -> bytes:
        return self.name.encode()

    def connect(self, peer_url: bytes) -> int:
        self.connected.append(peer_url)
        return 0

    def register_memory(self, tensor: torch.Tensor) -> _LoopbackMemory:
        key = len(self.registry)
        self.registry[key] = tensor
        self.num_registrations += 1
        return _LoopbackMemory(key, tensor)

    def unregister_memory(self, memory: _LoopbackMemory) -> None:
        self.num_unregistrations += 1
        del self.registry[memory.key]

    def read(
        self, local: _LoopbackView, remote: _LoopbackRemoteBuffer, *, async_op: bool
    ) -> _DoneWork:
        assert async_op and local.length == remote.length
        if self.fail_reads or (
            self.fail_after_num_reads is not None
            and len(self.read_ranges) >= self.fail_after_num_reads
        ):
            raise RuntimeError("NIXL_ERR_REMOTE_DISCONNECT")
        source = self.registry[remote.key][
            remote.offset : remote.offset + remote.length
        ]
        local.tensor[local.offset : local.offset + local.length] = source
        self.read_ranges.append((remote.offset, remote.length))
        return _DoneWork(self)


def loopback_buffers() -> tuple[RDMAByteBuffers, RDMAByteBuffers]:
    """An owner's and a reader's `RDMAByteBuffers` over `LoopbackTransport`s that see each
    other's registrations."""
    registry: dict[int, torch.Tensor] = {}
    return (
        RDMAByteBuffers(LoopbackTransport(registry, "owner")),
        RDMAByteBuffers(LoopbackTransport(registry, "reader")),
    )


def test_registered_bytes_are_each_size_classes_peak_in_use() -> None:
    """The bound the pool docstring states. Taking the smallest free buffer that fits does not
    have it: with a 16 MiB buffer free, a 4 MiB group took it and the next 16 MiB group
    registered another, 32 MiB for 20 MiB in use; here that is 20 MiB."""
    pool = RegisteredBufferPool(LoopbackTransport({}, "owner"))
    pool.release(pool.acquire(16 * MiB))
    pool.acquire(4 * MiB)
    pool.acquire(16 * MiB)
    assert pool.registered_bytes == pool.bytes_in_use == 20 * MiB

    pool = RegisteredBufferPool(LoopbackTransport({}, "owner"))
    rng = random.Random(0)
    held: list[tuple[int, int]] = []
    in_use = collections.Counter()
    peak = collections.Counter()
    for _ in range(2000):
        if held and rng.random() < 0.5:
            index, size = held.pop(rng.randrange(len(held)))
            pool.release(index)
            in_use[size] -= 1
        else:
            num_bytes = rng.choice([1, 4, 17, 66]) * MiB - rng.randrange(4096)
            size = buffer_size_class(num_bytes)
            held.append((pool.acquire(num_bytes), size))
            in_use[size] += 1
            peak[size] = max(peak[size], in_use[size])

    assert pool.registered_bytes == sum(size * count for size, count in peak.items())


def test_released_buffers_stay_registered() -> None:
    """Buffers go back to the free list, never to the transport: a reader may hold their
    metadata, and NIXL would match a later buffer at the same address to it."""
    owner, _ = loopback_buffers()
    transport = owner._transport
    for size in (4 * MiB, 16 * MiB, 4 * MiB, 1 * MiB, 16 * MiB):
        handles = [owner.allocate(size)[1] for _ in range(3)]
        for handle in handles:
            owner.release(handle)

    assert transport.num_unregistrations == 0
    assert transport.num_registrations == 9  # 3 each of 4 MiB, 16 MiB, then 1 MiB
    assert owner.pool.free_bytes == owner.pool.registered_bytes


def test_a_read_whose_later_submission_raises_waits_for_the_reads_already_submitted() -> (
    None
):
    """Those reads may still be writing into the staging buffer; the next read (e.g. the next
    probe) must not start until they end."""
    owner, reader = loopback_buffers()
    _, handle = owner.allocate(1000)
    reader._transport.fail_after_num_reads = 2

    with pytest.raises(RuntimeError, match="NIXL_ERR_REMOTE_DISCONNECT"):
        reader.read([(handle, [(0, 10), (10, 10), (20, 10)])])

    assert reader._transport.num_waits == 2


def test_a_probe_read_that_fails_raises_at_setup() -> None:
    """Fail fast: a host whose reads fail stops the run at setup, with the worker's host named,
    instead of failing mid-step."""
    owner, reader = loopback_buffers()
    probe = replace(make_rdma_probe(owner), host="rollout-host-3")
    check_rdma_probes(reader, [probe])
    reader._transport.fail_reads = True

    with pytest.raises(ValueError, match="rollout-host-3"):
        check_rdma_probes(reader, [probe])


def _nixl_available() -> bool:
    try:
        from torch.distributed._transport.nixl import NIXLTransport
    except ImportError:
        return False
    return NIXLTransport.supported()


requires_nixl = pytest.mark.skipif(
    not _nixl_available(), reason="needs torch transport with the nixl package"
)


def _stamp(tensor: torch.Tensor, value: int) -> None:
    tensor.copy_((torch.arange(tensor.numel()) + value) % 251)


@requires_nixl
def test_nixl_reads_slices_after_freeing_small_buffers_for_larger_ones() -> None:
    """The stale-section case: a NIXL reader keeps each buffer it read as its owner registered
    it. Freeing 4 MiB buffers and registering 16 MiB ones over the freed addresses made reads of
    the second half fail with NIXL_ERR_NOT_FOUND. The pool keeps the 4 MiB buffers registered,
    so the 16 MiB ones land elsewhere and every slice reads the right bytes."""
    owner, reader = RDMAByteBuffers(), RDMAByteBuffers()
    small = []
    for value in range(16):
        tensor, handle = owner.allocate(4 * MiB)
        _stamp(tensor, value)
        small.append(handle)
    for value, handle in enumerate(small):
        [[chunk]] = reader.read([(handle, [(0, 4 * MiB)])])
        np.testing.assert_array_equal(chunk, (np.arange(4 * MiB) + value) % 251)
    for handle in small:
        owner.release(handle)

    large = []
    for value in range(16):
        tensor, handle = owner.allocate(16 * MiB)
        _stamp(tensor, 100 + value)
        large.append(handle)

    assert owner.pool.num_registrations == 32
    expected_all = np.arange(16 * MiB)
    for value, handle in enumerate(large):
        [[second_half, inside]] = reader.read(
            [(handle, [(8 * MiB, 8 * MiB), (4 * MiB + 64, 64)])]
        )
        expected = (expected_all + 100 + value) % 251
        np.testing.assert_array_equal(second_half, expected[8 * MiB :])
        np.testing.assert_array_equal(inside, expected[4 * MiB + 64 : 4 * MiB + 128])


@requires_nixl
def test_nixl_reads_new_bytes_from_reused_buffers_without_registering() -> None:
    """A reader already loaded each buffer; after the owner reuses it for a smaller group of the
    same size class, reads see the new group's bytes."""
    owner, reader = RDMAByteBuffers(), RDMAByteBuffers()
    first = [owner.allocate(16 * MiB)[1] for _ in range(16)]
    for handle in first:
        reader.read([(handle, [(0, 16)])])
        owner.release(handle)

    reused = []
    for value in range(16):
        tensor, handle = owner.allocate(15 * MiB)
        _stamp(tensor, value)
        reused.append(handle)

    assert owner.pool.num_registrations == 16
    for value, handle in enumerate(reused):
        [[tail]] = reader.read([(handle, [(13 * MiB, 2 * MiB)])])
        np.testing.assert_array_equal(
            tail, (np.arange(13 * MiB, 15 * MiB) + value) % 251
        )
