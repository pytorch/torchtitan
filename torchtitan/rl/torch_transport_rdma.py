# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Byte buffers that stay in the process that wrote them (their owner) and that other processes
read by byte range over torch transport (`torch.distributed._transport`, NIXL backend). The
rollout workers keep each packed rollout group in one; the trainer's DP replica leaders read only
the rollouts they train on (`BaseRolloutWorker.Config.rdma_min_group_bytes`).

    owner (rollout worker)                         reader (DP replica leader)
    tensor, handle = buffers.allocate(n)
    write tensor, send handle  ------ handle ----> [[chunk]] = buffers.read([(handle, [(64, 128)])])
                                                   (one-sided reads of RemoteBuffer.slice)
    buffers.release(handle), after the last read: back to the pool, still registered

Why buffers are reused instead of unregistered: each NIXL read loads the owner's metadata for the
buffer into the reader, and the owner's unregister never reaches the reader. Once the owner
registers a new buffer over an address range it freed, a reader's lookup can match the freed
buffer and fail (`NIXL_ERR_NOT_FOUND`) or read through a deregistered key. A pool that never
unregisters keeps every buffer a reader has loaded valid.

Prompt image files to the generators take another path today, the actor system's RDMA objects
(`rl/rdma_object.py`, Monarch `RDMABuffer`): they are read whole, once per generate call, and
Monarch's buffers have no slice read, which the trainer's per-rollout reads need here.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

# TODO: torch.distributed._transport is a private upstream API; switch to a public one once it
# exists (no upstream issue yet; the same TODO as in model_state_dict_transport.py).
if TYPE_CHECKING:
    from torch.distributed._transport import Memory, RemoteBuffer, Transport

# The smallest buffer the pool registers; smaller requests share its size.
_MIN_BUFFER_BYTES = 64 * 1024


def buffer_size_class(num_bytes: int) -> int:
    """The size of the buffer the pool registers for a request of `num_bytes`: the smallest
    `m * 2**e` of at least `num_bytes` with `m` in 4..7, and at least 64 KiB. So a new buffer is
    at most 25% larger than its request, and requests of similar sizes share buffers.

    Example:
        buffer_size_class(16 << 20)  # 16 MiB
        buffer_size_class(17 << 20)  # 20 MiB
        buffer_size_class(66 << 20)  # 80 MiB
        buffer_size_class(1024)      # 64 KiB
    """
    if num_bytes <= _MIN_BUFFER_BYTES:
        return _MIN_BUFFER_BYTES
    # 2**exponent is 1/8 to 1/4 of num_bytes, so the multiple is 5 to 8 (8 * 2**e = 4 * 2**(e + 1)).
    exponent = (num_bytes - 1).bit_length() - 3
    multiple = -(-num_bytes // (1 << exponent))
    return multiple << exponent


@dataclass(frozen=True, slots=True)
class _PooledBuffer:
    tensor: torch.Tensor
    """uint8, `buffer_size_class` bytes."""

    memory: Memory

    remote_buffer: RemoteBuffer
    """The whole registration, as readers address it."""


# TODO: free idle buffers once torch transport invalidates a reader's loaded sections when their
# owner unregisters them (https://github.com/pytorch/pytorch/issues/206300). Today it does not
# (module docstring), so freeing a buffer a reader has read can make a later buffer at the same
# addresses unreadable; that is why this pool never unregisters, and why its memory never shrinks.
class RegisteredBufferPool:
    """uint8 CPU buffers registered with one transport, handed out and taken back, and never
    unregistered: a released buffer waits on its size class's free list for a later request of
    the same size class (see the module docstring for why it is never unregistered). The
    transport unregisters them when it closes, at the latest when the process exits.

    A request takes only a buffer of its own size class, never a larger free one: then a small
    group cannot hold a large buffer while a large group registers another, and the bytes
    registered for each size class are its own peak of buffers in use at once. The pool never
    shrinks, so that peak is the run's busiest moment (e.g. the first steps, when every rollout
    slot fills before the first release). Use from one thread at a time.

    Example:
        index = pool.acquire(5 << 20)  # registers a 5 MiB buffer
        pool.release(index)
        pool.acquire(9 << 19)          # 4.5 MiB, the same size class: the same buffer
        pool.acquire(5 << 20)          # none free in the class: registers another
        pool.acquire(1 << 20)          # another class: registers a 1 MiB buffer
    """

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._buffers: list[_PooledBuffer] = []
        # Released buffers by size; a set to catch a buffer released twice.
        self._free_indices_by_size: dict[int, list[int]] = {}
        self._free_indices: set[int] = set()
        self.registered_bytes = 0
        """Bytes of every buffer registered so far (the pool never shrinks)."""
        self.num_registrations = 0

    def acquire(self, num_bytes: int) -> int:
        """The index of a buffer of `buffer_size_class(num_bytes)` bytes: a free one, else a new
        one, registered now."""
        size = buffer_size_class(num_bytes)
        free_indices = self._free_indices_by_size.get(size)
        if free_indices:
            index = free_indices.pop()
            self._free_indices.remove(index)
            return index
        tensor = torch.empty(size, dtype=torch.uint8)
        memory = self._transport.register_memory(tensor)
        self._buffers.append(
            _PooledBuffer(
                tensor=tensor, memory=memory, remote_buffer=memory.to_remote_buffer()
            )
        )
        self.registered_bytes += size
        self.num_registrations += 1
        return len(self._buffers) - 1

    def release(self, index: int) -> None:
        """Put buffer `index` back on the free list; it stays registered."""
        assert index not in self._free_indices, f"buffer {index} released twice"
        self._free_indices.add(index)
        size = self._buffers[index].tensor.numel()
        self._free_indices_by_size.setdefault(size, []).append(index)

    def buffer(self, index: int) -> _PooledBuffer:
        return self._buffers[index]

    @property
    def free_bytes(self) -> int:
        return sum(self._buffers[index].tensor.numel() for index in self._free_indices)

    @property
    def bytes_in_use(self) -> int:
        """Bytes of the buffers handed out and not yet released (size-class rounded)."""
        return self.registered_bytes - self.free_bytes


@dataclass(frozen=True, slots=True)
class RDMAByteBuffer:
    """Bytes `[0, num_bytes)` of a registered buffer in its owner. A reader reads byte ranges of
    them (`RDMAByteBuffers.read`); the owner releases them after the last read
    (`RDMAByteBuffers.release`)."""

    owner_url: bytes
    """The owner's `Transport.bind()`; a reader connects to it once."""

    remote_buffer: RemoteBuffer
    """The owner's whole pooled buffer; reads take slices of it."""

    num_bytes: int

    buffer_index: int
    """The buffer's index in the owner's `RegisteredBufferPool`."""


class RDMAByteBuffers:
    """One process's torch transport and its `RegisteredBufferPool`: as an owner it hands out
    registered buffers for others to read, and as a reader it reads others' buffers into its own
    registered staging buffer. Use from one thread at a time (a leader probes on a worker
    thread at setup and reads on its reader thread later).

    Raises:
        ValueError: torch transport (NIXL) is unavailable in this process. The message names
            the host and the knob that turns the path off.

    Example:
        owner = RDMAByteBuffers()
        tensor, handle = owner.allocate(1 << 20)
        tensor[:4] = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)

        reader = RDMAByteBuffers()  # in another process; `handle` pickled to it
        [[chunk]] = reader.read([(handle, [(0, 4)])])  # array([1, 2, 3, 4], dtype=uint8)

        owner.release(handle)  # after the last read
    """

    def __init__(self, transport: Transport | None = None) -> None:
        if transport is None:
            try:
                from torch.distributed._transport import new_transport

                transport = new_transport("nixl", "cpu")
            except Exception as error:
                raise ValueError(
                    f"torch transport (NIXL) is unavailable on {socket.gethostname()}: "
                    f"{type(error).__name__}: {error}. Rollout groups by RDMA need it; set "
                    "BaseRolloutWorker.Config.rdma_min_group_bytes=None to send them by value."
                ) from error
        self._transport = transport
        self._url = transport.bind()
        self.pool = RegisteredBufferPool(transport)
        self._connected_urls: set[bytes] = set()
        # The reader's staging buffer and its registration, grown when a read needs more. No
        # other process reads it, so replacing it (unregistering the old one) is safe.
        self._staging: tuple[torch.Tensor, Memory] | None = None

    def allocate(self, num_bytes: int) -> tuple[torch.Tensor, RDMAByteBuffer]:
        """A writable uint8 tensor of `num_bytes` (a view of a pooled, registered buffer) and the
        handle readers read it with."""
        index = self.pool.acquire(num_bytes)
        pooled = self.pool.buffer(index)
        handle = RDMAByteBuffer(
            owner_url=self._url,
            remote_buffer=pooled.remote_buffer,
            num_bytes=num_bytes,
            buffer_index=index,
        )
        return pooled.tensor[:num_bytes], handle

    def release(self, buffer: RDMAByteBuffer) -> None:
        """On the owner, after the last read of `buffer`: its pooled buffer can be handed out
        again. It stays registered."""
        assert (
            buffer.owner_url == self._url
        ), "released by a process that does not own it"
        self.pool.release(buffer.buffer_index)

    def read(
        self, requests: Sequence[tuple[RDMAByteBuffer, Sequence[tuple[int, int]]]]
    ) -> list[list[np.ndarray]]:
        """Read each `(offset, num_bytes)` range of each buffer, all in flight together, into one
        staging buffer; returns uint8 arrays the caller owns, `[request][range]`.

        Example:
            [[header, rollout_3], [image]] = buffers.read(
                [(group_buffer, [(0, 16), (4096, 2048)]), (other_buffer, [(0, 5_000_000)])]
            )
        """
        for buffer, _ in requests:
            if buffer.owner_url not in self._connected_urls:
                self._transport.connect(buffer.owner_url)
                self._connected_urls.add(buffer.owner_url)
        total_bytes = sum(length for _, ranges in requests for _, length in ranges)
        if self._staging is None or self._staging[0].numel() < total_bytes:
            if self._staging is not None:
                self._transport.unregister_memory(self._staging[1])
            tensor = torch.empty(buffer_size_class(total_bytes), dtype=torch.uint8)
            self._staging = (tensor, self._transport.register_memory(tensor))
        staging_tensor, staging_memory = self._staging
        works = []
        spans = []
        position = 0
        errors = []
        try:
            for buffer, ranges in requests:
                for offset, length in ranges:
                    assert 0 <= offset and offset + length <= buffer.num_bytes
                    works.append(
                        self._transport.read(
                            staging_memory.to_mutable_view(position, length),
                            buffer.remote_buffer.slice(offset, length),
                            async_op=True,
                        )
                    )
                    spans.append((position, length))
                    position += length
        finally:
            # Wait for every submitted read before the staging buffer is used again, even when
            # one fails or a later submission raises: the others may still be writing into it.
            for work in works:
                try:
                    work.wait()
                except Exception as error:
                    errors.append(error)
        if errors:
            raise errors[0]
        staging_array = staging_tensor.numpy()
        chunks = iter(
            staging_array[start : start + length].copy() for start, length in spans
        )
        return [[next(chunks) for _ in ranges] for _, ranges in requests]


_PROBE_BYTES = bytes(range(256)) * 16


@dataclass(frozen=True, slots=True)
class RDMAProbe:
    """A small buffer a rollout worker allocates at setup, so every trainer DP replica leader can
    check that it reads that worker's buffers before the first step (`check_rdma_probes`)."""

    host: str

    buffer: RDMAByteBuffer


def make_rdma_probe(buffers: RDMAByteBuffers) -> RDMAProbe:
    """Allocate and fill a probe buffer; release it with `buffers.release(probe.buffer)`."""
    tensor, handle = buffers.allocate(len(_PROBE_BYTES))
    tensor.numpy()[:] = np.frombuffer(_PROBE_BYTES, dtype=np.uint8)
    return RDMAProbe(host=socket.gethostname(), buffer=handle)


def check_rdma_probes(buffers: RDMAByteBuffers, probes: Sequence[RDMAProbe]) -> None:
    """Read each probe whole and at a non-zero offset, and check the bytes. This also connects
    this reader to every owner, so the first step pays no connection cost.

    Raises:
        ValueError: a read failed or returned wrong bytes. A host can report RDMA hardware and
            still fail transfers (e.g. InfiniBand without a RoCEv2 GID), so this fails at setup
            instead of mid-step, naming each owner host and this reader's host.
    """
    failures = []
    for probe in probes:
        try:
            [[whole, middle]] = buffers.read(
                [(probe.buffer, [(0, 4096), (1024, 1024)])]
            )
        except Exception as error:
            failures.append(f"from {probe.host}: {type(error).__name__}: {error}")
            continue
        if (
            whole.tobytes() != _PROBE_BYTES
            or middle.tobytes() != _PROBE_BYTES[1024:2048]
        ):
            failures.append(f"from {probe.host}: wrong bytes")
    if failures:
        raise ValueError(
            f"The trainer DP replica leader on {socket.gethostname()} "
            f"(UCX_TLS={os.environ.get('UCX_TLS')}) cannot read rollout worker buffers over "
            f"torch transport: {'; '.join(failures)}. Set "
            "BaseRolloutWorker.Config.rdma_min_group_bytes=None to send rollout groups by value."
        )
