# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Byte buffers that stay in the process that wrote them (their owner) and that other processes
read by byte range over torch transport (`torch.distributed._transport`, NIXL backend). A rollout
worker keeps two kinds of large values in them, from one pool:
- prompt image files (`BaseRolloutWorker.Config.rdma_min_image_bytes`): `RDMAImageFiles` keeps a
  group's large files, and generator rank 0 reads them whole (`RDMAPrompt.read`);
- packed training groups (`BaseRolloutWorker.Config.rdma_min_group_bytes`,
  `rollout/packed_group_handle.py`): the trainer's DP replica leaders read only the rollouts they
  train on.
A group's image files go back to the pool when its last generate call returns, before the group
is packed, so later groups' files and packed groups reuse the same registered buffers.

    owner (rollout worker)                         reader (generator rank 0 or DP replica leader)
    tensor, handle = buffers.allocate(n)
    write tensor, send handle  ------ handle ----> [[chunk]] = buffers.read([(handle, [(64, 128)])])
                                                   (one-sided reads of RemoteBuffer.slice)
    buffers.release(handle), after the last read: back to the pool, still registered

Why buffers are reused instead of unregistered: each NIXL read loads the owner's metadata for the
buffer into the reader, and the owner's unregister never reaches the reader. Once the owner
registers a new buffer over an address range it freed, a reader's lookup can match the freed
buffer and fail (`NIXL_ERR_NOT_FOUND`) or read through a deregistered key. A pool that never
unregisters keeps every buffer a reader has loaded valid.
"""

from __future__ import annotations

import itertools
import os
import socket
import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch

# TODO: torch.distributed._transport is a private upstream API; switch to a public one once it
# exists (no upstream issue yet; the same TODO as in model_state_dict_transport.py).
if TYPE_CHECKING:
    from torch.distributed._transport import Memory, RemoteBuffer, Transport

    from torchtitan.rl.rollout.types import Prompt

# The smallest buffer the pool registers; smaller requests share its size.
_MIN_BUFFER_BYTES = 64 * 1024

# What an owner writes before and after each region a reader must tell from a later use of the
# same pooled buffer: (group_id, `RDMAByteBuffer.generation`). Image files and packed groups share
# the pool, so the generation, stamped per allocation, is what makes a tag name one use.
REGION_TAG = struct.Struct("<qq")


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

    generation: int
    """The owner's count of allocations, so it names this use of the pooled buffer: readers check
    it in the region tags (`REGION_TAG`)."""


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
                    f"{type(error).__name__}: {error}. Rollout groups and prompt image files "
                    "by RDMA need it; set BaseRolloutWorker.Config.rdma_min_group_bytes and "
                    "rdma_min_image_bytes to None to send them by value."
                ) from error
        self._transport = transport
        self._url = transport.bind()
        self.pool = RegisteredBufferPool(transport)
        self._generations = itertools.count(1)
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
            generation=next(self._generations),
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
    """A small buffer a rollout worker allocates at setup, so each reader of its buffers (every
    trainer DP replica leader for packed groups, rank 0 of every generator for prompt image
    files) checks that it reads them before the first step (`check_rdma_probes`)."""

    host: str

    buffer: RDMAByteBuffer


def make_rdma_probe(buffers: RDMAByteBuffers) -> RDMAProbe:
    """Allocate and fill a probe buffer; release it with `buffers.release(probe.buffer)`."""
    tensor, handle = buffers.allocate(len(_PROBE_BYTES))
    tensor.numpy()[:] = np.frombuffer(_PROBE_BYTES, dtype=np.uint8)
    return RDMAProbe(host=socket.gethostname(), buffer=handle)


def check_rdma_probes(
    buffers: RDMAByteBuffers, probes: Sequence[RDMAProbe], *, reader: str, knob: str
) -> None:
    """Read each probe whole and at a non-zero offset, and check the bytes. This also connects
    this reader to every owner, so the first step pays no connection cost.

    Args:
        reader: who reads, for the error, e.g. "The trainer DP replica leader".
        knob: the `BaseRolloutWorker.Config` field that turns this reader's RDMA path off, for
            the error, e.g. "rdma_min_group_bytes".

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
            f"{reader} on {socket.gethostname()} (UCX_TLS={os.environ.get('UCX_TLS')}) cannot "
            f"read rollout worker buffers over torch transport: {'; '.join(failures)}. Set "
            f"BaseRolloutWorker.Config.{knob}=None to send these values in the calls instead."
        )


@dataclass(frozen=True, slots=True)
class RDMAImageFile:
    """A prompt image file kept in its rollout worker's registered buffer, laid out as
    `| tag | file | tag |` with tag `(group_id, buffer.generation)` (`REGION_TAG`). The tags catch
    a read made after the worker reused the buffer; the release order rules out a read racing the
    reuse (the worker releases a group's files only after the group's last generate call has
    returned, and the generator reads them before it generates).

    TODO: that order holds because nothing cancels a generate call (`Controller._make_generate_fn`).
    If something does, a read racing the reuse can pass the tag check with a torn middle, and the
    engine would cache the pixels under `hash_value`; release only after rank 0's read returns.
    """

    buffer: RDMAByteBuffer
    """Both tags and the file."""

    group_id: int

    @property
    def generation(self) -> int:
        return self.buffer.generation

    def file_bytes(self, data: np.ndarray) -> bytes:
        """The file, from the bytes read from `buffer`.

        Raises:
            RuntimeError: the tags differ from this file's, i.e. the worker reused the buffer
                before this read (a bug in the release order).
        """
        tag = REGION_TAG.pack(self.group_id, self.generation)
        head = data[: REGION_TAG.size].tobytes()
        tail = data[-REGION_TAG.size :].tobytes()
        if head != tag or tail != tag:
            raise RuntimeError(
                f"Image file of rollout group {self.group_id} (generation {self.generation}): "
                f"its buffer holds {REGION_TAG.unpack(head)} ... "
                f"{REGION_TAG.unpack(tail)}, so the rollout worker reused it before this read."
            )
        return data[REGION_TAG.size : -REGION_TAG.size].tobytes()


@dataclass(frozen=True, slots=True)
class RDMAPrompt:
    """A prompt as a rollout worker sends it to generator rank 0: the image files that stay in
    the rollout worker, and the rest of the prompt by value. Without them, it is the whole prompt.

    Example:
        image_files = RDMAImageFiles(buffers=worker_buffers, min_bytes=1 << 20, group_id=7)
        rdma_prompt = image_files.register(prompt)          # rollout worker: a 5 MB file stays
        prompt = rdma_prompt.read(generator_buffers)        # generator rank 0: one read call
        image_files.release()                               # rollout worker, after the group's
                                                            # last generate call

        RDMAPrompt(prompt)                                  # everything by value
    """

    prompt: Prompt
    """The prompt, with empty `data` in each image whose file is in `image_files`."""

    image_files: tuple[RDMAImageFile | None, ...] = ()
    """Per image of `prompt.images`, the file kept in the rollout worker, or None when the file
    is in `prompt`. Empty when every file is in `prompt`."""

    @property
    def has_kept_image_files(self) -> bool:
        return any(file is not None for file in self.image_files)

    def read(self, buffers: RDMAByteBuffers) -> Prompt:
        """The prompt with every image file: the kept ones are read from the rollout worker with
        `buffers`, all in flight together. Blocks until the reads finish, so call it off the
        event loop, on the thread that uses `buffers`."""
        if not self.has_kept_image_files:
            return self.prompt
        kept_files = [file for file in self.image_files if file is not None]
        chunks = buffers.read(
            [(file.buffer, [(0, file.buffer.num_bytes)]) for file in kept_files]
        )
        files = iter(
            file.file_bytes(chunk)
            for file, [chunk] in zip(kept_files, chunks, strict=True)
        )
        return replace(
            self.prompt,
            images=tuple(
                image if file is None else replace(image, data=next(files))
                for image, file in zip(
                    self.prompt.images, self.image_files, strict=True
                )
            ),
        )


class RDMAImageFiles:
    """One rollout group's image files that stay in the rollout worker, in buffers from its
    `RDMAByteBuffers` pool, for generator rank 0 to read.

    A group's siblings carry the same images, and each turn's prompt carries every earlier
    image again, so each file of at least `min_bytes` is copied into a buffer once, on the first
    prompt that carries it (keyed by `PromptImage.hash_value`), and every later prompt sends the
    same handle. `release` returns them all to the pool, after the group's last generate call; a
    file `register` would keep after that raises instead, since nothing would release it.

    With `buffers` None, `register` sends every prompt whole.
    """

    def __init__(
        self,
        *,
        buffers: RDMAByteBuffers | None,
        min_bytes: int | None,
        group_id: int,
    ) -> None:
        assert (buffers is None) == (min_bytes is None)
        self._buffers = buffers
        self._min_bytes = min_bytes
        self._group_id = group_id
        self._files: dict[str, RDMAImageFile] = {}
        self._released = False

    def register(self, prompt: Prompt) -> RDMAPrompt:
        """`prompt` as generator rank 0 reads it: its files of at least `min_bytes` stay here."""
        min_bytes = self._min_bytes
        if min_bytes is None or all(
            len(image.data) < min_bytes for image in prompt.images
        ):
            return RDMAPrompt(prompt)
        image_files: list[RDMAImageFile | None] = []
        for image in prompt.images:
            if len(image.data) < min_bytes:
                image_files.append(None)
                continue
            file = self._files.get(image.hash_value)
            if file is None:
                if self._released:
                    raise RuntimeError(
                        "RDMAImageFiles.register after release(): the group has finished, so "
                        "nothing would release this file's buffer"
                    )
                file = self._keep(image.data)
                self._files[image.hash_value] = file
            image_files.append(file)
        return RDMAPrompt(
            replace(
                prompt,
                images=tuple(
                    image if file is None else replace(image, data=b"")
                    for image, file in zip(prompt.images, image_files, strict=True)
                ),
            ),
            tuple(image_files),
        )

    def release(self) -> None:
        """Return every kept file's buffer to the pool; after the group's last generate call has
        returned."""
        self._released = True
        if self._buffers is not None:
            for file in self._files.values():
                self._buffers.release(file.buffer)
        self._files.clear()

    def _keep(self, data: bytes) -> RDMAImageFile:
        assert self._buffers is not None
        tag_bytes = REGION_TAG.size
        tensor, buffer = self._buffers.allocate(len(data) + 2 * tag_bytes)
        tag = np.frombuffer(
            REGION_TAG.pack(self._group_id, buffer.generation), dtype=np.uint8
        )
        destination = tensor.numpy()
        destination[:tag_bytes] = tag
        destination[tag_bytes:-tag_bytes] = np.frombuffer(data, dtype=np.uint8)
        destination[-tag_bytes:] = tag
        return RDMAImageFile(buffer=buffer, group_id=self._group_id)
