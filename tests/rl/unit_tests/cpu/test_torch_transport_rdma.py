# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.rl.torch_transport_rdma`: the registered-buffer pool, range reads,
the setup probe, and the prompt image files a rollout worker keeps for generator rank 0
(`RDMAImageFiles`, `BaseRolloutWorker.Config.rdma_min_image_bytes`).

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
from renderers import PlaceholderRange

from torchtitan.rl.rollout.types import Prompt, PromptImage
from torchtitan.rl.torch_transport_rdma import (
    buffer_size_class,
    check_rdma_probes,
    make_rdma_probe,
    RDMAByteBuffers,
    RDMAImageFiles,
    RDMAPrompt,
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
    names = {"reader": "Generator rank 0", "knob": "rdma_min_image_bytes"}
    check_rdma_probes(reader, [probe], **names)
    reader._transport.fail_reads = True

    with pytest.raises(
        ValueError,
        match=r"Generator rank 0 on .*rollout-host-3.*Config\.rdma_min_image_bytes=None",
    ):
        check_rdma_probes(reader, [probe], **names)


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


def _image(name: str, num_bytes: int) -> PromptImage:
    return PromptImage(
        hash_value=name,
        source_path=f"{name}.png",
        grid_thw=(1, 2, 2),
        data=name.encode()[:1] * num_bytes,
    )


def _image_prompt(*images: PromptImage) -> Prompt:
    return Prompt(
        token_ids=[1, 2, 3],
        images=images,
        image_placeholders=tuple(PlaceholderRange(offset=0, length=1) for _ in images),
    )


def test_image_files_keep_each_large_file_once_and_release_them() -> None:
    owner, reader = loopback_buffers()
    image_files = RDMAImageFiles(buffers=owner, min_bytes=100, group_id=7)
    large, small = _image("large", 200), _image("small", 1)
    # A sibling renders the same file into a new bytes object; a later turn adds an image.
    sibling_large = replace(large, data=bytes(large.data))
    later_turn = _image_prompt(large, small, _image("new", 300))

    sent = [
        image_files.register(prompt)
        for prompt in (
            _image_prompt(large, small),
            _image_prompt(sibling_large),
            later_turn,
        )
    ]

    # Each large file takes one buffer, on the first prompt that carries it.
    assert owner.pool.num_registrations == 2
    assert sent[0].image_files[0] is sent[1].image_files[0] is sent[2].image_files[0]
    # Only the handle goes with the prompt; small files go by value.
    assert sent[0].prompt.images[0].data == b"" and sent[0].image_files[1] is None
    assert sent[0].prompt.images[1].data == small.data
    read = sent[2].read(reader)
    assert [image.data for image in read.images] == [
        image.data for image in later_turn.images
    ]
    assert read.token_ids == later_turn.token_ids

    image_files.release()
    assert owner.pool.bytes_in_use == 0
    # A sibling still running after the release can't keep a buffer nothing would release.
    with pytest.raises(RuntimeError, match="after release"):
        image_files.register(_image_prompt(large))
    assert image_files.register(_image_prompt(small)) == RDMAPrompt(
        _image_prompt(small)
    )


def test_image_files_send_the_prompt_whole_below_the_threshold_or_without_one() -> None:
    owner, reader = loopback_buffers()
    prompt = _image_prompt(_image("small", 10))
    for image_files in (
        RDMAImageFiles(buffers=owner, min_bytes=11, group_id=1),
        RDMAImageFiles(buffers=None, min_bytes=None, group_id=1),
    ):
        sent = image_files.register(prompt)
        assert sent == RDMAPrompt(prompt) and sent.prompt is prompt
        assert not sent.has_kept_image_files
        assert sent.read(reader) is prompt
    assert owner.pool.num_registrations == 0


def test_a_read_after_the_worker_reused_the_files_buffer_raises() -> None:
    """A group's file buffer goes back to the pool when its generate calls end, and the next
    allocation of the same size class reuses it, e.g. the same group's packed buffer: same
    `group_id`, so only the pool's per-allocation generation tells the two uses apart. A late
    read of the file's handle then sees the new tags instead of returning the new bytes as the
    file."""
    owner, reader = loopback_buffers()
    first_use = RDMAImageFiles(buffers=owner, min_bytes=100, group_id=1)
    stale = first_use.register(_image_prompt(_image("a", 1000)))
    first_use.release()
    second_use = RDMAImageFiles(buffers=owner, min_bytes=100, group_id=1)
    fresh = second_use.register(_image_prompt(_image("b", 1000)))

    assert owner.pool.num_registrations == 1
    assert fresh.read(reader).images[0].data == b"b" * 1000
    with pytest.raises(RuntimeError, match=r"group 1 \(generation 1\).*\(1, 2\)"):
        stale.read(reader)
