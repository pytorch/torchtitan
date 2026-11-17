# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `MonarchRDMAObject` and for prompt image files a rollout worker keeps for
generator rank 0 to read (`RDMAImageFiles`, `BaseRolloutWorker.Config.rdma_min_image_bytes`),
with Monarch's RDMA classes faked except in the tests over real Monarch (TCP)."""

import asyncio
import dataclasses

import numpy as np
import torch
from monarch.actor import Actor, endpoint, this_host
from renderers import PlaceholderRange

from torchtitan.rl.actors.monarch import rdma
from torchtitan.rl.actors.monarch.rdma import (
    _ArrayRef,
    _BytesRef,
    _TensorRef,
    MonarchRDMAObject,
)
from torchtitan.rl.rollout.rdma_prompt import RDMAImageFiles, RDMAPrompt
from torchtitan.rl.rollout.types import Prompt, PromptImage


class _FakeRDMABuffer:
    """Stands in for `monarch.rdma.RDMABuffer`: keeps the owner's byte view."""

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data
        self.dropped = False

    def size(self) -> int:
        return self.data.numel()

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


@dataclasses.dataclass(frozen=True, slots=True)
class _File:
    name: str
    data: bytes


@dataclasses.dataclass(frozen=True, slots=True)
class _Settings:
    """A dataclass with no large leaf, like a microbatch's `VisionProcessor.Config`."""

    sizes: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _Payload:
    """A value with every kind of leaf, large and small, inside dataclasses and containers."""

    tensor: torch.Tensor
    array: np.ndarray
    files: tuple[_File, ...]
    extras: dict
    settings: _Settings
    note: str | None = None


def _payload() -> _Payload:
    return _Payload(
        tensor=torch.arange(64, dtype=torch.float32).view(8, 8),  # 256 B
        array=np.arange(32, dtype=np.int32),  # 128 B
        files=(
            _File(name="large", data=bytes(range(200))),
            _File(name="small", data=b"x"),
        ),
        extras={"small_tensor": torch.tensor([1, 2]), "count": 3},
        settings=_Settings(sizes=(1, 2)),
    )


def test_rdma_object_round_trips_tensor_array_and_bytes_leaves_and_drops_their_buffers(
    monkeypatch,
) -> None:
    monkeypatch.setattr(rdma, "RDMABuffer", _FakeRDMABuffer)
    monkeypatch.setattr(rdma, "RDMAAction", _FakeRDMAAction)
    payload = _payload()

    registered = MonarchRDMAObject.register(payload, min_bytes=100)

    # Leaves of at least 100 bytes stay in the owner; the rest travel with the handle.
    registered_value = registered.value
    assert isinstance(registered_value.tensor, _TensorRef)
    assert isinstance(registered_value.array, _ArrayRef)
    assert isinstance(registered_value.files[0].data, _BytesRef)
    assert registered_value.files[1].data == b"x"
    assert isinstance(registered_value.extras["small_tensor"], torch.Tensor)
    assert registered_value.extras["count"] == 3
    # A dataclass without a registered leaf is sent as is, not rebuilt.
    assert registered_value.settings is payload.settings

    read = asyncio.run(registered.read())
    assert torch.equal(read.tensor, payload.tensor)
    assert read.tensor.dtype == payload.tensor.dtype
    np.testing.assert_array_equal(read.array, payload.array)
    assert read.array.dtype == payload.array.dtype
    assert read.files == payload.files
    assert torch.equal(read.extras["small_tensor"], payload.extras["small_tensor"])
    assert read.extras["count"] == 3 and read.note is None

    asyncio.run(registered.drop())
    buffers = [
        registered_value.tensor.buffer,
        registered_value.array.buffer,
        registered_value.files[0].data.buffer,
    ]
    assert all(buffer.dropped for buffer in buffers)


class _Reader(Actor):
    """Reads a registered value in another process."""

    @endpoint
    async def read(self, registered: MonarchRDMAObject) -> object:
        return await registered.read()


def test_monarch_rdma_object_round_trips_every_leaf_kind(monkeypatch) -> None:
    # Real Monarch RDMA over TCP (works on any CI host; the reader process inherits the env var).
    monkeypatch.setenv("MONARCH_RDMA_DISABLE_IBVERBS", "1")
    payload = _payload()

    async def run() -> _Payload:
        mesh = this_host().spawn_procs(per_host={"procs": 1})
        registered = MonarchRDMAObject.register(payload, min_bytes=100)
        try:
            reader = mesh.spawn("reader", _Reader)
            return await reader.read.call_one(registered)
        finally:
            await registered.drop()
            await mesh.stop()

    read = asyncio.run(run())
    assert torch.equal(read.tensor, payload.tensor)
    np.testing.assert_array_equal(read.array, payload.array)
    assert read.files == payload.files


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


def test_image_files_keep_each_large_file_once_and_drop_them(monkeypatch) -> None:
    monkeypatch.setattr(rdma, "RDMABuffer", _FakeRDMABuffer)
    monkeypatch.setattr(rdma, "RDMAAction", _FakeRDMAAction)
    registrations = []

    def register(value, *, min_bytes):
        registrations.append(value)
        return MonarchRDMAObject.register(value, min_bytes=min_bytes)

    image_files = RDMAImageFiles(register_rdma_object=register, min_bytes=100)
    large, small = _image("large", 200), _image("small", 1)
    # A sibling renders the same file into a new bytes object; a later turn adds an image.
    sibling_large = dataclasses.replace(large, data=bytes(large.data))
    later_turn = _image_prompt(large, small, _image("new", 300))

    sent = [
        image_files.register(prompt)
        for prompt in (
            _image_prompt(large, small),
            _image_prompt(sibling_large),
            later_turn,
        )
    ]

    # Each large file is registered once, on the first prompt that carries it.
    assert registrations == [large.data, later_turn.images[2].data]
    assert sent[0].image_files[0] is sent[1].image_files[0] is sent[2].image_files[0]
    # Only the handle goes with the prompt; small files go by value.
    assert sent[0].prompt.images[0].data == b"" and sent[0].image_files[1] is None
    assert sent[0].prompt.images[1].data == small.data
    read = asyncio.run(sent[2].read())
    assert [image.data for image in read.images] == [
        image.data for image in later_turn.images
    ]
    assert read.token_ids == later_turn.token_ids

    asyncio.run(image_files.drop())
    assert all(
        file.value.buffer.dropped
        for file in (sent[2].image_files[0], sent[2].image_files[2])
    )


def test_image_files_send_the_prompt_whole_below_the_threshold_or_without_one() -> None:
    def register(value, *, min_bytes):
        raise AssertionError("nothing should be registered")

    prompt = _image_prompt(_image("small", 10))
    for image_files in (
        RDMAImageFiles(register_rdma_object=register, min_bytes=11),
        RDMAImageFiles(register_rdma_object=None, min_bytes=None),
    ):
        sent = image_files.register(prompt)
        assert sent == RDMAPrompt(prompt) and sent.prompt is prompt
        assert asyncio.run(sent.read()) is prompt


class _GeneratorRank0(Actor):
    """Generator rank 0: reads a prompt sent by a rollout worker."""

    @endpoint
    async def read_image_data(self, prompt: RDMAPrompt) -> list[bytes]:
        # Only the handle crossed from the rollout worker.
        assert [image.data for image in prompt.prompt.images] == [b""]
        return [image.data for image in (await prompt.read()).images]


def test_image_file_reaches_generator_as_a_handle(monkeypatch) -> None:
    # Real Monarch RDMA over TCP: this process is the rollout worker that owns the file's buffer.
    monkeypatch.setenv("MONARCH_RDMA_DISABLE_IBVERBS", "1")
    image = _image("photo", 2048)
    image_files = RDMAImageFiles(
        register_rdma_object=MonarchRDMAObject.register, min_bytes=1024
    )

    async def run() -> list[bytes]:
        generator_mesh = this_host().spawn_procs(per_host={"procs": 1})
        try:
            generator = generator_mesh.spawn("generator", _GeneratorRank0)
            return await generator.read_image_data.call_one(
                image_files.register(_image_prompt(image))
            )
        finally:
            await image_files.drop()
            await generator_mesh.stop()

    assert asyncio.run(run()) == [image.data]
