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

import pytest
import torch
from monarch.actor import Actor, endpoint, this_host
from renderers import PlaceholderRange

from torchtitan.rl.actors.monarch import rdma_object
from torchtitan.rl.actors.monarch.rdma_object import MonarchRDMAObject
from torchtitan.rl.rdma_object import RDMAImageFiles, RDMAPrompt
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
    monkeypatch.setattr(rdma_object, "RDMABuffer", _FakeRDMABuffer)
    monkeypatch.setattr(rdma_object, "RDMAAction", _FakeRDMAAction)
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
    # A sibling still running after the drop can't keep a buffer nothing would drop.
    with pytest.raises(RuntimeError, match="after drop"):
        image_files.register(_image_prompt(large))
    assert image_files.register(_image_prompt(small)) == RDMAPrompt(
        _image_prompt(small)
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
