# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch's RDMA prompt for the opt-in RDMA path (`Controller.Config.rdma_min_image_bytes`): the
rollout worker registers a prompt's large image files as Monarch RDMA buffers, and generator rank 0
reads them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools

import torch
from monarch.rdma import RDMAAction, RDMABuffer

from torchtitan.rl.actors.monarch.rdma_microbatch import (
    create_cuda_context_to_silence_monarch_log,
)
from torchtitan.rl.rollout.types import Prompt


@dataclasses.dataclass(frozen=True, slots=True)
class MonarchRDMAPrompt:
    """A `Prompt` whose large image files stay in its owner, the process that registered it.

    Example:
        rdma_prompt = MonarchRDMAPrompt.register(prompt, min_bytes=1024)  # owner
        prompt = await rdma_prompt.read()  # generator rank 0: one RDMA action reads every file
        await rdma_prompt.drop()           # owner, after `generate` returns
    """

    prompt: Prompt
    """The prompt, with empty ``data`` on the images in `buffers`; `read` fills it in."""

    buffers: dict[int, RDMABuffer]
    """Index in ``prompt.images`` -> that image file's bytes, in the owner."""

    @classmethod
    def register(cls, prompt: Prompt, *, min_bytes: int) -> MonarchRDMAPrompt:
        # Call it on one thread (the event loop's): the CUDA context is current only there.
        _create_cuda_context_once()
        buffers = {
            # A copy: RDMABuffer takes a writable tensor, and bytes are read-only. The buffer
            # keeps it alive until drop().
            index: RDMABuffer(
                torch.frombuffer(bytearray(image.data), dtype=torch.uint8)
            )
            for index, image in enumerate(prompt.images)
            if len(image.data) >= min_bytes
        }
        images = tuple(
            dataclasses.replace(image, data=b"") if index in buffers else image
            for index, image in enumerate(prompt.images)
        )
        return cls(prompt=dataclasses.replace(prompt, images=images), buffers=buffers)

    async def read(self) -> Prompt:
        action = RDMAAction()
        files = {}
        for index, buffer in self.buffers.items():
            files[index] = torch.empty(buffer.size(), dtype=torch.uint8)
            action.read_remote(files[index], buffer)
        await action.submit()
        images = tuple(
            dataclasses.replace(image, data=files[index].numpy().tobytes())
            if index in files
            else image
            for index, image in enumerate(self.prompt.images)
        )
        return dataclasses.replace(self.prompt, images=images)

    async def drop(self) -> None:
        await asyncio.gather(*(buffer.drop() for buffer in self.buffers.values()))


@functools.cache
def _create_cuda_context_once() -> None:
    # Like the batcher's: without it, Monarch logs 2-3 lines per buffer, i.e. per image file sent.
    create_cuda_context_to_silence_monarch_log()
