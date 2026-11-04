# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in RDMA path for prompt image files (`Controller.Config.rdma_min_image_bytes`): the rollout
worker that registers a prompt keeps its large image files, and generator rank 0 reads them.

The actor system implements `RDMAPrompt` (e.g. `actors/monarch/rdma_prompt.py`); the controller's
`GenerateFn` and the generator use only this interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torchtitan.rl.rollout.types import Prompt


@runtime_checkable
class RDMAPrompt(Protocol):
    """A `Prompt` whose large image files stay in its owner, the process that registered it.

    Example:
        rdma_prompt = register_rdma_prompt(prompt, min_bytes=1024)  # owner
        prompt = await rdma_prompt.read()  # generator rank 0
        await rdma_prompt.drop()           # owner, after `generate` returns
    """

    async def read(self) -> Prompt:
        """Read the prompt, with every image file, from its owner."""
        ...

    async def drop(self) -> None:
        """Free the owner's buffers; on the owner, after rank 0 has read them."""
        ...


class RegisterRDMAPrompt(Protocol):
    """Registers a prompt's image files of at least `min_bytes` for RDMA reads, e.g.
    `MonarchRDMAPrompt.register`."""

    def __call__(self, prompt: Prompt, *, min_bytes: int) -> RDMAPrompt:
        ...
