# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The RDMA fast path for trainer microbatches: large tensors stay with the controller, each rank reads its own."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import torch
from monarch.rdma import RDMAAction, RDMABuffer
from torch.utils import _pytree as pytree

from torchtitan.rl.components.batcher import TrainingMicrobatch


@dataclasses.dataclass(frozen=True, slots=True)
class RDMAMicrobatch:
    """A `TrainingMicrobatch` whose large tensors stay in the controller's memory, for trainer ranks to read over RDMA.

    Example:
        rdma_microbatch = RDMAMicrobatch.register(microbatch, min_bytes=1024)  # controller
        microbatch = await rdma_microbatch.read()  # trainer: one RDMA action reads every buffer
        await rdma_microbatch.drop()               # controller, after every rank has read it
    """

    fields: dict[str, Any]
    """The microbatch's fields: tensors of at least `min_bytes` (also inside `model_kwargs`) as
    `_TensorRef`s, the rest by value. Tensors inside containers pytree doesn't know go by value too."""

    @classmethod
    def register(cls, microbatch: TrainingMicrobatch, min_bytes: int) -> RDMAMicrobatch:
        fields = {
            field.name: getattr(microbatch, field.name)
            for field in dataclasses.fields(microbatch)
        }
        return cls(
            fields=pytree.tree_map_only(
                lambda leaf: (
                    isinstance(leaf, torch.Tensor) and leaf.nbytes >= min_bytes
                ),
                _TensorRef.register,
                fields,
            )
        )

    async def read(self) -> TrainingMicrobatch:
        action = RDMAAction()

        def queue_read(tensor_ref: _TensorRef) -> torch.Tensor:
            tensor = torch.empty(tensor_ref.shape, dtype=tensor_ref.dtype)
            action.read_remote(tensor.view(-1).view(torch.uint8), tensor_ref.buffer)
            return tensor

        microbatch_fields = pytree.tree_map_only(_TensorRef, queue_read, self.fields)
        await action.submit()
        return TrainingMicrobatch(**microbatch_fields)

    async def drop(self) -> None:
        tensor_refs = [
            leaf
            for leaf in pytree.tree_leaves(self.fields)
            if isinstance(leaf, _TensorRef)
        ]
        await asyncio.gather(*(tensor_ref.buffer.drop() for tensor_ref in tensor_refs))


@dataclasses.dataclass(frozen=True, slots=True)
class _TensorRef:
    """A controller tensor as its RDMA buffer, plus the shape and dtype to rebuild it."""

    buffer: RDMABuffer
    shape: torch.Size
    dtype: torch.dtype

    @classmethod
    def register(cls, tensor: torch.Tensor) -> _TensorRef:
        # A view, not a copy: the buffer keeps the tensor's storage alive until drop().
        # view(-1) needs a contiguous tensor; the batcher builds only contiguous ones.
        return cls(
            buffer=RDMABuffer(tensor.view(-1).view(torch.uint8)),
            shape=tensor.shape,
            dtype=tensor.dtype,
        )
