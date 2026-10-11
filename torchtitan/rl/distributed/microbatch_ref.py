# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Refs to the controller's microbatches, which each trainer rank reads over RDMA."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import torch
from monarch.rdma import RDMAAction, RDMABuffer
from torch.utils import _pytree as pytree

from torchtitan.rl.components.batcher import TrainingMicrobatch


@dataclasses.dataclass(frozen=True, slots=True)
class MicrobatchRef:
    """A `TrainingMicrobatch` left in the controller's memory, for trainer ranks to read over RDMA.

    Example:
        ref = MicrobatchRef.register(microbatch)  # controller: one RDMABuffer per non-empty tensor
        microbatch = ref.read()                   # trainer thread: one RDMA action reads them all
        await ref.drop()                          # controller, after every rank has read it
    """

    fields: dict[str, Any]
    """The microbatch's fields, with every non-empty tensor (also inside `model_kwargs`) replaced by a `_TensorRef`."""

    @classmethod
    def register(cls, microbatch: TrainingMicrobatch) -> MicrobatchRef:
        fields = {
            field.name: getattr(microbatch, field.name)
            for field in dataclasses.fields(microbatch)
        }
        # RDMABuffer rejects empty tensors, so those stay in `fields` and go by value.
        return cls(
            fields=pytree.tree_map_only(
                lambda leaf: isinstance(leaf, torch.Tensor) and leaf.numel() > 0,
                _TensorRef.register,
                fields,
            )
        )

    def read(self) -> TrainingMicrobatch:
        action = RDMAAction()

        def queue_read(tensor_ref: _TensorRef) -> torch.Tensor:
            tensor = torch.empty(tensor_ref.shape, dtype=tensor_ref.dtype)
            action.read_remote(tensor.view(-1).view(torch.uint8), tensor_ref.buffer)
            return tensor

        microbatch_fields = pytree.tree_map_only(_TensorRef, queue_read, self.fields)
        action.submit().get()
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
        # A view, not a copy: the controller's tensor must stay the storage's owner until drop().
        return cls(
            buffer=RDMABuffer(tensor.view(-1).view(torch.uint8)),
            shape=tensor.shape,
            dtype=tensor.dtype,
        )
