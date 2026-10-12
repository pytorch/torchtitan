# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer microbatches as Monarch RDMA buffers: the process that registers them keeps the large
tensors, and each trainer rank reads its own column, over TCP or, with
`AsyncLoopConfig.enable_rdma`, over RDMA (ibverbs).
"""

from __future__ import annotations

import asyncio
import ctypes
import dataclasses
import logging
from typing import Any

import torch
from monarch.rdma import RDMAAction, RDMABuffer
from torch.utils import _pytree as pytree

from torchtitan.rl.components.batcher import TrainingMicrobatch

logger = logging.getLogger(__name__)

# Tensors under 1 KiB go inside the message instead of in a buffer:
#   token counts (8 B):   registering a buffer (~0.15 ms) costs more than sending the bytes
#   empty tensors (0 B):  RDMABuffer rejects them
# Per-token tensors reach 1 KiB at 1,024 tokens (bool masks), so those go over RDMA.
_MIN_RDMA_BYTES = 1024


@dataclasses.dataclass(frozen=True, slots=True)
class RDMAMicrobatch:
    """A `TrainingMicrobatch` whose large tensors stay in the controller's memory, for trainer ranks to read over RDMA.

    Example:
        rdma_microbatch = RDMAMicrobatch.register(microbatch)  # controller
        microbatch = await rdma_microbatch.read()  # trainer: one RDMA action reads every buffer
        await rdma_microbatch.drop()               # controller, after every rank has read it
    """

    fields: dict[str, Any]
    """The microbatch's fields: tensors of at least 1 KiB (also inside `model_kwargs`) as
    `_TensorRef`s, the rest by value. Tensors inside containers pytree doesn't know go by value too."""

    @classmethod
    def register(cls, microbatch: TrainingMicrobatch) -> RDMAMicrobatch:
        fields = {
            field.name: getattr(microbatch, field.name)
            for field in dataclasses.fields(microbatch)
        }
        return cls(
            fields=pytree.tree_map_only(
                lambda leaf: (
                    isinstance(leaf, torch.Tensor) and leaf.nbytes >= _MIN_RDMA_BYTES
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


# The CUDA driver's `CU_LIMIT_STACK_SIZE` (cuda.h).
_CU_LIMIT_STACK_SIZE = 0


def create_cuda_context_to_silence_monarch_log() -> None:
    """Make a CUDA context current on GPU 0, so Monarch stops logging one line per registered buffer.

    Monarch checks each registered pointer with a CUDA driver call, and prints
    `[RdmaXcel] No CUDA context ...` each time in a process without a context.
    """
    # TODO: this costs ~256 MiB on GPU 0 (H100). Remove it when Monarch prints this line once, not per
    # buffer (https://github.com/meta-pytorch/monarch/issues/5049; per buffer in torchmonarch 0.7.0).
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return
    try:
        torch.zeros(1, device="cuda")
    except RuntimeError as error:  # e.g. GPU 0 is full or in exclusive-process mode
        logger.warning(
            f"No CUDA context, so Monarch logs one line per RDMA buffer: {error}"
        )
        return
    # The driver raises the stack limit when a kernel needs more, so 0 halves the context (520 -> 256 MiB).
    ctypes.CDLL("libcuda.so.1").cuCtxSetLimit(_CU_LIMIT_STACK_SIZE, ctypes.c_size_t(0))
