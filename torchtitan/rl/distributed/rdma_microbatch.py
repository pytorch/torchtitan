# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in RDMA fast path for trainer microbatches (`AsyncLoopConfig.rdma_fast_path_threshold`): the
process that registers them keeps the large tensors, and each trainer rank reads its own column.
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


@dataclasses.dataclass(frozen=True, slots=True)
class RDMAMicrobatch:
    """A `TrainingMicrobatch` whose large tensors stay in its owner, the process that registered it.

    Example:
        rdma_microbatch = RDMAMicrobatch.register(microbatch, min_bytes=1024)  # owner
        microbatch = await rdma_microbatch.read()  # trainer rank: one RDMA action reads every buffer
        await rdma_microbatch.drop()               # owner, after every rank has read it
    """

    fields: dict[str, Any]
    """The microbatch's fields: tensors of at least `min_bytes` (also inside `model_kwargs`) as
    `_TensorRef`s, the rest by value. Tensors inside a container that pytree can't flatten also go by value."""

    @classmethod
    def register(
        cls, microbatch: TrainingMicrobatch, *, min_bytes: int
    ) -> RDMAMicrobatch:
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
    """A tensor as its RDMA buffer, plus the shape and dtype to rebuild it."""

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
