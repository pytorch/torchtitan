# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch's RDMA microbatch for the opt-in RDMA path (`Batcher.Config.rdma_min_tensor_bytes`): the
batcher registers each microbatch's large tensors as Monarch RDMA buffers, and each trainer rank reads its own
column.
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

from torchtitan.rl.trainer.batcher import TrainingMicrobatch

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class MonarchRDMAMicrobatch:
    """A `TrainingMicrobatch` whose large tensors stay in its owner, the process that registered it.

    Example:
        rdma_microbatch = MonarchRDMAMicrobatch.register(  # owner
            microbatch, min_bytes=1024, batch_index=0, dp_rank=1, microbatch_index=2
        )
        microbatch = await rdma_microbatch.read()  # trainer rank: one RDMA action reads every buffer
        await rdma_microbatch.drop()               # owner, after every rank has read it
    """

    fields: dict[str, Any]
    """The microbatch's fields: tensors of at least `min_bytes` (also inside `model_kwargs`) as
    `_TensorRef`s, the rest by value. Tensors inside a container that pytree can't flatten also go by value."""

    batch_index: int
    """How many batches the owner registered before this one."""

    dp_rank: int
    """The trainer DP rank that should read it."""

    microbatch_index: int
    """Its position in that rank's column."""

    @classmethod
    def register(
        cls,
        microbatch: TrainingMicrobatch,
        *,
        min_bytes: int,
        batch_index: int,
        dp_rank: int,
        microbatch_index: int,
    ) -> MonarchRDMAMicrobatch:
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
            ),
            batch_index=batch_index,
            dp_rank=dp_rank,
            microbatch_index=microbatch_index,
        )

    async def read(self) -> TrainingMicrobatch:
        action = RDMAAction()

        def queue_read(tensor_ref: _TensorRef) -> torch.Tensor:
            # TODO: allocate pinned tensors here. With max_num_documents also set, that removed the
            # ~4 ms per microbatch in `TrainingEngine._forward_backward_body`'s TODO; pinned alone did not.
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
    """Make a CUDA context current on GPU 0, so Monarch stops printing `[RdmaXcel] No CUDA context ...`
    for every registered buffer (its CUDA driver check on each pointer needs a context)."""
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
