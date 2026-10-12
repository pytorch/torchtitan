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
import contextvars
import ctypes
import dataclasses
import logging
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch
from monarch.rdma import RDMAAction, RDMABuffer
from torch.utils import _pytree as pytree

from torchtitan.observability import structured_logger as sl
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


# Reads queued behind the one the engine waits on. A rank holds at most 4 microbatches on
# the host: these 2, the one waited on, and the previous one, still in the engine's loop.
_NUM_MICROBATCHES_READ_AHEAD = 2


class MicrobatchStream:
    """Yields this rank's microbatches in order, reading the next ones on a background thread.

    Implements only what `TrainingEngine.forward_backward` uses: `len()` and one in-order pass.

    Example:
        stream = MicrobatchStream([rdma_microbatch_0, ..., rdma_microbatch_3])
        len(stream)  # 4
        for [microbatch] in stream:
            train(microbatch)

        read thread:  read 0 | read 1, read 2 | read 3  |
        main thread:  wait   | train 0        | train 1 | train 2 | train 3
    """

    def __init__(self, rdma_microbatches: list[RDMAMicrobatch]) -> None:
        self._rdma_microbatches = rdma_microbatches
        self.read_s = 0.0  # this rank's total read time, hidden behind training or not

    def __len__(self) -> int:
        return len(self._rdma_microbatches)

    def __iter__(self) -> Iterator[list[TrainingMicrobatch]]:
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="microbatch_read")
        reads: deque[Future[TrainingMicrobatch]] = deque()
        try:
            for rdma_microbatch in self._rdma_microbatches:
                # Copy the actor's context: in a bare thread Monarch would start a second client.
                # Create the coroutine there too, so a cancelled read doesn't leave one unawaited.
                reads.append(
                    pool.submit(
                        contextvars.copy_context().run, self._read, rdma_microbatch
                    )
                )
                if len(reads) > _NUM_MICROBATCHES_READ_AHEAD:
                    yield [_wait_for_microbatch(reads.popleft())]
            while reads:
                yield [_wait_for_microbatch(reads.popleft())]
        finally:
            # On an error, cancel queued reads: with the batcher gone, each would wait out
            # the 60 s RDMA timeout.
            pool.shutdown(cancel_futures=True)

    def _read(self, rdma_microbatch: RDMAMicrobatch) -> TrainingMicrobatch:
        start = time.perf_counter()
        microbatch = asyncio.run(rdma_microbatch.read())
        self.read_s += time.perf_counter() - start
        return microbatch


def _wait_for_microbatch(read: Future[TrainingMicrobatch]) -> TrainingMicrobatch:
    # The span is the read time that training did not hide.
    with sl.log_trace_span("wait_for_microbatch"):
        return read.result()


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
