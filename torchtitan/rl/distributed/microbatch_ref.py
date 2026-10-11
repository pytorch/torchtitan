# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Refs to the controller's microbatches, which each trainer rank reads over RDMA, a few ahead of training."""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch
from monarch.rdma import RDMAAction, RDMABuffer
from torch.utils import _pytree as pytree

from torchtitan.observability import structured_logger as sl
from torchtitan.rl.types import TrainingMicrobatch


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
            # TODO: read into pinned tensors so the engine's H2D copy runs async. That is part
            # of the ~4 ms per microbatch in the TODO in `TrainingEngine._forward_backward_body`;
            # needs a GPU benchmark.
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


# Reads queued behind the one the engine waits on. A rank holds at most 4 microbatches on
# the host: these 2, the one waited on, and the previous one, still in the engine's loop.
_NUM_MICROBATCHES_READ_AHEAD = 2


class MicrobatchStream:
    """Yields this rank's microbatches in order, reading the next ones on a background thread.

    Implements only what `TrainingEngine.forward_backward` uses: `len()` and one in-order pass.

    Example:
        stream = MicrobatchStream([ref_0, ref_1, ref_2, ref_3])
        len(stream)  # 4
        for [microbatch] in stream:
            train(microbatch)

        read thread:  read 0 | read 1, read 2 | read 3  |
        main thread:  wait   | train 0        | train 1 | train 2 | train 3
    """

    def __init__(self, refs: list[MicrobatchRef]) -> None:
        self._refs = refs

    def __len__(self) -> int:
        return len(self._refs)

    def __iter__(self) -> Iterator[list[TrainingMicrobatch]]:
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="microbatch_read")
        reads: deque[Future[TrainingMicrobatch]] = deque()
        try:
            for ref in self._refs:
                # Copy the actor's context: in a bare thread Monarch would start a second client.
                reads.append(pool.submit(contextvars.copy_context().run, ref.read))
                if len(reads) > _NUM_MICROBATCHES_READ_AHEAD:
                    yield [_wait_for_microbatch(reads.popleft())]
            while reads:
                yield [_wait_for_microbatch(reads.popleft())]
        finally:
            # On an error, cancel queued reads: with the controller gone, each would wait out
            # the 60 s RDMA timeout.
            pool.shutdown(cancel_futures=True)


def _wait_for_microbatch(read: Future[TrainingMicrobatch]) -> TrainingMicrobatch:
    # The span is the read time that training did not hide.
    with sl.log_trace_span("wait_for_microbatch"):
        return read.result()
