# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in RDMA path for trainer microbatches (`Batcher.Config.rdma_min_tensor_bytes`): the process
that registers them keeps the large tensors, and each trainer rank reads its own column over a
`torch.distributed._transport` backend.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import math
import threading
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Literal, Protocol, runtime_checkable, TYPE_CHECKING

import torch

# TODO: torch.distributed._transport is a private upstream API; switch to a public one once it exists.
from torch.distributed._transport import new_transport, wait_all
from torch.utils import _pytree as pytree

from torchtitan.observability import structured_logger as sl

if TYPE_CHECKING:
    from torchtitan.rl.trainer.batcher import TrainingMicrobatch

# Byte alignment of each tensor in a reader's buffer, so it can be viewed as any dtype.
_ALIGNMENT = 16

# Seconds a trainer rank waits for one microbatch's reads before raising.
_READ_TIMEOUT_S = 60.0

# One transport and its URL per backend per process, shared by the owner and the readers in it. The
# lock guards only creation: transports lock internally.
_transports: dict[str, tuple[Any, bytes]] = {}
_transports_lock = threading.Lock()


def _transport(backend: str) -> tuple[Any, bytes]:
    with _transports_lock:
        if backend not in _transports:
            transport = new_transport(backend)
            # Bound before any registration: NIXL's URL lists every registration live at bind() time,
            # and each remote buffer carries its own.
            _transports[backend] = (transport, transport.bind())
        return _transports[backend]


@runtime_checkable
class RDMAMicrobatch(Protocol):
    """A `TrainingMicrobatch` whose large tensors stay in its owner, the process that registered it.
    Implemented by `TransportRDMAMicrobatch`."""

    @property
    def batch_index(self) -> int:
        """How many batches the owner registered before this one."""
        ...

    @property
    def dp_rank(self) -> int:
        """The trainer DP rank that should read it."""
        ...

    @property
    def microbatch_index(self) -> int:
        """Its position in that rank's column."""
        ...

    async def read(self) -> TrainingMicrobatch:
        """Read the microbatch from its owner."""
        ...

    async def drop(self) -> None:
        """Free the owner's buffers; on the owner, after every rank has read it."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class TransportRDMAMicrobatch:
    """The `RDMAMicrobatch` over a `torch.distributed._transport` backend.

    Example:
        rdma_microbatch = TransportRDMAMicrobatch.register(  # owner
            microbatch, backend="nixl", min_bytes=1024, batch_index=0, dp_rank=1, microbatch_index=2
        )
        microbatch = await rdma_microbatch.read()  # trainer rank: one read per registered tensor
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

    backend: Literal["mooncake", "nixl"]
    """The `torch.distributed._transport` backend."""

    url: bytes
    """The owner's transport address."""

    registered: list[tuple[Any, torch.Tensor]] = dataclasses.field(
        default_factory=list, compare=False, repr=False
    )
    """Owner only, not pickled or copied: each registered tensor and its transport memory, kept until
    `drop()`."""

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            TransportRDMAMicrobatch,
            (
                self.fields,
                self.batch_index,
                self.dp_rank,
                self.microbatch_index,
                self.backend,
                self.url,
            ),
        )

    @classmethod
    def register(
        cls,
        microbatch: TrainingMicrobatch,
        *,
        backend: Literal["mooncake", "nixl"],
        min_bytes: int,
        batch_index: int,
        dp_rank: int,
        microbatch_index: int,
    ) -> TransportRDMAMicrobatch:
        transport, url = _transport(backend)
        registered: list[tuple[Any, torch.Tensor]] = []

        def register_tensor(tensor: torch.Tensor) -> _TensorRef:
            # A view, not a copy. view(-1) needs a contiguous tensor; the batcher builds only contiguous ones.
            memory = transport.register_memory(tensor.view(-1).view(torch.uint8))
            registered.append((memory, tensor))
            return _TensorRef(memory.to_remote_buffer(), tensor.shape, tensor.dtype)

        fields = {
            field.name: getattr(microbatch, field.name)
            for field in dataclasses.fields(microbatch)
        }
        return cls(
            fields=pytree.tree_map_only(
                lambda leaf: (
                    isinstance(leaf, torch.Tensor) and leaf.nbytes >= min_bytes
                ),
                register_tensor,
                fields,
            ),
            batch_index=batch_index,
            dp_rank=dp_rank,
            microbatch_index=microbatch_index,
            backend=backend,
            url=url,
            registered=registered,
        )

    async def read(self) -> TrainingMicrobatch:
        # Imported here: the batcher imports this module.
        from torchtitan.rl.trainer.batcher import TrainingMicrobatch

        tensor_refs = [
            leaf
            for leaf in pytree.tree_leaves(self.fields)
            if isinstance(leaf, _TensorRef)
        ]
        if not tensor_refs:
            return TrainingMicrobatch(**self.fields)
        offsets = []
        nbytes = 0
        for tensor_ref in tensor_refs:
            offsets.append(nbytes)
            nbytes += math.ceil(tensor_ref.nbytes / _ALIGNMENT) * _ALIGNMENT
        # TODO: read into a reused, registered pinned pool. With max_num_documents also set, pinned
        # tensors removed the ~4 ms per microbatch in `TrainingEngine._forward_backward_body`'s TODO.
        buffer = torch.empty(nbytes, dtype=torch.uint8)
        transport, _ = _transport(self.backend)
        transport.connect(self.url)
        memory = transport.register_memory(buffer)
        reads = (
            transport.read(
                memory.to_mutable_view(offset, tensor_ref.nbytes),
                tensor_ref.buffer,
                async_op=True,
            )
            for tensor_ref, offset in zip(tensor_refs, offsets)
        )
        try:
            # On a read or submit error, wait_all drains the submitted reads before raising.
            await wait_all(reads, timeout=_READ_TIMEOUT_S)
        except (TimeoutError, asyncio.CancelledError):
            # Reads may still be in flight: leave the buffer registered; the step fails.
            raise
        except BaseException:
            transport.unregister_memory(memory)
            raise
        transport.unregister_memory(memory)
        # tree_map_only visits the refs in tree_leaves order.
        tensors = iter(
            [
                buffer[offset : offset + tensor_ref.nbytes]
                .view(tensor_ref.dtype)
                .view(tensor_ref.shape)
                for tensor_ref, offset in zip(tensor_refs, offsets)
            ]
        )
        return TrainingMicrobatch(
            **pytree.tree_map_only(_TensorRef, lambda _: next(tensors), self.fields)
        )

    async def drop(self) -> None:
        transport, _ = _transport(self.backend)
        for memory, _ in self.registered:
            transport.unregister_memory(memory)
        self.registered.clear()


@dataclasses.dataclass(frozen=True, slots=True)
class _TensorRef:
    """A tensor as its owner's remote buffer, plus the shape and dtype to rebuild it."""

    buffer: Any
    """The `torch.distributed._transport.RemoteBuffer` of the tensor's bytes."""
    shape: torch.Size
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


# Reads queued behind the one the engine waits on. A rank holds at most 4 microbatches on
# the host: these 2, the one waited on, and the previous one, still in the engine's loop.
_NUM_MICROBATCHES_READ_AHEAD = 2


class RDMAMicrobatchPrefetcher:
    """Yields this rank's microbatches in order, reading the next ones on a background thread.

    Implements only what `TrainingEngine.forward_backward` uses: `len()` and one in-order pass.

    Example:
        prefetcher = RDMAMicrobatchPrefetcher([rdma_microbatch_0, ..., rdma_microbatch_3])
        len(prefetcher)  # 4
        for [microbatch] in prefetcher:
            train(microbatch)

        read thread:  read 0 | read 1, read 2 | read 3  |
        main thread:  wait   | train 0        | train 1 | train 2 | train 3
    """

    def __init__(self, rdma_microbatches: list[RDMAMicrobatch]) -> None:
        self._rdma_microbatches = rdma_microbatches
        # This rank's total read time, hidden or not; final once the pass ends.
        self.read_s = 0.0

    def __len__(self) -> int:
        return len(self._rdma_microbatches)

    def __iter__(self) -> Iterator[list[TrainingMicrobatch]]:
        def wait_for_microbatch(read: Future[TrainingMicrobatch]) -> TrainingMicrobatch:
            # The span is the read time that training did not hide.
            with sl.log_trace_span("wait_for_microbatch"):
                return read.result()

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="microbatch_read")
        reads: deque[Future[TrainingMicrobatch]] = deque()
        try:
            for rdma_microbatch in self._rdma_microbatches:
                # Copy the actor's context: in a bare thread an actor framework can start a second
                # client (Monarch does).
                # Create the coroutine there too, so a cancelled read doesn't leave one unawaited.
                reads.append(
                    pool.submit(
                        contextvars.copy_context().run, self._read, rdma_microbatch
                    )
                )
                if len(reads) > _NUM_MICROBATCHES_READ_AHEAD:
                    yield [wait_for_microbatch(reads.popleft())]
            while reads:
                yield [wait_for_microbatch(reads.popleft())]
        finally:
            # The pass ended, or the generator was closed or collected (e.g. a failed step). Cancel the queued
            # reads: with the batcher gone, each would wait out `_READ_TIMEOUT_S`. The running read finishes.
            pool.shutdown(cancel_futures=True)

    def _read(self, rdma_microbatch: RDMAMicrobatch) -> TrainingMicrobatch:
        start = time.perf_counter()
        microbatch = asyncio.run(rdma_microbatch.read())
        self.read_s += time.perf_counter() - start
        # Preprocess the images here too, so training hides it like the read.
        return microbatch.preprocess_images()
