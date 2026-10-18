# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in RDMA path for trainer microbatches (`Batcher.Config.rdma_min_tensor_bytes`): the process
that registers them keeps the large tensors, and each trainer rank reads its own column.

The actor system implements `RDMAMicrobatch` (e.g. `actors/monarch/rdma_microbatch.py`); the batcher and
the trainer use only this interface.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol, runtime_checkable, TYPE_CHECKING

from torchtitan.observability import structured_logger as sl

if TYPE_CHECKING:
    from torchtitan.rl.trainer.batcher import TrainingMicrobatch


@runtime_checkable
class RDMAMicrobatch(Protocol):
    """A `TrainingMicrobatch` whose large tensors stay in its owner, the process that registered it.

    Example:
        rdma_microbatch = register_rdma_microbatch(  # owner
            microbatch, min_bytes=1024, batch_index=0, dp_rank=1, microbatch_index=2
        )
        microbatch = await rdma_microbatch.read()  # trainer rank
        await rdma_microbatch.drop()               # owner, after every rank has read it
    """

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


class RegisterRDMAMicrobatch(Protocol):
    """Registers a microbatch's tensors of at least `min_bytes` for RDMA reads, e.g.
    `MonarchRDMAMicrobatch.register`."""

    def __call__(
        self,
        microbatch: TrainingMicrobatch,
        *,
        min_bytes: int,
        batch_index: int,
        dp_rank: int,
        microbatch_index: int,
    ) -> RDMAMicrobatch:
        ...


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
            # reads: with the batcher gone, each would wait out the 60 s RDMA timeout. The running read finishes.
            pool.shutdown(cancel_futures=True)

    def _read(self, rdma_microbatch: RDMAMicrobatch) -> TrainingMicrobatch:
        start = time.perf_counter()
        microbatch = asyncio.run(rdma_microbatch.read())
        self.read_s += time.perf_counter() - start
        return microbatch
