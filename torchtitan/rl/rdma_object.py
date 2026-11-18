# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in RDMA path for values sent between services: the process that registers a value (its
owner) keeps the value's large tensors and bytes, and the process that receives the value
reads them by RDMA instead of getting them in the message.

    batcher        --RDMAMicrobatch--> trainer rank      ==RDMA read==> batcher
    rollout worker --RDMAPrompt------> generator rank 0  ==RDMA read==> rollout worker (image files)

- `RDMAObject` and `RegisterRDMAObject` are the interface. The actor system implements them
  (e.g. `actors/monarch/rdma_object.py`) and gives the registration to the batcher and the
  rollout workers; the services use only this module.
- Trainer microbatches (`Batcher.Config.rdma_min_tensor_bytes`): the batcher sends
  `RDMAMicrobatch`es, and each trainer rank reads its column with `RDMAMicrobatchPrefetcher`.
- Prompt image files (`BaseRolloutWorker.Config.rdma_min_image_bytes`): `RDMAImageFiles` keeps a
  group's large files in the rollout worker, and generator rank 0 reads each `RDMAPrompt`.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Protocol, TYPE_CHECKING, TypeVar

from torchtitan.observability import structured_logger as sl

if TYPE_CHECKING:
    from torchtitan.rl.rollout.types import Prompt
    from torchtitan.rl.trainer.batcher import TrainingMicrobatch

_ValueT = TypeVar("_ValueT", covariant=True)
_RegisteredValueT = TypeVar("_RegisteredValueT")


class RDMAObject(Protocol[_ValueT]):
    """A value whose large tensors and bytes stay in its owner, the process that registered it.

    Example:
        rdma_file = register_rdma_object(image.data, min_bytes=1024)  # owner (a rollout worker)
        data = await rdma_file.read()  # reader (generator rank 0)
        await rdma_file.drop()         # owner, after every reader has read it
    """

    async def read(self) -> _ValueT:
        """Read the value, with every large leaf, from its owner."""
        ...

    async def drop(self) -> None:
        """Free the owner's buffers; on the owner, after every reader has read it."""
        ...


class RegisterRDMAObject(Protocol):
    """Registers a value's tensors and bytes of at least `min_bytes` for RDMA reads, e.g.
    `MonarchRDMAObject.register`."""

    def __call__(
        self, value: _RegisteredValueT, *, min_bytes: int
    ) -> RDMAObject[_RegisteredValueT]:
        ...


@dataclass(frozen=True, slots=True)
class RDMAMicrobatch:
    """A `TrainingMicrobatch` whose large tensors stay in its owner, the batcher, and where it belongs
    in the batch.

    Example:
        rdma_microbatch = RDMAMicrobatch(  # owner
            microbatch=register_rdma_object(microbatch, min_bytes=1024),
            batch_index=0,
            dp_rank=1,
            microbatch_index=2,
        )
        microbatch = await rdma_microbatch.read()  # trainer rank
        await rdma_microbatch.drop()               # owner, after every rank has read it
    """

    microbatch: RDMAObject[TrainingMicrobatch]
    """The microbatch, registered for RDMA reads."""

    batch_index: int
    """How many batches the owner registered before this one."""

    dp_rank: int
    """The trainer DP rank that should read it."""

    microbatch_index: int
    """Its position in that rank's column."""

    async def read(self) -> TrainingMicrobatch:
        """Read the microbatch from its owner."""
        return await self.microbatch.read()

    async def drop(self) -> None:
        """Free the owner's buffers; on the owner, after every rank has read it."""
        await self.microbatch.drop()


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
        # Preprocess the images here too, so training hides it like the read.
        return microbatch.preprocess_images()


@dataclass(frozen=True, slots=True)
class RDMAPrompt:
    """A prompt as a rollout worker sends it to generator rank 0: the image files that stay in
    the rollout worker, and the rest of the prompt by value. Without them, it is the whole prompt.

    Example:
        image_files = RDMAImageFiles(register_rdma_object=register, min_bytes=1 << 20)
        rdma_prompt = image_files.register(prompt)  # rollout worker: a 5 MB file stays here
        prompt = await rdma_prompt.read()           # generator rank 0: one read per kept file
        await image_files.drop()                    # rollout worker, after the group's last call

        RDMAPrompt(prompt)                          # everything by value
    """

    prompt: Prompt
    """The prompt, with empty `data` in each image whose file is in `image_files`."""

    image_files: tuple[RDMAObject[bytes] | None, ...] = ()
    """Per image of `prompt.images`, the file kept in the rollout worker, or None when the file
    is in `prompt`. Empty when every file is in `prompt`."""

    async def read(self) -> Prompt:
        """The prompt with every image file, reading the kept ones from the rollout worker."""
        if all(file is None for file in self.image_files):
            return self.prompt
        with sl.log_trace_span("read_prompt_image_files"):
            files = iter(
                await asyncio.gather(
                    *(file.read() for file in self.image_files if file is not None)
                )
            )
        return replace(
            self.prompt,
            images=tuple(
                image if file is None else replace(image, data=next(files))
                for image, file in zip(
                    self.prompt.images, self.image_files, strict=True
                )
            ),
        )


class RDMAImageFiles:
    """One rollout group's image files that stay in the rollout worker for RDMA reads.

    A group's siblings carry the same images, and each turn's prompt carries every earlier
    image again, so each file of at least `min_bytes` is registered once, on the first prompt
    that carries it (keyed by `PromptImage.hash_value`), and every later prompt sends the same
    buffer. `drop` frees them all, after the group's last generate call; a file `register`
    would keep after that raises instead, since nothing would drop it.

    With `min_bytes` None, `register` sends every prompt whole.
    """

    def __init__(
        self,
        *,
        register_rdma_object: RegisterRDMAObject | None,
        min_bytes: int | None,
    ) -> None:
        assert min_bytes is None or register_rdma_object is not None
        self._register_rdma_object = register_rdma_object
        self._min_bytes = min_bytes
        self._files: dict[str, RDMAObject[bytes]] = {}
        self._dropped = False

    def register(self, prompt: Prompt) -> RDMAPrompt:
        """`prompt` as generator rank 0 reads it: its files of at least `min_bytes` stay here."""
        min_bytes = self._min_bytes
        if min_bytes is None or all(
            len(image.data) < min_bytes for image in prompt.images
        ):
            return RDMAPrompt(prompt)
        assert self._register_rdma_object is not None
        image_files: list[RDMAObject[bytes] | None] = []
        for image in prompt.images:
            if len(image.data) < min_bytes:
                image_files.append(None)
                continue
            file = self._files.get(image.hash_value)
            if file is None:
                if self._dropped:
                    raise RuntimeError(
                        "RDMAImageFiles.register after drop(): the group has finished, so "
                        "nothing would drop this file's buffer"
                    )
                file = self._register_rdma_object(image.data, min_bytes=min_bytes)
                self._files[image.hash_value] = file
            image_files.append(file)
        return RDMAPrompt(
            replace(
                prompt,
                images=tuple(
                    image if file is None else replace(image, data=b"")
                    for image, file in zip(prompt.images, image_files, strict=True)
                ),
            ),
            tuple(image_files),
        )

    async def drop(self) -> None:
        """Free every registered file; after the group's last generate call has returned."""
        self._dropped = True
        files = list(self._files.values())
        self._files.clear()
        await asyncio.gather(*(file.drop() for file in files))
