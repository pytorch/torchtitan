# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in RDMA path for large bytes values sent between services: the process that registers a
value (its owner) keeps it, and the process that receives the value reads it by RDMA instead of
getting it in the message.

    rollout worker --RDMAPrompt--> generator rank 0  ==RDMA read==> rollout worker (image files)

- `RDMAObject` and `RegisterRDMAObject` are the interface. The actor system implements them
  (e.g. `actors/monarch/rdma_object.py`) and gives the registration to the rollout workers; the
  services use only this module.
- Prompt image files (`BaseRolloutWorker.Config.rdma_min_image_bytes`): `RDMAImageFiles` keeps a
  group's large files in the rollout worker, and generator rank 0 reads each `RDMAPrompt`.

Rollout groups to the trainer take another path, torch transport directly
(`rl/torch_transport_rdma.py`, `BaseRolloutWorker.Config.rdma_min_group_bytes`): its leaders read
byte ranges of pooled buffers, which this interface's whole-value reads do not offer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Protocol, TYPE_CHECKING, TypeVar

from torchtitan.observability import structured_logger as sl

if TYPE_CHECKING:
    from torchtitan.rl.rollout.types import Prompt

_ValueT = TypeVar("_ValueT", covariant=True)


class RDMAObject(Protocol[_ValueT]):
    """A value that stays in its owner, the process that registered it, when it is large.

    Example:
        rdma_file = register_rdma_object(image.data, min_bytes=1024)  # owner (a rollout worker)
        data = await rdma_file.read()  # reader (generator rank 0)
        await rdma_file.drop()         # owner, after every reader has read it
    """

    async def read(self) -> _ValueT:
        """Read the value from its owner."""
        ...

    async def drop(self) -> None:
        """Free the owner's buffers; on the owner, after every reader has read it."""
        ...


class RegisterRDMAObject(Protocol):
    """Registers a bytes value of at least `min_bytes` for RDMA reads, e.g.
    `MonarchRDMAObject.register`."""

    def __call__(self, value: bytes, *, min_bytes: int) -> RDMAObject[bytes]:
        ...


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
