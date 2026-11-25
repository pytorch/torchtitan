# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch's `RDMAObject` (`rl/rdma_object.py`): the owner registers a large bytes value as a
Monarch RDMA buffer, and a reader reads it by RDMA. The rollout workers send prompt image files
this way (`BaseRolloutWorker.Config.rdma_min_image_bytes`).
"""

from __future__ import annotations

import dataclasses

import torch
from monarch.rdma import RDMAAction, RDMABuffer


@dataclasses.dataclass(frozen=True, slots=True)
class MonarchRDMAObject:
    """A bytes value that stays in its owner, the process that registered it, when it has at
    least `min_bytes`.

    Example (a 5 MB prompt image file stays in the rollout worker):
        rdma_file = MonarchRDMAObject.register(image.data, min_bytes=1 << 20)  # owner
        data = await rdma_file.read()  # reader: one RDMA read
        await rdma_file.drop()         # owner, after every reader has read it
    """

    value: bytes | _BytesRef
    """The bytes, or their buffer reference when they have at least `min_bytes`."""

    @classmethod
    def register(cls, value: bytes, *, min_bytes: int) -> MonarchRDMAObject:
        return cls(
            value=_BytesRef.register(value) if len(value) >= min_bytes else value
        )

    async def read(self) -> bytes:
        if isinstance(self.value, bytes):
            return self.value
        destination = torch.empty(self.value.buffer.size(), dtype=torch.uint8)
        action = RDMAAction()
        action.read_remote(destination, self.value.buffer)
        await action.submit()
        return destination.numpy().tobytes()

    async def drop(self) -> None:
        if isinstance(self.value, _BytesRef):
            await self.value.buffer.drop()


@dataclasses.dataclass(frozen=True, slots=True)
class _BytesRef:
    """A bytes object as its RDMA buffer."""

    buffer: RDMABuffer

    @classmethod
    def register(cls, data: bytes) -> _BytesRef:
        # A copy: RDMABuffer takes a writable tensor, and bytes are read-only. The buffer keeps it
        # alive until drop().
        return cls(
            buffer=RDMABuffer(torch.frombuffer(bytearray(data), dtype=torch.uint8))
        )
