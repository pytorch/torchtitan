# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in RDMA path for values sent between services: the process that registers a value (its
owner) keeps the value's large tensors, arrays and bytes, and the process that receives the
value reads them by RDMA instead of getting them in the message.

    batcher        --RDMAMicrobatch-------------------> trainer rank      ==RDMA read==> batcher
    rollout worker --RDMAObject[Prompt]--> router -----> generator rank 0  ==RDMA read==> rollout worker

The actor system implements `RDMAObject` (e.g. `actors/monarch/rdma.py`); the services use only
this interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, TypeVar

_ValueT = TypeVar("_ValueT", covariant=True)
_RegisteredValueT = TypeVar("_RegisteredValueT")


@runtime_checkable
class RDMAObject(Protocol[_ValueT]):
    """A value whose large tensors, arrays and bytes stay in its owner, the process that
    registered it.

    Example:
        rdma_prompt = register_rdma_object(prompt, min_bytes=1024)  # owner (a rollout worker)
        prompt = await rdma_prompt.read()  # reader (generator rank 0)
        await rdma_prompt.drop()           # owner, after every reader has read it
    """

    async def read(self) -> _ValueT:
        """Read the value, with every large leaf, from its owner."""
        ...

    async def drop(self) -> None:
        """Free the owner's buffers; on the owner, after every reader has read it."""
        ...


class RegisterRDMAObject(Protocol):
    """Registers a value's tensors, arrays and bytes of at least `min_bytes` for RDMA reads, e.g.
    `MonarchRDMAObject.register`."""

    def __call__(
        self, value: _RegisteredValueT, *, min_bytes: int
    ) -> RDMAObject[_RegisteredValueT]:
        ...
