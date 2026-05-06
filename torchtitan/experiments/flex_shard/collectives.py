# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .storage import ParamInfo


class StreamHandoff:
    """Hold a tensor until it is safe to release on a target stream."""

    __slots__ = (
        "_tensor",
        "_event",
        "_release_stream",
        "_device_handle",
        "_released",
    )

    def __init__(
        self,
        tensor: torch.Tensor,
        ready_event: torch.Event | None,
        release_stream: torch.Stream,
        device_handle: ModuleType | None = None,
    ) -> None:
        if device_handle is None:
            device_handle = getattr(torch, tensor.device.type)
        self._tensor: torch.Tensor | None = tensor
        self._event = ready_event
        self._release_stream = release_stream
        self._device_handle = device_handle
        self._released = False

    def wait(self, stream: torch.Stream) -> None:
        if self._released or self._event is None:
            return
        stream.wait_event(self._event)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._tensor is None:
            return
        if self._event is not None:
            self._release_stream.wait_event(self._event)
        with self._device_handle.stream(self._release_stream):
            self._tensor = None

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


@dataclass
class EagerAllGatherResult:
    """State needed to finish an eager all-gather launched on a side stream."""

    gathered: list[torch.Tensor]
    infos: list[ParamInfo]
    mesh: DeviceMesh
    per_rank_param_offsets: list[list[int]]
    event: torch.Event | None
    send_buf: torch.Tensor | None
    debug_fqn: str | None


@dataclass
class EagerReduceScatterResult:
    """State needed to finish an eager reduce-scatter launched on a side stream."""

    sharded_grads: list[torch.Tensor]
    event: torch.Event | None
    send_buf: torch.Tensor | None
    recv_buf: torch.Tensor | None
    debug_fqn: str | None
