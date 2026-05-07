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

from .utils import _with_fqn

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_storage import ParamInfo


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


class PlacementUnshardResult:
    """Result handle for a placement-owned unshard operation."""

    def finish(self) -> list[torch.Tensor]:
        """Wait for the unshard and return full parameters."""
        raise NotImplementedError

    def wait(self) -> None:
        """Wait until the unshard result is usable on the current stream."""
        raise NotImplementedError

    def release_buffers(self) -> None:
        """Release temporary buffers owned by the unshard operation."""
        raise NotImplementedError


class PlacementReduceGradResult:
    """Result handle for a placement-owned gradient reduction operation."""

    def finish(self) -> list[torch.Tensor]:
        """Wait for the reduction and return local gradient shards."""
        raise NotImplementedError

    def wait(self) -> None:
        """Wait until the reduce-grad result is usable on the current stream."""
        raise NotImplementedError

    def release_buffers(self, release_sharded_grads: bool) -> None:
        """Release temporary buffers owned by the reduce-grad operation."""
        raise NotImplementedError


@dataclass
class AsyncAllGatherResult(PlacementUnshardResult):
    """State needed to finish an async all-gather launched on a side stream."""

    gathered: list[torch.Tensor]
    infos: list[ParamInfo]
    mesh: DeviceMesh
    per_rank_param_offsets: list[list[int]]
    event: torch.Event | None
    send_buf: torch.Tensor | None
    debug_fqn: str | None

    def finish(self) -> list[torch.Tensor]:
        self.wait()
        ws = self.mesh.size()
        device = self.gathered[0].device
        with torch.profiler.record_function(
            _with_fqn("FlexShard::all_gather_copy_out", self.debug_fqn)
        ):
            results: list[torch.Tensor] = []
            for i, info in enumerate(self.infos):
                placement = info.placement
                per_rank_shards: list[torch.Tensor] = []
                for r in range(ws):
                    numel = placement.compute_local_numel(info.global_shape, r, ws)
                    shape = placement.compute_local_shape(info.global_shape, r, ws)
                    if numel > 0:
                        offset = self.per_rank_param_offsets[r][i]
                        per_rank_shards.append(
                            self.gathered[r][offset : offset + numel].view(shape)
                        )
                    else:
                        per_rank_shards.append(
                            torch.empty(shape, dtype=info.dtype, device=device)
                        )
                results.append(
                    placement.assemble_from_shards(
                        per_rank_shards, info.global_shape, info.dtype
                    )
                )
                del per_rank_shards
            self.release_buffers()
            return results

    def wait(self) -> None:
        if self.gathered:
            device = self.gathered[0].device
        elif self.send_buf is not None:
            device = self.send_buf.device
        else:
            return
        if device.type == "cuda" and self.event is not None:
            torch.cuda.current_stream(device).wait_event(self.event)

    def release_buffers(self) -> None:
        """Release raw all-gather buffers after current-stream work is queued."""
        if self.gathered:
            device = self.gathered[0].device
        elif self.send_buf is not None:
            device = self.send_buf.device
        else:
            return
        if device.type != "cuda":
            self.gathered.clear()
            self.send_buf = None
            return

        stream = torch.cuda.current_stream(device)
        event = torch.cuda.Event()
        event.record(stream)
        handoffs: list[StreamHandoff] = []
        if self.send_buf is not None:
            handoffs.append(StreamHandoff(self.send_buf, event, stream))
            self.send_buf = None
        while self.gathered:
            handoffs.append(StreamHandoff(self.gathered.pop(), event, stream))
        for handoff in handoffs:
            handoff.release()


@dataclass
class AsyncReduceScatterResult(PlacementReduceGradResult):
    """State needed to finish an async reduce-scatter launched on a side stream."""

    sharded_grads: list[torch.Tensor]
    event: torch.Event | None
    send_buf: torch.Tensor | None
    recv_buf: torch.Tensor | None
    debug_fqn: str | None

    def finish(self) -> list[torch.Tensor]:
        self.wait()
        return self.sharded_grads

    def wait(self) -> None:
        if self.recv_buf is not None:
            device = self.recv_buf.device
        elif self.sharded_grads:
            device = self.sharded_grads[0].device
        elif self.send_buf is not None:
            device = self.send_buf.device
        else:
            return
        if device.type == "cuda" and self.event is not None:
            torch.cuda.current_stream(device).wait_event(self.event)

    def release_buffers(self, release_sharded_grads: bool) -> None:
        """Release pending reduce-scatter buffers after its completion wait."""
        tensors: list[torch.Tensor] = []
        if self.send_buf is not None:
            tensors.append(self.send_buf)
            self.send_buf = None
        if self.recv_buf is not None:
            tensors.append(self.recv_buf)
            self.recv_buf = None
        if release_sharded_grads:
            tensors.extend(self.sharded_grads)
            self.sharded_grads.clear()
        if not tensors:
            return
        device = tensors[0].device
        if device.type != "cuda":
            return

        stream = torch.cuda.current_stream(device)
        event = torch.cuda.Event()
        event.record(stream)
        handoffs = [StreamHandoff(tensor, event, stream) for tensor in tensors]
        tensors.clear()
        for handoff in handoffs:
            handoff.release()
