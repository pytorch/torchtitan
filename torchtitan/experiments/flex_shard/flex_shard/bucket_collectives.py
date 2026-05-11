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
import torch.distributed as dist
from torch.distributed.device_mesh import _get_device_handle

from .utils import _record_function_if_eager, _with_fqn

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
            device_handle = _get_device_handle(tensor.device.type)
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


class AllGatherUnshardHandle:
    """Handle for a FlexShard bucket all-gather unshard operation."""

    def finish(self) -> list[torch.Tensor]:
        """Wait for the unshard and return full parameters."""
        raise NotImplementedError

    def wait(self) -> None:
        """Wait until the unshard result is usable on the current stream."""
        raise NotImplementedError

    def release_buffers(self) -> None:
        """Release temporary buffers owned by the unshard operation."""
        raise NotImplementedError


class ReduceScatterGradHandle:
    """Handle for a FlexShard bucket reduce-scatter grad operation."""

    def finish(self) -> list[torch.Tensor]:
        """Wait for the reduction and return local gradient shards."""
        raise NotImplementedError

    def wait(self) -> None:
        """Wait until the reduce-grad result is usable on the current stream."""
        raise NotImplementedError

    def release_buffers(self, release_sharded_grads: bool) -> None:
        """Release temporary buffers owned by the reduce-grad operation."""
        raise NotImplementedError

    def record_sharded_grads(
        self,
        sharded_grads: list[torch.Tensor],
        stream: torch.Stream,
    ) -> None:
        """Track sharded grads until queued work on stream is complete."""
        raise NotImplementedError


@dataclass
class AsyncAllGatherResult(AllGatherUnshardHandle):
    """State needed to finish an async all-gather launched on a side stream."""

    gathered: list[torch.Tensor]
    infos: list[ParamInfo]
    mesh: DeviceMesh
    per_rank_param_offsets: list[list[int]]
    event: torch.Event | None
    send_buf: torch.Tensor | None
    device_handle: ModuleType
    debug_fqn: str | None

    def finish(self) -> list[torch.Tensor]:
        self.wait()
        with _record_function_if_eager(
            "FlexShard::all_gather_copy_out", self.debug_fqn
        ):
            results = _assemble_full_params(
                self.gathered,
                self.infos,
                self.mesh,
                self.per_rank_param_offsets,
            )
            self.release_buffers()
            return results

    def wait(self) -> None:
        if self.gathered:
            device = self.gathered[0].device
        elif self.send_buf is not None:
            device = self.send_buf.device
        else:
            return
        if self.event is not None:
            self.device_handle.current_stream(device).wait_event(self.event)

    def release_buffers(self) -> None:
        """Release raw all-gather buffers after current-stream work is queued."""
        if self.gathered:
            device = self.gathered[0].device
        elif self.send_buf is not None:
            device = self.send_buf.device
        else:
            return

        stream = self.device_handle.current_stream(device)
        event = self.device_handle.Event()
        event.record(stream)
        handoffs: list[StreamHandoff] = []
        if self.send_buf is not None:
            handoffs.append(
                StreamHandoff(self.send_buf, event, stream, self.device_handle)
            )
            self.send_buf = None
        while self.gathered:
            handoffs.append(
                StreamHandoff(self.gathered.pop(), event, stream, self.device_handle)
            )
        for handoff in handoffs:
            handoff.release()


def _assemble_full_params(
    gathered: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    per_rank_param_offsets: list[list[int]],
) -> list[torch.Tensor]:
    ws = mesh.size()
    device = gathered[0].device
    results: list[torch.Tensor] = []
    for i, info in enumerate(infos):
        placement = info.placement
        per_rank_shards: list[torch.Tensor] = []
        for r in range(ws):
            numel = placement.compute_local_numel(info.global_shape, r, ws)
            shape = placement.compute_local_shape(info.global_shape, r, ws)
            if numel > 0:
                offset = per_rank_param_offsets[r][i]
                per_rank_shards.append(
                    gathered[r][offset : offset + numel].view(shape)
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
    return results


def begin_all_gather_unshard(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    all_gather_stream: torch.Stream,
    debug_fqn: str | None = None,
) -> AllGatherUnshardHandle:
    """Begin a bucket all-gather and return a handle for full params."""
    ws = mesh.size()
    pg = mesh.get_group()
    dtype = tensors[0].dtype
    device = tensors[0].device
    device_handle = _get_device_handle(device.type)

    with _record_function_if_eager("FlexShard::all_gather_copy_in", debug_fqn):
        send_buf = torch.cat([t.reshape(-1) for t in tensors])

        per_rank_sizes: list[int] = []
        per_rank_param_offsets: list[list[int]] = []
        for r in range(ws):
            offset = 0
            offsets_r: list[int] = []
            for info in infos:
                offsets_r.append(offset)
                offset += info.placement.compute_local_numel(info.global_shape, r, ws)
            per_rank_sizes.append(offset)
            per_rank_param_offsets.append(offsets_r)

        gathered = [
            torch.empty(per_rank_sizes[r], dtype=dtype, device=device)
            for r in range(ws)
        ]

    copy_in_done = device_handle.Event()
    copy_in_done.record(device_handle.current_stream(device))
    with device_handle.stream(all_gather_stream):
        all_gather_stream.wait_event(copy_in_done)
        label = _with_fqn("FlexShard::all_gather", debug_fqn)
        with dist.record_comm(label):
            dist.all_gather(gathered, send_buf, group=pg)
        event = device_handle.Event()
        event.record(all_gather_stream)
    return AsyncAllGatherResult(
        gathered=gathered,
        infos=infos,
        mesh=mesh,
        per_rank_param_offsets=per_rank_param_offsets,
        event=event,
        send_buf=send_buf,
        device_handle=device_handle,
        debug_fqn=debug_fqn,
    )


@dataclass
class AsyncReduceScatterResult(ReduceScatterGradHandle):
    """State needed to finish an async reduce-scatter launched on a side stream."""

    sharded_grads: list[torch.Tensor]
    event: torch.Event | None
    send_buf: torch.Tensor | None
    recv_buf: torch.Tensor | None
    device_handle: ModuleType
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
        if self.event is not None:
            self.device_handle.current_stream(device).wait_event(self.event)

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

        stream = self.device_handle.current_stream(device)
        event = self.device_handle.Event()
        event.record(stream)
        handoffs = [
            StreamHandoff(tensor, event, stream, self.device_handle)
            for tensor in tensors
        ]
        tensors.clear()
        for handoff in handoffs:
            handoff.release()

    def record_sharded_grads(
        self,
        sharded_grads: list[torch.Tensor],
        stream: torch.Stream,
    ) -> None:
        self.sharded_grads = sharded_grads
        self.event = self.device_handle.Event()
        self.event.record(stream)


def begin_reduce_scatter_grad(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    reduce_scatter_stream: torch.Stream,
    debug_fqn: str | None = None,
) -> ReduceScatterGradHandle:
    """Begin a bucket reduce-scatter and return a handle for local grad shards."""
    ws = mesh.size()
    rank = mesh.get_local_rank()
    pg = mesh.get_group()
    placement = infos[0].placement
    for info in infos[1:]:
        if info.placement != placement:
            raise ValueError(
                "FlexShard bucket reduce-scatter requires all parameters in a "
                f"bucket to use the same placement, but {infos[0].fqn!r} uses "
                f"{placement!r} and {info.fqn!r} uses {info.placement!r}."
            )

    with _record_function_if_eager("FlexShard::reduce_scatter_copy_in", debug_fqn):
        send_buf, layout = placement.pack_reduce_grad(tensors, infos, ws)
        if send_buf.numel() % ws != 0:
            raise AssertionError(
                f"Packed reduce-scatter buffer has {send_buf.numel()} elements, "
                f"which is not divisible by world size {ws}."
            )

    device = send_buf.device
    device_handle = _get_device_handle(device.type)
    copy_in_done = device_handle.Event()
    copy_in_done.record(device_handle.current_stream(device))

    recv_buf: torch.Tensor
    with device_handle.stream(reduce_scatter_stream):
        reduce_scatter_stream.wait_event(copy_in_done)
        recv_buf = torch.empty(
            send_buf.numel() // ws,
            dtype=send_buf.dtype,
            device=device,
        )
        label = _with_fqn("FlexShard::reduce_scatter", debug_fqn)
        with dist.record_comm(label):
            dist.reduce_scatter_tensor(
                output=recv_buf,
                input=send_buf,
                op=dist.ReduceOp.AVG,
                group=pg,
            )
        with _record_function_if_eager(
            "FlexShard::reduce_scatter_copy_out", debug_fqn
        ):
            sharded_grads = placement.unpack_reduced_grad(
                recv_buf, infos, layout, rank, ws
            )
        event = device_handle.Event()
        event.record(reduce_scatter_stream)
    return AsyncReduceScatterResult(
        sharded_grads=sharded_grads,
        event=event,
        send_buf=send_buf,
        recv_buf=recv_buf,
        device_handle=device_handle,
        debug_fqn=debug_fqn,
    )
