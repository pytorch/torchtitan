# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from types import ModuleType
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import _get_device_handle

from .reshard_provenance import _flex_shard_all_gather_region
from .utils import _record_function_if_eager, _with_fqn

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_storage import ParamInfo


def _record_comm_if_eager(
    label: str,
    fqn: str | None,
) -> AbstractContextManager[Any]:
    """Return a c10d profiler range in eager and a no-op during compile."""
    if torch.compiler.is_compiling():
        return nullcontext()
    return dist.record_comm(_with_fqn(label, fqn))


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
class PreparedReduceScatterGrad:
    """Packed reduce-scatter inputs whose NCCL launch can be deferred."""

    send_buf: torch.Tensor
    infos: list[ParamInfo]
    placement: Any
    layout: Any
    rank: int
    world_size: int
    pg: Any
    copy_in_done: torch.Event
    device_handle: ModuleType
    debug_fqn: str | None


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
        results = _finish_all_gather(
            self.gathered,
            self.infos,
            self.mesh,
            self.per_rank_param_offsets,
            self.debug_fqn,
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
                per_rank_shards.append(gathered[r][offset : offset + numel].view(shape))
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


def _finish_all_gather(
    gathered: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    per_rank_param_offsets: list[list[int]],
    debug_fqn: str | None,
) -> list[torch.Tensor]:
    with _record_function_if_eager("FlexShard::all_gather_copy_out", debug_fqn):
        return _assemble_full_params(
            gathered,
            infos,
            mesh,
            per_rank_param_offsets,
        )


def _run_all_gather(
    gathered: list[torch.Tensor],
    send_buf: torch.Tensor,
    pg,
    debug_fqn: str | None,
) -> None:
    with _record_comm_if_eager("FlexShard::all_gather", debug_fqn):
        with _flex_shard_all_gather_region():
            dist.all_gather(gathered, send_buf, group=pg)


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
        _run_all_gather(gathered, send_buf, pg, debug_fqn)
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


def _run_reduce_scatter(
    send_buf: torch.Tensor,
    world_size: int,
    pg,
    debug_fqn: str | None,
) -> torch.Tensor:
    recv_buf = torch.empty(
        send_buf.numel() // world_size,
        dtype=send_buf.dtype,
        device=send_buf.device,
    )
    with _record_comm_if_eager("FlexShard::reduce_scatter", debug_fqn):
        # TODO: Plumb the reduction/scaling policy from SPMD gradient semantics.
        # AVG is a convenient default, but delayed grad scaling may need SUM
        # plus an explicit scale at a different point in the training step.
        dist.reduce_scatter_tensor(
            output=recv_buf,
            input=send_buf,
            op=dist.ReduceOp.AVG,
            group=pg,
        )
    return recv_buf


def _finish_reduce_scatter(
    placement: Any,
    recv_buf: torch.Tensor,
    infos: list[ParamInfo],
    layout: Any,
    rank: int,
    world_size: int,
    debug_fqn: str | None,
) -> list[torch.Tensor]:
    with _record_function_if_eager("FlexShard::reduce_scatter_copy_out", debug_fqn):
        return placement.unpack_reduced_grad(
            recv_buf,
            infos,
            layout,
            rank,
            world_size,
        )


def prepare_reduce_scatter_grad(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    debug_fqn: str | None = None,
) -> PreparedReduceScatterGrad:
    """Pack reduce-scatter inputs without launching the collective."""
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
    return PreparedReduceScatterGrad(
        send_buf=send_buf,
        infos=infos,
        placement=placement,
        layout=layout,
        rank=rank,
        world_size=ws,
        pg=pg,
        copy_in_done=copy_in_done,
        device_handle=device_handle,
        debug_fqn=debug_fqn,
    )


def launch_reduce_scatter_grad(
    prepared: PreparedReduceScatterGrad,
    reduce_scatter_stream: torch.Stream,
) -> ReduceScatterGradHandle:
    """Launch a previously packed reduce-scatter request."""
    recv_buf: torch.Tensor
    with prepared.device_handle.stream(reduce_scatter_stream):
        reduce_scatter_stream.wait_event(prepared.copy_in_done)
        recv_buf = _run_reduce_scatter(
            prepared.send_buf,
            prepared.world_size,
            prepared.pg,
            prepared.debug_fqn,
        )
        sharded_grads = _finish_reduce_scatter(
            prepared.placement,
            recv_buf,
            prepared.infos,
            prepared.layout,
            prepared.rank,
            prepared.world_size,
            prepared.debug_fqn,
        )
        event = prepared.device_handle.Event()
        event.record(reduce_scatter_stream)
    return AsyncReduceScatterResult(
        sharded_grads=sharded_grads,
        event=event,
        send_buf=prepared.send_buf,
        recv_buf=recv_buf,
        device_handle=prepared.device_handle,
        debug_fqn=prepared.debug_fqn,
    )


def begin_reduce_scatter_grad(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    reduce_scatter_stream: torch.Stream,
    debug_fqn: str | None = None,
) -> ReduceScatterGradHandle:
    """Begin a bucket reduce-scatter and return a handle for local grad shards."""
    prepared = prepare_reduce_scatter_grad(tensors, infos, mesh, debug_fqn)
    return launch_reduce_scatter_grad(prepared, reduce_scatter_stream)
