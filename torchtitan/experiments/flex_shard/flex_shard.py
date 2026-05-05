# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import fnmatch
import logging
import sys
from collections.abc import Callable, Generator

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for
from torch.distributed.fsdp import DataParallelMeshDims


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


@dataclass
class EagerAllGatherContext:
    """Communication streams for eager batched collectives."""

    all_gather_stream: torch.Stream
    reduce_scatter_stream: torch.Stream
    buckets: list[EagerAllGatherBucket] = field(default_factory=list)
    pending: PendingEagerAllGather | None = None
    reduce_scatter_states: list[EagerReduceScatterResult] = field(default_factory=list)
    reduce_scatter_callback_queued: bool = False


@dataclass
class EagerAllGatherBucket:
    """Runtime metadata for one eager batched all-gather bucket."""

    storage: DStorage
    entries: list[tuple[nn.Module, str, nn.Module, ParamInfo]]
    infos: list[ParamInfo]
    debug_fqn: str | None
    use_autograd_unshard: bool


@dataclass
class PendingEagerAllGather:
    """The single one-bucket-ahead eager all-gather in flight."""

    bucket: EagerAllGatherBucket
    result: EagerAllGatherResult
    recompute: bool


@dataclass
class EagerAllGatherResult:
    """State needed to finish an eager all-gather launched on a side stream."""

    gathered: list[torch.Tensor]
    infos: list[ParamInfo]
    mesh: DeviceMesh
    per_rank_param_offsets: list[list[int]]
    event: torch.Event | None
    send_buf: torch.Tensor
    debug_fqn: str | None


@dataclass
class EagerBucketAllGatherRuntime:
    """Runtime metadata passed to eager RAF bucket autograd."""

    prefetched_result: EagerAllGatherResult | None
    infos: list[ParamInfo]
    param_refs: list[tuple[nn.Module, str]]
    mesh: DeviceMesh
    context: EagerAllGatherContext
    debug_fqn: str | None


@dataclass
class EagerReduceScatterResult:
    """State needed to finish an eager reduce-scatter launched on a side stream."""

    sharded_grads: list[torch.Tensor]
    event: torch.Event | None
    send_buf: torch.Tensor
    recv_buf: torch.Tensor
    debug_fqn: str | None


def _with_fqn(label: str, fqn: str | None) -> str:
    """Append a module/bucket FQN to profiler labels, matching FSDP style."""
    if fqn:
        return f"{label} ({fqn})"
    return label


def _queue_reduce_scatter_wait(context: EagerAllGatherContext) -> None:
    """Queue a post-backward wait for eager reduce-scatter work."""
    if context.reduce_scatter_callback_queued:
        return
    context.reduce_scatter_callback_queued = True

    def _wait_for_reduce_scatter() -> None:
        try:
            for result in context.reduce_scatter_states:
                Shard.wait_for_reduce_grad(result)
        finally:
            context.reduce_scatter_states.clear()
            context.reduce_scatter_callback_queued = False

    torch.autograd.Variable._execution_engine.queue_callback(_wait_for_reduce_scatter)


def _wait_and_clear_reduce_scatter_states(
    context: EagerAllGatherContext,
    debug_fqn: str | None,
) -> None:
    """Wait for prior eager reduce-scatter states and release their buffers."""
    if not context.reduce_scatter_states:
        return
    with torch.profiler.record_function(
        _with_fqn("FlexShard::post_backward_rs_wait", debug_fqn)
    ):
        for result in context.reduce_scatter_states:
            Shard.wait_for_reduce_grad(result)
        context.reduce_scatter_states.clear()


class Placement:
    """Base class for FlexShard placement strategies.

    Each subclass implements per-param sharding (extract_local_shard,
    assemble_from_shards) and batched communication (unshard, reduce_grad).
    """

    def compute_local_numel(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> int:
        """How many elements this rank holds for a param with global_shape."""
        raise NotImplementedError

    def compute_local_shape(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> torch.Size:
        """Local shape this rank holds for a param with global_shape."""
        raise NotImplementedError

    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        """Extract this rank's local shard from the full (unsharded) param.

        Returns a contiguous typed tensor. DStorage handles copying into
        the byte buffer.

        Args:
            param: the full (unsharded) parameter tensor
            rank: this rank's index
            world_size: total number of ranks
        """
        raise NotImplementedError

    def assemble_from_shards(
        self,
        per_rank_shards: list[torch.Tensor],
        global_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reconstruct the full unsharded param from typed per-rank shards.

        DStorage extracts typed shards from gathered byte buffers and passes
        them here. Each shard is a contiguous typed tensor.

        Args:
            per_rank_shards: list of typed tensors, one per rank
            global_shape: the full unsharded shape
            dtype: parameter dtype
        """
        raise NotImplementedError

    def create_parametrization(
        self,
        info: ParamInfo,
        group_name: str,
        world_size: int,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = None,
        compute_device: torch.device | None = None,
    ) -> nn.Module:
        """Create an FX-traceable parametrization module for this placement.

        Called by flex_shard() to build the property-getter parametrization
        that emits _c10d_functional ops. Subclasses override to return their
        specific parametrization class.
        """
        raise NotImplementedError

    @classmethod
    def unshard(
        cls,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        """Batched gather communication for all params in a storage unit."""
        raise NotImplementedError

    @classmethod
    def reduce_grad(
        cls,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        """Batched reduce communication for all param gradients in a storage unit."""
        raise NotImplementedError


class Shard(Placement):
    """Symmetric sharding — parameter split along dim across all ranks."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Shard):
            return False
        return self.dim == other.dim

    def __hash__(self) -> int:
        return hash((type(self), self.dim))

    def __repr__(self) -> str:
        return f"Shard({self.dim})"

    def create_parametrization(self, info, group_name, world_size, **kwargs):
        dim_size = info.global_shape[self.dim]
        uneven = dim_size % world_size != 0
        return ShardParametrization(
            shard_dim=self.dim,
            group_name=group_name,
            world_size=world_size,
            padded_shard_size=(
                (dim_size + world_size - 1) // world_size if uneven else None
            ),
            global_dim_size=dim_size if uneven else None,
            **kwargs,
        )

    def compute_local_shape(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> torch.Size:
        dim_size = global_shape[self.dim]
        chunk = (dim_size + world_size - 1) // world_size
        start = chunk * rank
        local_dim = min(dim_size, start + chunk) - min(dim_size, start)
        shape = list(global_shape)
        shape[self.dim] = local_dim
        return torch.Size(shape)

    def compute_local_numel(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> int:
        local_shape = self.compute_local_shape(global_shape, rank, world_size)
        numel = 1
        for d in local_shape:
            numel *= d
        return numel

    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        chunks = list(torch.chunk(param, world_size, dim=self.dim))
        while len(chunks) < world_size:
            chunks.append(chunks[0].new_empty(0))
        return chunks[rank].contiguous()

    def assemble_from_shards(
        self,
        per_rank_shards: list[torch.Tensor],
        global_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        non_empty = [s for s in per_rank_shards if s.numel() > 0]
        if non_empty:
            return torch.cat(non_empty, dim=self.dim)
        return torch.empty(global_shape, dtype=dtype, device=per_rank_shards[0].device)

    @classmethod
    def unshard(
        cls,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        result = cls.begin_unshard(
            tensors,
            infos,
            mesh,
            all_gather_stream=None,
            debug_fqn=None,
        )
        return cls.finish_unshard(result)

    @classmethod
    def begin_unshard(
        cls,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        all_gather_stream: torch.Stream | None,
        debug_fqn: str | None = None,
    ) -> EagerAllGatherResult:
        ws = mesh.size()
        pg = mesh.get_group()
        dtype = tensors[0].dtype
        device = tensors[0].device

        # Pack local shards into one flat buffer
        with torch.profiler.record_function(
            _with_fqn("FlexShard::all_gather_copy_in", debug_fqn)
        ):
            send_buf = torch.cat([t.reshape(-1) for t in tensors])

            # Compute per-rank buffer sizes and param offsets
            per_rank_sizes: list[int] = []
            per_rank_param_offsets: list[list[int]] = []
            for r in range(ws):
                offset = 0
                offsets_r: list[int] = []
                for info in infos:
                    offsets_r.append(offset)
                    offset += info.placements[0].compute_local_numel(
                        info.global_shape, r, ws
                    )
                per_rank_sizes.append(offset)
                per_rank_param_offsets.append(offsets_r)

            # Variable-size all_gather outputs
            gathered = [
                torch.empty(per_rank_sizes[r], dtype=dtype, device=device)
                for r in range(ws)
            ]

        if all_gather_stream is None or device.type != "cuda":
            label = _with_fqn("FlexShard::all_gather", debug_fqn)
            with dist.record_comm(label):
                dist.all_gather(gathered, send_buf, group=pg)
            event = None
            if device.type == "cuda":
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
            return EagerAllGatherResult(
                gathered=gathered,
                infos=infos,
                mesh=mesh,
                per_rank_param_offsets=per_rank_param_offsets,
                event=event,
                send_buf=send_buf,
                debug_fqn=debug_fqn,
            )

        copy_in_done = torch.cuda.Event()
        copy_in_done.record(torch.cuda.current_stream(device))
        with torch.cuda.stream(all_gather_stream):
            all_gather_stream.wait_event(copy_in_done)
            label = _with_fqn("FlexShard::all_gather", debug_fqn)
            with dist.record_comm(label):
                dist.all_gather(gathered, send_buf, group=pg)
            event = torch.cuda.Event()
            event.record(all_gather_stream)
        send_buf.record_stream(all_gather_stream)
        for tensor in gathered:
            tensor.record_stream(all_gather_stream)
        return EagerAllGatherResult(
            gathered=gathered,
            infos=infos,
            mesh=mesh,
            per_rank_param_offsets=per_rank_param_offsets,
            event=event,
            send_buf=send_buf,
            debug_fqn=debug_fqn,
        )

    @classmethod
    def finish_unshard(cls, result: EagerAllGatherResult) -> list[torch.Tensor]:
        cls.wait_for_unshard(result)
        ws = result.mesh.size()
        device = result.gathered[0].device
        # Unpack: per param, extract shard from each rank, assemble_from_shards
        with torch.profiler.record_function(
            _with_fqn("FlexShard::all_gather_copy_out", result.debug_fqn)
        ):
            results: list[torch.Tensor] = []
            for i, info in enumerate(result.infos):
                p = info.placements[0]
                per_rank_shards: list[torch.Tensor] = []
                for r in range(ws):
                    numel = p.compute_local_numel(info.global_shape, r, ws)
                    shape = p.compute_local_shape(info.global_shape, r, ws)
                    if numel > 0:
                        off = result.per_rank_param_offsets[r][i]
                        per_rank_shards.append(
                            result.gathered[r][off : off + numel].view(shape)
                        )
                    else:
                        per_rank_shards.append(
                            torch.empty(shape, dtype=info.dtype, device=device)
                        )
                results.append(
                    p.assemble_from_shards(
                        per_rank_shards, info.global_shape, info.dtype
                    )
                )
            return results

    @classmethod
    def wait_for_unshard(cls, result: EagerAllGatherResult) -> None:
        device = result.gathered[0].device
        if device.type == "cuda" and result.event is not None:
            torch.cuda.current_stream(device).wait_event(result.event)

    @classmethod
    def reduce_grad(
        cls,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        result = cls.begin_reduce_grad(
            tensors,
            infos,
            mesh,
            reduce_scatter_stream=None,
            debug_fqn=None,
        )
        return cls.finish_reduce_grad(result)

    @classmethod
    def begin_reduce_grad(
        cls,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        reduce_scatter_stream: torch.Stream | None,
        debug_fqn: str | None = None,
    ) -> EagerReduceScatterResult:
        ws = mesh.size()
        rank = mesh.get_local_rank()
        pg = mesh.get_group()
        dtype = tensors[0].dtype
        device = tensors[0].device

        with torch.profiler.record_function(
            _with_fqn("FlexShard::reduce_scatter_copy_in", debug_fqn)
        ):
            padded_sizes = []
            for tensor in tensors:
                padded_dim0 = ((tensor.size(0) + ws - 1) // ws) * ws
                padded_sizes.append(torch.Size([padded_dim0] + list(tensor.shape[1:])))

            input_numel = sum(s.numel() for s in padded_sizes)
            output_numel = input_numel // ws

            send_buf = torch.empty(input_numel, dtype=dtype, device=device)
            send_buf_2d = send_buf.view(ws, -1)
            torch._chunk_cat(tensors, dim=0, num_chunks=ws, out=send_buf_2d)

        def _copy_out(recv_buf: torch.Tensor) -> list[torch.Tensor]:
            with torch.profiler.record_function(
                _with_fqn("FlexShard::reduce_scatter_copy_out", debug_fqn)
            ):
                results: list[torch.Tensor] = []
                flat_offset = 0
                for info, padded_size in zip(infos, padded_sizes, strict=True):
                    local_shape = info.placements[0].compute_local_shape(
                        info.global_shape, rank, ws
                    )
                    stride = make_contiguous_strides_for(local_shape)
                    shard = torch.as_strided(
                        recv_buf,
                        size=local_shape,
                        stride=stride,
                        storage_offset=flat_offset,
                    ).contiguous()
                    results.append(shard)
                    flat_offset += padded_size.numel() // ws
                return results

        if reduce_scatter_stream is None or device.type != "cuda":
            recv_buf = torch.empty(output_numel, dtype=dtype, device=device)
            label = _with_fqn("FlexShard::reduce_scatter", debug_fqn)
            with dist.record_comm(label):
                dist.reduce_scatter_tensor(
                    output=recv_buf,
                    input=send_buf,
                    op=dist.ReduceOp.AVG,
                    group=pg,
                )
            sharded_grads = _copy_out(recv_buf)
            event = None
            if device.type == "cuda":
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
            return EagerReduceScatterResult(
                sharded_grads=sharded_grads,
                event=event,
                send_buf=send_buf,
                recv_buf=recv_buf,
                debug_fqn=debug_fqn,
            )

        copy_in_done = torch.cuda.Event()
        copy_in_done.record(torch.cuda.current_stream(device))
        recv_buf: torch.Tensor
        with torch.cuda.stream(reduce_scatter_stream):
            reduce_scatter_stream.wait_event(copy_in_done)
            recv_buf = torch.empty(output_numel, dtype=dtype, device=device)
            label = _with_fqn("FlexShard::reduce_scatter", debug_fqn)
            with dist.record_comm(label):
                dist.reduce_scatter_tensor(
                    output=recv_buf,
                    input=send_buf,
                    op=dist.ReduceOp.AVG,
                    group=pg,
                )
            sharded_grads = _copy_out(recv_buf)
            event = torch.cuda.Event()
            event.record(reduce_scatter_stream)
        send_buf.record_stream(reduce_scatter_stream)
        recv_buf.record_stream(reduce_scatter_stream)
        for grad in sharded_grads:
            grad.record_stream(reduce_scatter_stream)
        return EagerReduceScatterResult(
            sharded_grads=sharded_grads,
            event=event,
            send_buf=send_buf,
            recv_buf=recv_buf,
            debug_fqn=debug_fqn,
        )

    @classmethod
    def finish_reduce_grad(cls, result: EagerReduceScatterResult) -> list[torch.Tensor]:
        cls.wait_for_reduce_grad(result)
        return result.sharded_grads

    @classmethod
    def wait_for_reduce_grad(cls, result: EagerReduceScatterResult) -> None:
        device = result.recv_buf.device
        if device.type == "cuda" and result.event is not None:
            torch.cuda.current_stream(device).wait_event(result.event)


class _EagerBucketAllGather(torch.autograd.Function):
    """Autograd boundary for eager RAF bucket all-gather.

    Forward consumes a raw all-gather result, either prefetched by the previous
    bucket or launched on demand. Backward packs full-parameter gradients and
    launches one explicit bucket reduce-scatter.
    """

    @staticmethod
    def forward(
        ctx: Any,
        runtime: EagerBucketAllGatherRuntime,
        *local_shards: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.runtime = runtime
        ctx.num_inputs = len(local_shards)

        result = runtime.prefetched_result
        if result is None:
            result = Shard.begin_unshard(
                [shard.detach() for shard in local_shards],
                runtime.infos,
                runtime.mesh,
                runtime.context.all_gather_stream,
                debug_fqn=runtime.debug_fqn,
            )
        full_params = Shard.finish_unshard(result)
        return tuple(
            _set_tensor_reshard_after_forward(full_param, True)
            for full_param in full_params
        )

    @staticmethod
    def backward(
        ctx: Any,
        *full_param_grads: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        runtime: EagerBucketAllGatherRuntime = ctx.runtime
        grads: list[torch.Tensor] = []
        valid_infos: list[ParamInfo] = []
        valid_param_refs: list[tuple[nn.Module, str]] = []
        for grad, info, param_ref in zip(
            full_param_grads,
            runtime.infos,
            runtime.param_refs,
            strict=True,
        ):
            if grad is None:
                continue
            grads.append(grad.contiguous())
            valid_infos.append(info)
            valid_param_refs.append(param_ref)

        if grads:
            with torch.no_grad():
                _wait_and_clear_reduce_scatter_states(
                    runtime.context,
                    runtime.debug_fqn,
                )
                result = Shard.begin_reduce_grad(
                    grads,
                    valid_infos,
                    runtime.mesh,
                    runtime.context.reduce_scatter_stream,
                    debug_fqn=runtime.debug_fqn,
                )
                stored_grads: list[torch.Tensor] = []
                with torch.cuda.stream(runtime.context.reduce_scatter_stream):
                    for (leaf, name), grad in zip(
                        valid_param_refs,
                        result.sharded_grads,
                        strict=True,
                    ):
                        param = leaf._parameters[name]
                        if grad.dtype != param.dtype:
                            grad = grad.to(param.dtype)
                        stored_grads.append(grad)
                        if param.grad is None:
                            param.grad = grad
                        else:
                            param.grad += grad
                    result.sharded_grads = stored_grads
                    result.event = torch.cuda.Event()
                    result.event.record(runtime.context.reduce_scatter_stream)
                runtime.context.reduce_scatter_states.append(result)
                _queue_reduce_scatter_wait(runtime.context)

        # Gradients are accumulated into the original sharded parameters above
        # so the autograd input grads can stay empty and avoid blocking here.
        return (None, *([None] * ctx.num_inputs))


@dataclass
class FlatShard(Placement):
    """FSDP1-style flat sharding — all params flattened into one 1D tensor,
    divided evenly across ranks. A single param may straddle rank boundaries.

    Each FlatShard instance describes one param's position in the global flat
    buffer via flat_offset and numel. total_flat_numel is the total size of the
    buffer (sum of all param numels).

    Attrs:
        flat_offset: start position of this param in the global flat buffer
        numel: number of elements in this param
        total_flat_numel: total elements across all params in the flat buffer
    """

    flat_offset: int = 0
    numel: int = 0
    total_flat_numel: int = 0

    def create_parametrization(self, info, group_name, world_size, **kwargs):
        numel = info.global_numel
        flat_uneven = numel % world_size != 0
        return FlatShardParametrization(
            group_name=group_name,
            world_size=world_size,
            original_shape=info.global_shape,
            padded_shard_size=(
                (numel + world_size - 1) // world_size if flat_uneven else None
            ),
            global_numel=numel if flat_uneven else None,
            **kwargs,
        )

    def _intersection(self, rank: int, world_size: int) -> tuple[int, int]:
        """Compute overlap between this param's flat range and a rank's flat range.

        Returns:
            overlap: number of elements in the intersection
            offset_in_param: where the intersection starts within this param
        """
        chunk = (self.total_flat_numel + world_size - 1) // world_size
        r_start = rank * chunk
        r_end = min((rank + 1) * chunk, self.total_flat_numel)
        p_start = self.flat_offset
        p_end = self.flat_offset + self.numel
        overlap = max(0, min(r_end, p_end) - max(r_start, p_start))
        offset_in_param = max(0, r_start - p_start)
        return overlap, offset_in_param

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlatShard):
            return False
        return (
            self.flat_offset == other.flat_offset
            and self.numel == other.numel
            and self.total_flat_numel == other.total_flat_numel
        )

    def __hash__(self) -> int:
        return hash((type(self), self.flat_offset, self.numel, self.total_flat_numel))

    def __repr__(self) -> str:
        return (
            f"FlatShard(flat_offset={self.flat_offset}, numel={self.numel}, "
            f"total_flat_numel={self.total_flat_numel})"
        )

    def compute_local_numel(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> int:
        overlap, _ = self._intersection(rank, world_size)
        return overlap

    def compute_local_shape(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> torch.Size:
        overlap, _ = self._intersection(rank, world_size)
        return torch.Size([overlap])

    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        overlap, offset_in_param = self._intersection(rank, world_size)
        if overlap == 0:
            return param.new_empty(0)
        return param.reshape(-1)[
            offset_in_param : offset_in_param + overlap
        ].contiguous()

    def assemble_from_shards(
        self,
        per_rank_shards: list[torch.Tensor],
        global_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        non_empty = [s for s in per_rank_shards if s.numel() > 0]
        if non_empty:
            return torch.cat(non_empty).view(global_shape)
        return torch.empty(global_shape, dtype=dtype, device=per_rank_shards[0].device)

    @classmethod
    def unshard(
        cls,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        ws = mesh.size()
        pg = mesh.get_group()
        total_flat = infos[0].placements[0].total_flat_numel
        chunk = (total_flat + ws - 1) // ws
        dtype = tensors[0].dtype
        device = tensors[0].device

        # Cat local shards = this rank's chunk of the flat buffer
        non_empty = [t for t in tensors if t.numel() > 0]
        send_buf = (
            torch.cat(non_empty)
            if non_empty
            else torch.empty(0, dtype=dtype, device=device)
        )
        # Pad to uniform chunk size for all_gather_into_tensor
        if send_buf.numel() < chunk:
            padded = torch.zeros(chunk, dtype=dtype, device=device)
            padded[: send_buf.numel()].copy_(send_buf)
            send_buf = padded

        recv_buf = torch.empty(chunk * ws, dtype=dtype, device=device)
        dist.all_gather_into_tensor(recv_buf, send_buf, group=pg)

        # Extract each param from the full flat buffer
        full_flat = recv_buf[:total_flat]
        results: list[torch.Tensor] = []
        for info in infos:
            p = info.placements[0]
            results.append(
                full_flat[p.flat_offset : p.flat_offset + p.numel]
                .view(info.global_shape)
                .contiguous()
            )
        return results

    @classmethod
    def reduce_grad(
        cls,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        ws = mesh.size()
        rank = mesh.get_local_rank()
        pg = mesh.get_group()
        total_flat = infos[0].placements[0].total_flat_numel
        chunk = (total_flat + ws - 1) // ws
        dtype = tensors[0].dtype
        device = tensors[0].device

        # Flatten and concatenate all grads
        flat_grads = torch.cat([g.reshape(-1) for g in tensors])
        total = flat_grads.numel()

        # Pad to world_size-divisible
        padded_total = chunk * ws
        if padded_total > total:
            send_buf = torch.zeros(padded_total, dtype=dtype, device=device)
            send_buf[:total].copy_(flat_grads)
        else:
            send_buf = flat_grads

        recv_buf = torch.empty(chunk, dtype=dtype, device=device)
        dist.reduce_scatter_tensor(
            output=recv_buf,
            input=send_buf,
            op=dist.ReduceOp.AVG,
            group=pg,
        )

        # Extract per-param sharded grads from recv_buf
        results: list[torch.Tensor] = []
        for info in infos:
            p = info.placements[0]
            overlap, _ = p._intersection(rank, ws)
            r_start = rank * chunk
            offset_in_recv = max(0, p.flat_offset - r_start)
            results.append(
                recv_buf[offset_in_recv : offset_in_recv + overlap].contiguous()
            )
        return results


__all__ = [
    "auto_buckets",
    "BucketSpec",
    "disable_active_parametrization",
    "DStorage",
    "FlexShardMeshInfo",
    "flat_shard_placements",
    "FlatShard",
    "FlatShardParametrization",
    "flex_shard",
    "FlexShardModule",
    "get_global_shape",
    "get_placements",
    "is_flex_shard_param",
    "lift_params_to_global_spmd_mesh",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "Owned",
    "param_boundary_placements",
    "per_param_placements",
    "Placement",
    "RaggedShard",
    "set_sharding_info",
    "Shard",
    "ShardParametrization",
]


# Module attribute names for storing DStorage
_DSTORAGE_ATTR = "_dstorage"
_DSTORAGES_ATTR = "_dstorages"

# Hidden attribute names for placement metadata on plain tensors
_PLACEMENTS_ATTR = "_placements"
_GLOBAL_SHAPE_ATTR = "_global_shape"
_GLOBAL_STRIDE_ATTR = "_global_stride"
_MESH_ATTR = "_mesh"
_SPMD_MESH_ATTR = "_spmd_mesh"
_SPMD_PLACEMENTS_ATTR = "_spmd_placements"
_SPMD_DP_DIM_INDICES_ATTR = "_spmd_dp_dim_indices"
_RESHARD_AFTER_FORWARD_ATTR = "_flex_shard_reshard_after_forward"
_REQUIRES_EAGER_BATCHED_UNSHARD_ATTR = "_flex_shard_requires_eager_batched_unshard"
_EAGER_BATCHED_HOOK_REGISTERED_ATTR = "_flex_shard_eager_batched_hook_registered"
_EAGER_COMM_CONTEXTS_ATTR = "_flex_shard_eager_comm_contexts"
_PARAM_FQN_ATTR = "_flex_shard_param_fqn"
_BUCKET_FQN_ATTR = "_flex_shard_bucket_fqn"
_EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR = "_flex_shard_eager_autograd_bucket_unshard"


@dataclass(frozen=True)
class FlexShardMeshInfo:
    """Mesh metadata for FlexShard's DP and global SPMD views.

    ``dp_shard_mesh`` is the one-dimensional mesh used for FlexShard
    collectives and storage sharding. ``spmd_mesh`` is the full mesh the
    model parameters live on when ``dp_mesh_dims`` is provided.
    """

    dp_shard_mesh: DeviceMesh
    spmd_mesh: DeviceMesh
    dp_mesh_dims: DataParallelMeshDims
    dp_shard_dim_names: tuple[str, ...] = ()
    dp_replicate_dim_names: tuple[str, ...] = ()
    dp_dim_indices: tuple[int, ...] = ()

    @property
    def is_spmd_mesh(self) -> bool:
        return True


def _validate_flex_shard_mesh(
    mesh: DeviceMesh,
    dp_mesh_dims: DataParallelMeshDims,
) -> None:
    """Validate mesh inputs for FlexShard global SPMD mode."""
    if dp_mesh_dims.shard is None:
        raise ValueError("flex_shard requires dp_mesh_dims.shard to be set")
    if dp_mesh_dims.replicate is not None:
        raise NotImplementedError(
            "flex_shard global SPMD mode does not yet support " "dp_mesh_dims.replicate"
        )
    if mesh.mesh_dim_names is None:
        raise ValueError("mesh must have mesh_dim_names when dp_mesh_dims is provided")

    mesh_names = tuple(mesh.mesh_dim_names)
    dim_names = dp_mesh_dims.shard_names + dp_mesh_dims.replicate_names
    if len(set(dim_names)) != len(dim_names):
        raise ValueError(f"dp_mesh_dims contains duplicate mesh dim names: {dim_names}")
    for name in dim_names:
        if name not in mesh_names:
            raise ValueError(
                f"Mesh dim name {name!r} not found in mesh.mesh_dim_names "
                f"{mesh_names}"
            )


def _get_submesh(mesh: DeviceMesh, names: tuple[str, ...]) -> DeviceMesh:
    """Return one mesh dim or flatten several named mesh dims."""
    if len(names) == 1:
        return mesh[names[0]]
    return mesh[names]._flatten("_".join(names))


def _get_flex_shard_mesh_info(
    mesh: DeviceMesh,
    dp_mesh_dims: DataParallelMeshDims,
) -> FlexShardMeshInfo:
    """Derive FlexShard's DP shard mesh from a global SPMD mesh."""
    _validate_flex_shard_mesh(mesh, dp_mesh_dims)

    assert mesh.mesh_dim_names is not None
    shard_names = dp_mesh_dims.shard_names
    replicate_names = dp_mesh_dims.replicate_names
    dp_dim_names = shard_names + replicate_names
    dp_dim_indices = tuple(mesh.mesh_dim_names.index(name) for name in dp_dim_names)
    return FlexShardMeshInfo(
        dp_shard_mesh=_get_submesh(mesh, shard_names),
        spmd_mesh=mesh,
        dp_mesh_dims=dp_mesh_dims,
        dp_shard_dim_names=shard_names,
        dp_replicate_dim_names=replicate_names,
        dp_dim_indices=dp_dim_indices,
    )


def _maybe_dtensor_local_tensor(param: torch.Tensor) -> torch.Tensor:
    """Return the DTensor local tensor if ``param`` is a DTensor."""
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(param, DTensor):
            return param.to_local()
    except ImportError:
        pass
    return param


def _get_param_spmd_metadata(
    param: torch.Tensor,
) -> tuple[DeviceMesh | None, tuple[Any, ...] | None]:
    """Return full-SPMD metadata from either a DTensor or annotated tensor."""
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(param, DTensor):
            return param._spec.mesh, tuple(param._spec.placements)
    except ImportError:
        pass

    spmd_mesh = getattr(param, _SPMD_MESH_ATTR, None)
    spmd_placements = getattr(param, _SPMD_PLACEMENTS_ATTR, None)
    if spmd_mesh is None or spmd_placements is None:
        return None, None
    return spmd_mesh, tuple(spmd_placements)


def _get_param_tensor_for_flex_shard(
    param: torch.Tensor,
    mesh_info: FlexShardMeshInfo,
    fqn: str,
    *,
    contiguous: bool,
) -> torch.Tensor:
    """Return the tensor FlexShard should shard over the DP mesh.

    DTensor parameters already carry local tensors for all non-DP placements.
    For parameters annotated by ``model.parallelize(materialize_state=False)``,
    derive the same non-DP local tensor view directly from the full plain tensor
    while leaving DP dimensions replicated for FlexShard.
    """
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(param, DTensor):
            return param.to_local()
    except ImportError:
        pass

    spmd_mesh, spmd_placements = _get_param_spmd_metadata(param)
    if spmd_mesh is None or spmd_placements is None:
        return param

    coordinate = spmd_mesh.get_coordinate()
    if coordinate is None:
        raise ValueError(
            f"Current rank is not part of the SPMD mesh for parameter {fqn!r}."
        )

    from torch.distributed.tensor import Replicate, Shard as DTensorShard

    local = param.detach()
    dp_dims = set(mesh_info.dp_dim_indices)
    for mesh_dim, placement in enumerate(spmd_placements):
        if mesh_dim in dp_dims or isinstance(placement, Replicate):
            continue
        if isinstance(placement, DTensorShard):
            local = placement._select_split_tensor(
                local,
                spmd_mesh.size(mesh_dim=mesh_dim),
                coordinate[mesh_dim],
                with_padding=False,
                contiguous=contiguous,
                clone=False,
            )
            continue
        raise NotImplementedError(
            "model.parallelize(..., materialize_state=False) supports "
            f"Replicate and Shard non-DP parameter placements, but {fqn!r} "
            f"uses {placement} on mesh dim {mesh_dim}."
        )
    if contiguous and not local.is_contiguous():
        local = local.contiguous()
    return local


def _same_device_mesh(lhs: DeviceMesh, rhs: DeviceMesh) -> bool:
    """Return whether two DeviceMesh objects describe the same mesh."""
    if lhs is rhs:
        return True
    if lhs.device_type != rhs.device_type:
        return False
    if lhs.mesh_dim_names != rhs.mesh_dim_names:
        return False
    return torch.equal(lhs.mesh, rhs.mesh)


def _get_device_from_mesh(mesh: DeviceMesh) -> torch.device:
    """Return the current rank's device for ``mesh``."""
    if mesh.device_type == "cpu":
        return torch.device("cpu")
    if mesh.device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    try:
        device_module = torch.get_device_module(mesh.device_type)
    except (AttributeError, RuntimeError):
        return torch.device(mesh.device_type)
    return torch.device(mesh.device_type, device_module.current_device())


def _validate_global_spmd_params(
    named_params: list[tuple[str, nn.Parameter]],
    mesh_info: FlexShardMeshInfo,
    expected_device: torch.device | None = None,
) -> None:
    """Validate parameters for global SPMD mode."""
    from torch.distributed.tensor import DTensor, Replicate

    for fqn, param in named_params:
        param_is_dtensor = isinstance(param, DTensor)
        spmd_mesh, placements = _get_param_spmd_metadata(param)
        if spmd_mesh is None or placements is None:
            raise ValueError(
                "flex_shard with dp_mesh_dims expects all parameters to be "
                "DTensors on the full SPMD mesh or plain tensors annotated by "
                "model.parallelize(..., materialize_state=False), but "
                f"{fqn!r} is {type(param).__name__}."
            )
        if not _same_device_mesh(spmd_mesh, mesh_info.spmd_mesh):
            raise ValueError(
                f"Parameter {fqn!r} is on mesh {spmd_mesh}, but "
                f"flex_shard was given SPMD mesh {mesh_info.spmd_mesh}."
            )
        if len(placements) != mesh_info.spmd_mesh.ndim:
            raise ValueError(
                f"Parameter {fqn!r} has {len(placements)} placements, but "
                f"SPMD mesh has {mesh_info.spmd_mesh.ndim} dims."
            )
        for dim in mesh_info.dp_dim_indices:
            if not isinstance(placements[dim], Replicate):
                raise ValueError(
                    f"Parameter {fqn!r} has non-Replicate placement "
                    f"{placements[dim]} on data-parallel mesh dim {dim}. "
                    "FlexShard global SPMD mode expects DP dims to be "
                    "Replicate() before sharding."
                )
        if expected_device is not None:
            if spmd_mesh.device_type != expected_device.type:
                raise ValueError(
                    f"Parameter {fqn!r} is on a "
                    f"{spmd_mesh.device_type!r} SPMD mesh, but "
                    f"FlexShard mesh device is {expected_device.type!r}."
                )
            local = (
                param.to_local()
                if param_is_dtensor
                else _get_param_tensor_for_flex_shard(
                    param, mesh_info, fqn, contiguous=False
                )
            )
            valid_plain_source = not param_is_dtensor and local.device.type == "cpu"
            if (
                local.device.type != "meta"
                and local.device != expected_device
                and not valid_plain_source
            ):
                raise ValueError(
                    f"Parameter {fqn!r} has local tensor on {local.device}, but "
                    f"FlexShard expected {expected_device}. Use "
                    "model.parallelize(..., materialize_state=False) for "
                    "plain CPU initialization, or distribute the model onto "
                    "the target SPMD mesh before calling flex_shard()."
                )


def lift_params_to_global_spmd_mesh(module: nn.Module, mesh: DeviceMesh) -> None:
    """Lift plain or submesh DTensor parameters onto a full named SPMD mesh.

    FlexShard's public API expects parameters to already be DTensors on the
    global mesh with data-parallel dims replicated. This helper is a migration
    utility for callers that still initialize plain parameters or TP-only
    DTensors before calling ``flex_shard``.
    """
    from torch.distributed.tensor import DTensor, Replicate

    if mesh.mesh_dim_names is None:
        raise ValueError("global SPMD mesh must have mesh_dim_names")
    spmd_names = tuple(mesh.mesh_dim_names)

    for submodule in module.modules():
        for name, param in list(submodule._parameters.items()):
            if param is None:
                continue

            if isinstance(param, DTensor) and _same_device_mesh(param._spec.mesh, mesh):
                continue

            placements = [Replicate() for _ in range(mesh.ndim)]
            if isinstance(param, DTensor):
                source_mesh = param._spec.mesh
                source_names = source_mesh.mesh_dim_names
                if source_names is None:
                    if source_mesh.ndim == 1 and "tp" in spmd_names:
                        source_names = ("tp",)
                    else:
                        raise ValueError(
                            "Cannot lift DTensor parameter from unnamed mesh "
                            f"{source_mesh} to SPMD mesh {mesh}"
                        )
                for source_name, placement in zip(
                    source_names, param._spec.placements, strict=True
                ):
                    if source_name not in spmd_names:
                        raise ValueError(
                            f"Cannot lift DTensor mesh dim {source_name!r} "
                            f"into SPMD mesh dims {spmd_names}"
                        )
                    placements[spmd_names.index(source_name)] = placement
                local = param.to_local()
            else:
                local = param.detach()
            if local.device.type != "meta" and local.device.type != mesh.device_type:
                local = local.to(_get_device_from_mesh(mesh))

            lifted = DTensor.from_local(
                local,
                mesh,
                placements,
                run_check=False,
                shape=param.shape,
                stride=tuple(param.stride()),
                grad_placements=placements,
            )
            submodule._parameters[name] = nn.Parameter(
                lifted, requires_grad=param.requires_grad
            )


def _validate_placements_for_tracing(
    param_placements: dict[str, tuple[Placement, ...]],
    named_params: list[tuple[str, nn.Parameter]],
    mesh_info: FlexShardMeshInfo,
    mesh: DeviceMesh,
) -> None:
    """Validate that placements are compatible with FX tracing.

    Raises ValueError if any placement uses invalid configuration. The mesh
    passed here is FlexShard's derived 1D DP shard mesh.
    """
    param_dict = dict(named_params)
    world_size = mesh.size()

    for fqn, placements in param_placements.items():
        for placement in placements:
            if isinstance(placement, Owned):
                if placement.owner_rank >= mesh.size():
                    raise ValueError(
                        f"Parameter {fqn!r} uses Owned({placement.owner_rank}) "
                        f"but world_size is {mesh.size()}."
                    )
            if isinstance(placement, RaggedShard):
                if len(placement.local_units) != world_size:
                    raise ValueError(
                        f"Parameter {fqn!r} uses RaggedShard with "
                        f"{len(placement.local_units)} local_units but "
                        f"world_size is {world_size}."
                    )
            if isinstance(placement, Shard):
                param = _get_param_tensor_for_flex_shard(
                    param_dict[fqn], mesh_info, fqn, contiguous=False
                )
                if placement.dim >= param.ndim:
                    raise ValueError(
                        f"Parameter {fqn!r} has {param.ndim} dimensions but "
                        f"Shard(dim={placement.dim}) is out of range."
                    )
            if isinstance(placement, FlatShard):
                param = _get_param_tensor_for_flex_shard(
                    param_dict[fqn], mesh_info, fqn, contiguous=False
                )
                if param.numel() == 0:
                    raise ValueError(
                        f"Parameter {fqn!r} has 0 elements, " "cannot apply FlatShard."
                    )


# ---------------------------------------------------------------------------
# Phase 2b: Bucket assignment and validation
# ---------------------------------------------------------------------------


def _assign_params_to_buckets(
    param_fqns: list[str],
    buckets: list[BucketSpec],
) -> list[list[str]]:
    """Assign each param FQN to exactly one bucket via fnmatch.

    Returns:
        List of lists: assignments[i] = [fqn, ...] for bucket i.

    Raises:
        ValueError: if any param matches zero or multiple buckets.
    """
    param_to_buckets: dict[str, list[int]] = {fqn: [] for fqn in param_fqns}
    for bucket_idx, bucket in enumerate(buckets):
        for fqn in param_fqns:
            for pattern in bucket.patterns:
                if fnmatch.fnmatch(fqn, pattern):
                    param_to_buckets[fqn].append(bucket_idx)
                    break  # one match per bucket is enough

    # Check for orphans
    orphans = [fqn for fqn, idxs in param_to_buckets.items() if len(idxs) == 0]
    if orphans:
        orphan_list = "\n  ".join(orphans)
        raise ValueError(
            f"flex_shard: {len(orphans)} parameters not covered by any bucket:\n"
            f"  {orphan_list}\n"
            'Add these to an existing bucket or add a catch-all bucket: ["*"]'
        )

    # Check for overlaps
    overlaps = {fqn: idxs for fqn, idxs in param_to_buckets.items() if len(idxs) > 1}
    if overlaps:
        lines = []
        for fqn, idxs in overlaps.items():
            bucket_descs = ", ".join(f"bucket {i} {buckets[i].patterns}" for i in idxs)
            lines.append(f"  {fqn} -> {bucket_descs}")
        overlap_list = "\n".join(lines)
        raise ValueError(
            f"flex_shard: {len(overlaps)} parameters matched multiple buckets:\n"
            f"{overlap_list}\n"
            "Ensure each parameter matches exactly one bucket."
        )

    # Build assignments
    assignments: list[list[str]] = [[] for _ in buckets]
    for fqn, idxs in param_to_buckets.items():
        assignments[idxs[0]].append(fqn)

    return assignments


def _validate_bucket_placements(
    bucket_assignments: list[list[str]],
    param_placements: dict[str, tuple[Placement, ...]],
    buckets: list[BucketSpec],
) -> None:
    """Validate that all params in each bucket share the same placement type.

    Shard(0) + Shard(0) is valid. Shard(0) + Shard(1) is not.
    Shard + FlatShard is not. Owned(0) + Owned(1) is not.
    """
    for bucket_idx, fqns in enumerate(bucket_assignments):
        if not fqns:
            continue
        reference_placement = param_placements[fqns[0]][0]
        for fqn in fqns[1:]:
            placement = param_placements[fqn][0]
            if type(placement) is not type(reference_placement):
                raise ValueError(
                    f"Bucket {bucket_idx} "
                    f"{buckets[bucket_idx].patterns} "
                    f"has mixed placement types: {fqns[0]!r} uses "
                    f"{type(reference_placement).__name__} but {fqn!r} uses "
                    f"{type(placement).__name__}. "
                    "All params in a bucket must share the same placement type."
                )
            if isinstance(placement, Shard) and isinstance(reference_placement, Shard):
                if placement.dim != reference_placement.dim:
                    raise ValueError(
                        f"Bucket {bucket_idx} "
                        f"{buckets[bucket_idx].patterns} "
                        f"has mixed shard dimensions: {fqns[0]!r} uses "
                        f"Shard({reference_placement.dim}) but {fqn!r} uses "
                        f"Shard({placement.dim}). "
                        "All Shard params in a bucket must share the same "
                        "dimension."
                    )
            if isinstance(placement, Owned) and isinstance(reference_placement, Owned):
                if placement.owner_rank != reference_placement.owner_rank:
                    raise ValueError(
                        f"Bucket {bucket_idx} "
                        f"{buckets[bucket_idx].patterns} "
                        f"has mixed owner ranks: {fqns[0]!r} uses "
                        f"Owned({reference_placement.owner_rank}) but "
                        f"{fqn!r} uses Owned({placement.owner_rank}). "
                        "All Owned params in a bucket must share the same "
                        "owner_rank."
                    )
            if isinstance(placement, RaggedShard) and isinstance(
                reference_placement, RaggedShard
            ):
                if placement.local_units != reference_placement.local_units:
                    raise ValueError(
                        f"Bucket {bucket_idx} "
                        f"{buckets[bucket_idx].patterns} "
                        f"has mixed local_units: {fqns[0]!r} uses "
                        f"{reference_placement.local_units} but {fqn!r} uses "
                        f"{placement.local_units}. "
                        "All RaggedShard params in a bucket must share the "
                        "same local_units."
                    )


# ---------------------------------------------------------------------------
# Phase 2a: Parametrization guard and property-based registration
# ---------------------------------------------------------------------------

_active_parametrization = True
_reshard_checkpoint_enabled: ContextVar[bool] = ContextVar(
    "_reshard_checkpoint_enabled",
    default=True,
)
_reshard_checkpoint_recompute: ContextVar[bool] = ContextVar(
    "_reshard_checkpoint_recompute",
    default=False,
)


def _set_tensor_reshard_after_forward(
    tensor: torch.Tensor,
    reshard_after_forward: bool,
) -> torch.Tensor:
    """Attach per-bucket reshard policy to tensors produced while tracing."""
    if torch.compiler.is_compiling():
        return tensor
    try:
        setattr(tensor, _RESHARD_AFTER_FORWARD_ATTR, reshard_after_forward)
    except Exception:
        pass
    return tensor


@contextmanager
def disable_active_parametrization() -> Generator[None, None, None]:
    """Disable parametrization forward (returns raw sharded tensor).

    Use during initialization, checkpointing, or any context where
    parameter access should not trigger collective communication.
    """
    global _active_parametrization
    try:
        _active_parametrization = False
        yield
    finally:
        _active_parametrization = True


@contextmanager
def _disable_reshard_checkpoint() -> Generator[None, None, None]:
    """Internal compile frontend hook to skip eager reshard checkpoint wrappers."""
    token = _reshard_checkpoint_enabled.set(False)
    try:
        yield
    finally:
        _reshard_checkpoint_enabled.reset(token)


@contextmanager
def _mark_reshard_checkpoint_recompute(ctx: Any) -> Generator[None, None, None]:
    """Mark execution as FlexShard checkpoint recomputation."""
    token = _reshard_checkpoint_recompute.set(True)
    try:
        with ctx:
            yield
    finally:
        _reshard_checkpoint_recompute.reset(token)


def _is_graph_capture_active() -> bool:
    """Return whether parametrizations should emit traceable collectives."""
    if torch.compiler.is_compiling():
        return True
    try:
        return torch._guards.TracingContext.try_get() is not None
    except AttributeError:
        return False


def _raise_missing_eager_batched_unshard(parametrization: nn.Module) -> None:
    param_fqn = getattr(parametrization, _PARAM_FQN_ATTR, "<unknown>")
    bucket_fqn = getattr(parametrization, _BUCKET_FQN_ATTR, None)
    bucket_msg = f" in bucket {bucket_fqn!r}" if bucket_fqn else ""
    raise RuntimeError(
        "FlexShard eager mode would fall back to per-parameter "
        f"_c10d_functional collectives for parameter {param_fqn!r}{bucket_msg}, "
        "but eager Shard/RaggedShard parameters must be served by a batched "
        "all-gather hook. This usually means the BucketSpec boundary does not "
        "match the module hook/checkpoint execution unit. Split the bucket to "
        "match forward module boundaries, or use torch.compile so functional "
        "collectives are captured by the graph path."
    )


def _get_or_create_eager_comm_context(
    root_module: nn.Module,
    device: torch.device,
) -> EagerAllGatherContext:
    contexts = getattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, None)
    if contexts is None:
        contexts = {}
        setattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, contexts)

    context = contexts.get(device)
    if context is None:
        context = EagerAllGatherContext(
            all_gather_stream=torch.cuda.Stream(device=device, priority=-1),
            reduce_scatter_stream=torch.cuda.Stream(device=device, priority=-1),
        )
        contexts[device] = context
    return context


_wrap_class_counter = 0


def _register_parametrization(
    module: nn.Module,
    param_parametrizations: dict[str, nn.Module],
) -> None:
    """Register per-parameter property getters that call parametrization forward.

    Uses dynamic subclass creation (not nn.utils.parametrize) to avoid
    state_dict key mangling. state_dict reads self._parameters directly,
    bypassing property getters.

    Args:
        module: The leaf module owning the parameters.
        param_parametrizations: Maps parameter name to its parametrization module.
    """
    global _wrap_class_counter
    _wrap_class_counter += 1

    def _make_getter(pn, p):
        def getter(self):
            # In eager batched mode, _pre_gathered is set on the
            # parametrization by the batched all-gather pre_forward hook.
            pre = getattr(p, "_pre_gathered", None)
            if pre is not None:
                p._pre_gathered = None
                param_dtype = getattr(p, "param_dtype", None)
                reduce_dtype = getattr(p, "reduce_dtype", None)
                if getattr(p, _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR, False):
                    if param_dtype is not None or reduce_dtype is not None:
                        pre = _MixedPrecisionCast.apply(pre, param_dtype, reduce_dtype)
                    return pre

                if param_dtype is not None and pre.dtype != param_dtype:
                    pre = pre.to(param_dtype)
                unsharded = pre.detach().requires_grad_(True)
                if (
                    torch.is_grad_enabled()
                    and getattr(p, "_unsharded_for_reduce", None) is None
                ):
                    p._unsharded_for_reduce = unsharded
                return unsharded
            if (
                getattr(p, _REQUIRES_EAGER_BATCHED_UNSHARD_ATTR, False)
                and not getattr(p, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, False)
                and not _is_graph_capture_active()
            ):
                _raise_missing_eager_batched_unshard(p)
            return p(self._parameters[pn])

        return getter

    param_name_to_property = {
        param_name: property(_make_getter(param_name, param))
        for param_name, param in param_parametrizations.items()
    }
    module_cls = type(
        f"FlexShard{module.__class__.__name__}_{_wrap_class_counter}",
        (module.__class__,),
        param_name_to_property,
    )
    module.__class__ = module_cls
    sys.modules[module_cls.__module__].__dict__[module_cls.__name__] = module_cls


def set_sharding_info(
    tensor: torch.Tensor,
    placements: tuple[Placement, ...],
    global_shape: torch.Size,
    global_stride: tuple[int, ...],
    mesh: DeviceMesh,
) -> None:
    """Annotate a tensor with FlexShard placement metadata."""
    tensor._placements = placements
    tensor._global_shape = global_shape
    tensor._global_stride = global_stride
    tensor._mesh = mesh


def get_placements(tensor: torch.Tensor) -> tuple[Placement, ...] | None:
    """Get FlexShard placements from a tensor, or None if not annotated."""
    return getattr(tensor, _PLACEMENTS_ATTR, None)


def get_global_shape(tensor: torch.Tensor) -> torch.Size | None:
    """Get the global (unsharded) shape from a tensor, or None if not annotated."""
    return getattr(tensor, _GLOBAL_SHAPE_ATTR, None)


def is_flex_shard_param(tensor: torch.Tensor) -> bool:
    """Check if a tensor has FlexShard placement metadata."""
    return hasattr(tensor, _PLACEMENTS_ATTR)


class FlexShardModule:
    """Mixin added to modules after flex_shard(). Provides sharding methods."""

    def unshard(self) -> None:
        for storage in getattr(self, _DSTORAGES_ATTR):
            storage.unshard()

    def reshard(self) -> None:
        raise NotImplementedError

    def reduce_grad(self) -> None:
        for storage in getattr(self, _DSTORAGES_ATTR):
            storage.reduce_grad()

    @property
    def dstorage(self) -> DStorage:
        """First (or only) DStorage. For multi-bucket, use .dstorages."""
        return getattr(self, _DSTORAGE_ATTR)

    @property
    def dstorages(self) -> list:
        """All DStorage instances (one per bucket)."""
        return getattr(self, _DSTORAGES_ATTR)

    @property
    def eager_comm_contexts(self) -> dict[torch.device, EagerAllGatherContext]:
        """Root-owned eager communication contexts keyed by CUDA device."""
        return getattr(self, _EAGER_COMM_CONTEXTS_ATTR)


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """Mixed precision policy for FlexShard buckets.

    Args:
        param_dtype: Dtype for forward compute. Parameters are all-gathered
            in storage dtype, then cast to param_dtype. If None, no cast.
        reduce_dtype: Dtype for gradient reduction. Gradients are cast to
            this dtype before reduce-scatter. If None, uses param_dtype
            (or storage dtype if param_dtype is also None).
    """

    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class OffloadPolicy:
    """CPU offload policy for FlexShard buckets.

    When set on a BucketSpec, the bucket's byte storage is allocated on
    CPU (optionally pinned). The parametrization handles H2D transfer
    before all-gather; backward autograd handles D2H automatically.

    Args:
        pin_memory: Whether to pin CPU memory for faster H2D/D2H
            transfers via DMA. Set to False if insufficient CPU memory.
            Default True.
    """

    pin_memory: bool = True


@dataclass(frozen=True)
class BucketSpec:
    """Specification for a parameter communication bucket.

    Args:
        patterns: fnmatch glob patterns matched against parameter FQNs.
            A parameter matches this bucket if its FQN matches any pattern.
        mp_policy: Mixed precision policy for this bucket.
        offload_policy: CPU offload policy for this bucket.
        reshard_after_forward: Whether to free this bucket's unsharded
            parameters after forward and recompute them in backward.
    """

    patterns: list[str]
    mp_policy: MixedPrecisionPolicy | None = None
    offload_policy: OffloadPolicy | None = None
    reshard_after_forward: bool = True


class Owned(Placement):
    """
    Placement indicating a parameter is fully owned by one rank.

    In parameter-boundary sharding, each parameter is assigned to exactly one
    rank (the owner). The owner has the full parameter data, while other ranks
    have an empty tensor.

    This enables sharding at parameter boundaries rather than within parameters,
    which can be useful for models where parameter sizes don't divide evenly.
    """

    def __init__(self, owner_rank: int):
        self.owner_rank = owner_rank

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Owned):
            return False
        return self.owner_rank == other.owner_rank

    def __hash__(self) -> int:
        return hash((type(self), self.owner_rank))

    def __repr__(self) -> str:
        return f"Owned({self.owner_rank})"

    def compute_local_shape(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> torch.Size:
        if rank == self.owner_rank:
            return global_shape
        return torch.Size([0] * len(global_shape))

    def compute_local_numel(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> int:
        if rank == self.owner_rank:
            numel = 1
            for d in global_shape:
                numel *= d
            return numel
        return 0

    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        if rank == self.owner_rank:
            return param.contiguous()
        return param.new_empty(0)

    def assemble_from_shards(
        self,
        per_rank_shards: list[torch.Tensor],
        global_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return per_rank_shards[self.owner_rank].view(global_shape)

    @classmethod
    def unshard(
        cls,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        rank = mesh.get_local_rank()
        pg = mesh.get_group()

        results: list[torch.Tensor] = []
        for tensor, info in zip(tensors, infos, strict=True):
            p = info.placements[0]
            # Broadcast from owner
            if rank == p.owner_rank:
                full = tensor.contiguous()
            else:
                full = torch.empty(
                    info.global_shape, dtype=info.dtype, device=tensor.device
                )
            dist.broadcast(full, src=p.owner_rank, group=pg)
            results.append(full)
        return results

    @classmethod
    def reduce_grad(
        cls,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        rank = mesh.get_local_rank()
        pg = mesh.get_group()

        results: list[torch.Tensor] = []
        for tensor, info in zip(tensors, infos, strict=True):
            p = info.placements[0]
            send = tensor.contiguous()
            dist.reduce(send, dst=p.owner_rank, op=dist.ReduceOp.AVG, group=pg)
            if rank == p.owner_rank:
                results.append(send)
            else:
                results.append(tensor.new_empty(0))
        return results


class _OwnedFullCopy(Owned):
    """Internal variant of Owned used in parametrization mode.

    All ranks store a full copy of the parameter (not just the owner).
    This enables FX-traceable broadcast from owner_rank via
    OwnedParametrization. The memory overhead is acceptable since Owned
    is typically used for small params (biases, norms).
    """

    def create_parametrization(self, info, group_name, world_size, **kwargs):
        return OwnedParametrization(
            owner_rank=self.owner_rank,
            group_name=group_name,
            world_size=world_size,
            **kwargs,
        )

    def compute_local_shape(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> torch.Size:
        return global_shape

    def compute_local_numel(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> int:
        numel = 1
        for d in global_shape:
            numel *= d
        return numel

    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        return param.contiguous()


class RaggedShard(Placement):
    """Asymmetric sharding — variable chunk sizes per rank.

    All ranks hold data, but sizes are determined by local_units ratios.
    Unlike Shard (uniform chunks) or Owned (full param on one rank),
    RaggedShard distributes every param across all ranks with variable sizes.

    Args:
        dims: Dimensions to shard along. Currently only single-dim supported.
        local_units: Relative allocation per rank. Length must equal world_size.
            E.g., (1, 2, 1, 1) means rank 1 gets 2/5 of the dimension.
    """

    def __init__(self, dims: tuple[int, ...] = (0,), local_units: tuple[int, ...] = ()):
        if len(dims) != 1:
            raise NotImplementedError("Only single-dim RaggedShard supported")
        self.dims = dims
        self.local_units = local_units
        self.dim = dims[0]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RaggedShard):
            return False
        return self.dims == other.dims and self.local_units == other.local_units

    def __hash__(self) -> int:
        return hash((type(self), self.dims, self.local_units))

    def __repr__(self) -> str:
        return f"RaggedShard(dims={self.dims}, local_units={self.local_units})"

    def create_parametrization(self, info, group_name, world_size, **kwargs):
        split_sizes = self._compute_dim_splits(info.global_shape[self.dim])
        return RaggedShardParametrization(
            shard_dim=self.dim,
            split_sizes=split_sizes,
            group_name=group_name,
            world_size=world_size,
            **kwargs,
        )

    def _compute_dim_splits(self, dim_size: int) -> list[int]:
        """Compute per-rank sizes along the sharded dimension.

        Distributes dim_size proportionally by local_units, with remainder
        distributed to first ranks.
        """
        total_units = sum(self.local_units)
        ws = len(self.local_units)
        splits = []
        remaining = dim_size
        remaining_units = total_units
        for r in range(ws):
            if remaining_units == 0:
                splits.append(0)
            else:
                # Proportional allocation with rounding
                chunk = (
                    remaining * self.local_units[r] + remaining_units - 1
                ) // remaining_units
                chunk = min(chunk, remaining)
                splits.append(chunk)
                remaining -= chunk
                remaining_units -= self.local_units[r]
        return splits

    def compute_local_shape(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> torch.Size:
        dim_size = global_shape[self.dim]
        splits = self._compute_dim_splits(dim_size)
        shape = list(global_shape)
        shape[self.dim] = splits[rank]
        return torch.Size(shape)

    def compute_local_numel(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> int:
        local_shape = self.compute_local_shape(global_shape, rank, world_size)
        numel = 1
        for d in local_shape:
            numel *= d
        return numel

    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        dim_size = param.shape[self.dim]
        splits = self._compute_dim_splits(dim_size)
        start = sum(splits[:rank])
        return param.narrow(self.dim, start, splits[rank]).contiguous()

    def assemble_from_shards(
        self,
        per_rank_shards: list[torch.Tensor],
        global_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        non_empty = [s for s in per_rank_shards if s.numel() > 0]
        if non_empty:
            return torch.cat(non_empty, dim=self.dim)
        return torch.empty(global_shape, dtype=dtype, device=per_rank_shards[0].device)

    @classmethod
    def unshard(
        cls,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        ws = mesh.size()
        pg = mesh.get_group()

        results: list[torch.Tensor] = []
        for tensor, info in zip(tensors, infos, strict=True):
            p = info.placements[0]
            # Variable-size all_gather
            per_rank_shards = [
                torch.empty(
                    p.compute_local_shape(info.global_shape, r, ws),
                    dtype=info.dtype,
                    device=tensor.device,
                )
                for r in range(ws)
            ]
            dist.all_gather(per_rank_shards, tensor.contiguous(), group=pg)
            results.append(
                p.assemble_from_shards(per_rank_shards, info.global_shape, info.dtype)
            )
        return results

    @classmethod
    def reduce_grad(
        cls,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        ws = mesh.size()
        rank = mesh.get_local_rank()
        pg = mesh.get_group()

        results: list[torch.Tensor] = []
        for tensor, info in zip(tensors, infos, strict=True):
            p = info.placements[0]
            dim_size = tensor.shape[p.dim]
            splits = p._compute_dim_splits(dim_size)
            input_list = list(torch.split(tensor, splits, dim=p.dim))
            flat_inputs = [chunk.contiguous().view(-1) for chunk in input_list]
            local_shape = p.compute_local_shape(info.global_shape, rank, ws)
            flat_output = torch.empty(
                info.local_numel, dtype=tensor.dtype, device=tensor.device
            )
            dist.reduce_scatter(
                flat_output, flat_inputs, op=dist.ReduceOp.AVG, group=pg
            )
            results.append(flat_output.view(local_shape))
        return results


class ShardParametrization(nn.Module):
    """FX-traceable parametrization for Shard placement.

    Reconstructs the full parameter from a local shard using _c10d_functional
    ops that are visible to make_fx and torch.compile. The backward pass
    (reduce_scatter) is autograd-generated.

    For dim != 0, all_gather_into_tensor concatenates along dim 0, so we
    chunk and re-cat along the correct shard_dim.

    For uneven splits (dim_size % world_size != 0), each rank pads its local
    shard to ``padded_shard_size`` before all-gather, then slices the result
    back to ``global_dim_size`` along ``shard_dim``.
    """

    def __init__(
        self,
        shard_dim: int,
        group_name: str,
        world_size: int,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = None,
        compute_device: torch.device | None = None,
        padded_shard_size: int | None = None,
        global_dim_size: int | None = None,
        reshard_after_forward: bool = True,
    ):
        super().__init__()
        self.shard_dim = shard_dim
        self.group_name = group_name
        self.world_size = world_size
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.compute_device = compute_device
        self.padded_shard_size = padded_shard_size
        self.global_dim_size = global_dim_size
        self.reshard_after_forward = reshard_after_forward

    def forward(self, local_shard: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return local_shard
        if (
            self.compute_device is not None
            and local_shard.device != self.compute_device
        ):
            local_shard = local_shard.to(self.compute_device, non_blocking=True)

        # Pad local shard to uniform size if uneven
        if self.padded_shard_size is not None:
            local_size = local_shard.shape[self.shard_dim]
            pad_size = self.padded_shard_size - local_size
            if pad_size > 0:
                pad_shape = list(local_shard.shape)
                pad_shape[self.shard_dim] = pad_size
                padding = local_shard.new_zeros(pad_shape)
                local_shard = torch.cat([local_shard, padding], dim=self.shard_dim)

        full = torch.ops._c10d_functional.all_gather_into_tensor(
            local_shard, self.world_size, self.group_name
        )
        full = torch.ops._c10d_functional.wait_tensor(full)
        full = _set_tensor_reshard_after_forward(full, self.reshard_after_forward)
        if full.requires_grad and torch.is_grad_enabled():
            full.register_hook(lambda grad: grad / self.world_size)
        if self.shard_dim != 0:
            chunks = full.chunk(self.world_size, dim=0)
            full = torch.cat(chunks, dim=self.shard_dim)
            full = _set_tensor_reshard_after_forward(full, self.reshard_after_forward)

        # Slice out padding if uneven
        if self.global_dim_size is not None:
            full = full.narrow(self.shard_dim, 0, self.global_dim_size)
            full = _set_tensor_reshard_after_forward(full, self.reshard_after_forward)

        if self.param_dtype is not None or self.reduce_dtype is not None:
            full = _MixedPrecisionCast.apply(full, self.param_dtype, self.reduce_dtype)
            full = _set_tensor_reshard_after_forward(full, self.reshard_after_forward)
        return full


class FlatShardParametrization(nn.Module):
    """FX-traceable parametrization for FlatShard placement.

    Reconstructs the full parameter from a flat 1D shard using
    _c10d_functional ops. The flat shard has shape (numel // world_size,).
    After all-gather, the result is reshaped to the original parameter shape.

    flex_shard() decomposes bucket-level FlatShard into per-parameter flat
    sharding in parametrization mode: each param gets FlatShard(0, numel, numel)
    so every rank holds exactly numel // world_size elements.

    For uneven splits (numel % world_size != 0), each rank pads its flat shard
    to ``padded_shard_size`` before all-gather, then slices the result back to
    ``global_numel`` before reshaping.
    """

    def __init__(
        self,
        group_name: str,
        world_size: int,
        original_shape: torch.Size,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = None,
        compute_device: torch.device | None = None,
        padded_shard_size: int | None = None,
        global_numel: int | None = None,
        reshard_after_forward: bool = True,
    ):
        super().__init__()
        self.group_name = group_name
        self.world_size = world_size
        self.original_shape = original_shape
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.compute_device = compute_device
        self.padded_shard_size = padded_shard_size
        self.global_numel = global_numel
        self.reshard_after_forward = reshard_after_forward

    def forward(self, flat_shard: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return flat_shard
        if self.compute_device is not None and flat_shard.device != self.compute_device:
            flat_shard = flat_shard.to(self.compute_device, non_blocking=True)

        # Pad flat shard to uniform size if uneven
        if self.padded_shard_size is not None:
            pad_size = self.padded_shard_size - flat_shard.numel()
            if pad_size > 0:
                padding = flat_shard.new_zeros(pad_size)
                flat_shard = torch.cat([flat_shard, padding])

        full_flat = torch.ops._c10d_functional.all_gather_into_tensor(
            flat_shard, self.world_size, self.group_name
        )
        full_flat = torch.ops._c10d_functional.wait_tensor(full_flat)
        full_flat = _set_tensor_reshard_after_forward(
            full_flat, self.reshard_after_forward
        )
        if full_flat.requires_grad and torch.is_grad_enabled():
            full_flat.register_hook(lambda grad: grad / self.world_size)

        # Slice off padding if uneven
        if self.global_numel is not None:
            full_flat = full_flat[: self.global_numel]
            full_flat = _set_tensor_reshard_after_forward(
                full_flat, self.reshard_after_forward
            )

        full = full_flat.view(self.original_shape)
        full = _set_tensor_reshard_after_forward(full, self.reshard_after_forward)

        if self.param_dtype is not None or self.reduce_dtype is not None:
            full = _MixedPrecisionCast.apply(full, self.param_dtype, self.reduce_dtype)
            full = _set_tensor_reshard_after_forward(full, self.reshard_after_forward)
        return full


class _MixedPrecisionCast(torch.autograd.Function):
    """Cast with decoupled forward/backward dtype control.

    Forward casts to param_dtype (compute dtype).
    Backward casts to reduce_dtype (gradient reduction dtype).
    This allows all-gather in storage dtype (e.g., fp32), compute in
    param_dtype (e.g., bf16), and reduce-scatter in reduce_dtype (e.g., fp32).
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        param_dtype: torch.dtype | None,
        reduce_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        ctx.reduce_dtype = reduce_dtype
        if param_dtype is not None and x.dtype != param_dtype:
            return x.to(param_dtype)
        return x

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        if ctx.reduce_dtype is not None and grad.dtype != ctx.reduce_dtype:
            return grad.to(ctx.reduce_dtype), None, None
        return grad, None, None


class _OwnedBroadcast(torch.autograd.Function):
    """Differentiable broadcast for Owned placement.

    Forward: broadcast from owner_rank to all ranks.
    Backward: all_reduce(sum) / world_size (averaged gradient to all ranks).

    _c10d_functional.broadcast has no backward registered, so this custom
    autograd function provides the gradient flow.
    """

    @staticmethod
    def forward(
        ctx: Any,
        param: torch.Tensor,
        owner_rank: int,
        group_name: str,
        world_size: int,
    ) -> torch.Tensor:
        ctx.group_name = group_name
        ctx.world_size = world_size
        result = torch.ops._c10d_functional.broadcast(param, owner_rank, group_name)
        return torch.ops._c10d_functional.wait_tensor(result)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        reduced = torch.ops._c10d_functional.all_reduce(grad, "sum", ctx.group_name)
        reduced = torch.ops._c10d_functional.wait_tensor(reduced)
        return reduced / ctx.world_size, None, None, None


class OwnedParametrization(nn.Module):
    """FX-traceable parametrization for Owned placement.

    All ranks store a full copy of the parameter. The broadcast from
    owner_rank ensures consistency, and the backward all-reduce averages
    gradients across all ranks. Since all ranks hold the same param values
    and receive the same averaged gradient, the optimizer produces identical
    updates, keeping all copies in sync without explicit synchronization.
    """

    def __init__(
        self,
        owner_rank: int,
        group_name: str,
        world_size: int,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = None,
        compute_device: torch.device | None = None,
        reshard_after_forward: bool = True,
    ):
        super().__init__()
        self.owner_rank = owner_rank
        self.group_name = group_name
        self.world_size = world_size
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.compute_device = compute_device
        self.reshard_after_forward = reshard_after_forward

    def forward(self, param: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return param
        if self.compute_device is not None and param.device != self.compute_device:
            param = param.to(self.compute_device, non_blocking=True)
        # Owned always uses per-param broadcast (can't batch across
        # different owner_ranks). No _pre_gathered support needed.
        full = _OwnedBroadcast.apply(
            param, self.owner_rank, self.group_name, self.world_size
        )
        full = _set_tensor_reshard_after_forward(full, self.reshard_after_forward)
        if self.param_dtype is not None or self.reduce_dtype is not None:
            full = _MixedPrecisionCast.apply(full, self.param_dtype, self.reduce_dtype)
            full = _set_tensor_reshard_after_forward(full, self.reshard_after_forward)
        return full


class RaggedShardParametrization(nn.Module):
    """FX-traceable parametrization for RaggedShard placement.

    Uses pad-to-uniform: each rank pads its local shard to the maximum shard
    size across all ranks, then uniform all_gather_into_tensor, then narrow
    each chunk to its real size and cat to reconstruct the full parameter.
    The backward (reduce-scatter + slice) is autograd-generated.
    """

    def __init__(
        self,
        shard_dim: int,
        split_sizes: list[int],
        group_name: str,
        world_size: int,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = None,
        compute_device: torch.device | None = None,
        reshard_after_forward: bool = True,
    ):
        super().__init__()
        self.shard_dim = shard_dim
        self.split_sizes = split_sizes
        self.max_shard_size = max(split_sizes)
        self.group_name = group_name
        self.world_size = world_size
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.compute_device = compute_device
        self.reshard_after_forward = reshard_after_forward

    def forward(self, local_shard: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return local_shard
        if (
            self.compute_device is not None
            and local_shard.device != self.compute_device
        ):
            local_shard = local_shard.to(self.compute_device, non_blocking=True)

        # Pad to max shard size along shard_dim
        local_size = local_shard.shape[self.shard_dim]
        pad_size = self.max_shard_size - local_size
        if pad_size > 0:
            pad_shape = list(local_shard.shape)
            pad_shape[self.shard_dim] = pad_size
            padding = local_shard.new_zeros(pad_shape)
            local_shard = torch.cat([local_shard, padding], dim=self.shard_dim)

        # Uniform all-gather (all ranks now have max_shard_size)
        full = torch.ops._c10d_functional.all_gather_into_tensor(
            local_shard, self.world_size, self.group_name
        )
        full = torch.ops._c10d_functional.wait_tensor(full)
        full = _set_tensor_reshard_after_forward(full, self.reshard_after_forward)
        if full.requires_grad and torch.is_grad_enabled():
            full.register_hook(lambda grad: grad / self.world_size)

        # Reassemble: chunk along dim 0, narrow each to its real size, cat
        chunks = full.chunk(self.world_size, dim=0)
        real_chunks = [
            chunk.narrow(self.shard_dim, 0, self.split_sizes[r])
            for r, chunk in enumerate(chunks)
        ]
        full = torch.cat(real_chunks, dim=self.shard_dim)
        full = _set_tensor_reshard_after_forward(full, self.reshard_after_forward)

        if self.param_dtype is not None or self.reduce_dtype is not None:
            full = _MixedPrecisionCast.apply(full, self.param_dtype, self.reduce_dtype)
            full = _set_tensor_reshard_after_forward(full, self.reshard_after_forward)
        return full


class DTensorAwareParametrization(nn.Module):
    """Wrapper that handles nested DTensor inputs from TP/EP composition.

    FlexShard stores raw sharded parameters as plain local tensors. If the
    original parameter was a DTensor (e.g., from TP/EP composition), this wrapper
    re-wraps the unsharded FlexShard result with the original non-DP DTensor
    mesh and placements.

    It also handles the direct DTensor-input case for compatibility.
    """

    def __init__(
        self,
        inner: nn.Module,
        mesh: DeviceMesh | None = None,
        placements: tuple[Any, ...] | None = None,
        dp_dim_indices: tuple[int, ...] = (),
    ):
        super().__init__()
        self.inner = inner
        self.mesh = mesh
        self.placements = placements
        self.dp_dim_indices = dp_dim_indices
        # The eager batched all-gather fast path reads these attributes from
        # the outer parametrization when returning a pre-gathered tensor.
        self.param_dtype = getattr(inner, "param_dtype", None)
        self.reduce_dtype = getattr(inner, "reduce_dtype", None)
        self.reshard_after_forward = getattr(inner, "reshard_after_forward", True)

    def _requires_dtensor_output(self) -> bool:
        if self.placements is None:
            return False

        dp_dims = set(self.dp_dim_indices)
        return any(dim not in dp_dims for dim in range(len(self.placements)))

    def _compute_dtensor_layout(self):
        if (
            self.mesh is None
            or self.placements is None
            or not self._requires_dtensor_output()
        ):
            return None, None

        if self.mesh.mesh_dim_names is None:
            return self.mesh, self.placements

        from torch.distributed.tensor import Replicate

        dp_dims = set(self.dp_dim_indices)
        for dim in dp_dims:
            if dim < len(self.placements) and not isinstance(
                self.placements[dim], Replicate
            ):
                return self.mesh, self.placements

        non_dp_indices = tuple(
            dim for dim in range(len(self.placements)) if dim not in dp_dims
        )
        if not non_dp_indices:
            return None, None
        if len(non_dp_indices) == len(self.placements):
            return self.mesh, self.placements

        mesh_dim_names = self.mesh.mesh_dim_names
        names = tuple(mesh_dim_names[dim] for dim in non_dp_indices)
        placements = tuple(self.placements[dim] for dim in non_dp_indices)
        mesh = self.mesh[names[0]] if len(names) == 1 else self.mesh[names]
        return mesh, placements

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return x

        from torch.distributed.tensor import DTensor

        if isinstance(x, DTensor):
            inner_mesh = x._spec.mesh
            inner_placements = tuple(x._spec.placements)
            local = x.to_local(grad_placements=inner_placements)
            result = self.inner(local)
            return DTensor.from_local(
                result,
                inner_mesh,
                inner_placements,
                run_check=False,
                grad_placements=inner_placements,
            )
        compute_mesh, compute_placements = self._compute_dtensor_layout()
        if compute_mesh is not None and compute_placements is not None:
            result = self.inner(x)
            return DTensor.from_local(
                result,
                compute_mesh,
                compute_placements,
                run_check=False,
                grad_placements=compute_placements,
            )
        return self.inner(x)


# ---------------------------------------------------------------------------
# Eager reshard-after-forward via checkpoint + selective AC policy
# ---------------------------------------------------------------------------

# Collective ops that should be recomputed (not saved) by checkpoint.
class _RegisterPostBackward(torch.autograd.Function):
    """Identity on forward inputs; fires a callback in backward.

    Inserted on a layer's forward inputs so that the callback fires AFTER
    all the layer's parameter gradients are computed. Used for batched
    reduce-scatter per bucket.
    """

    @staticmethod
    def forward(ctx, reduce_fn, *inputs):
        ctx.reduce_fn = reduce_fn
        return inputs

    @staticmethod
    def backward(ctx, *grads):
        ctx.reduce_fn()
        return (None,) + grads


# These produce the unsharded param tensors that we want freed per-layer.
_FLEX_SHARD_COLLECTIVE_OPS = {
    torch.ops._c10d_functional.all_gather_into_tensor.default,
    torch.ops._c10d_functional.wait_tensor.default,
    torch.ops._c10d_functional.broadcast.default,
}


def _flex_shard_reshard_policy(ctx, func, *args, **kwargs):
    """Checkpoint policy for per-layer reshard-after-forward.

    Marks collective ops (all-gather, broadcast, wait_tensor) for
    recomputation — checkpoint discards their outputs after each layer's
    forward. All other ops (matmul, attention, etc.) are saved, avoiding
    redundant compute recomputation in backward.
    """
    from torch.utils.checkpoint import CheckpointPolicy

    if func in _FLEX_SHARD_COLLECTIVE_OPS:
        return CheckpointPolicy.MUST_RECOMPUTE
    # PREFER_RECOMPUTE lets checkpoint decide what to save vs recompute
    # for non-collective ops, matching standard AC behavior.
    return CheckpointPolicy.PREFER_RECOMPUTE


def _compose_reshard_with_ac_policy(ac_context_fn):
    """Compose FlexShard reshard policy with an existing AC context_fn.

    Returns a new context_fn that wraps the AC policy: FlexShard collective
    ops are forced to MUST_RECOMPUTE, everything else delegates to the
    original AC policy. The two op sets are disjoint so no conflicts arise.
    """

    def merged_context_fn():
        from torch.utils.checkpoint import CheckpointPolicy

        contexts = ac_context_fn()
        for ctx in contexts:
            original_policy = getattr(ctx, "policy_fn", None)
            if original_policy is None:
                continue

            def merged_policy(sctx, func, *args, _orig=original_policy, **kwargs):
                if func in _FLEX_SHARD_COLLECTIVE_OPS:
                    return CheckpointPolicy.MUST_RECOMPUTE
                return _orig(sctx, func, *args, **kwargs)

            ctx.policy_fn = merged_policy
        forward_ctx, recompute_ctx = contexts
        return forward_ctx, _mark_reshard_checkpoint_recompute(recompute_ctx)

    return merged_context_fn


class ShardedState(Enum):
    """State of the parameters in DStorage."""

    SHARDED = auto()  # Parameters are sharded tensors with placement metadata
    UNSHARDED = auto()  # Parameters are unsharded for forward/backward


@dataclass
class ParamInfo:
    """Metadata for a parameter in chunked storage."""

    fqn: str
    global_shape: torch.Size
    global_stride: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool
    placements: tuple[Placement, ...]
    local_shape: torch.Size = field(default_factory=lambda: torch.Size([]))
    local_numel: int = 0
    byte_offset: int = 0  # byte offset into the sharded storage
    global_numel: int = 0  # total elements in unsharded param
    unsharded_byte_offset: int = 0  # byte offset into the unsharded storage
    spmd_mesh: DeviceMesh | None = None
    spmd_placements: tuple[Any, ...] | None = None
    spmd_dp_dim_indices: tuple[int, ...] = ()


def _get_dtype_alignment(dtype: torch.dtype) -> int:
    """Get alignment requirement in bytes for a dtype."""
    # Most dtypes need alignment equal to their element size
    return dtype.itemsize


def _align_offset(offset: int, alignment: int) -> int:
    """Round up offset to next aligned boundary."""
    return (offset + alignment - 1) // alignment * alignment


class DStorage:
    """
    Manages a unified byte buffer that backs multiple sharded parameters.

    All parameters, regardless of dtype, are stored in a single contiguous
    byte buffer. Each parameter's local shard is a typed view into this buffer
    at the appropriate byte offset with proper alignment.

    Communication is delegated to Placement.unshard() and
    Placement.reduce_grad(), which handle batching per placement type. The
    default FlexShard execution path uses property-based parametrization; this
    storage object owns buffer layout and metadata.
    """

    def __init__(
        self,
        byte_storage: torch.Tensor,
        param_infos: dict[str, ParamInfo],
        mesh: DeviceMesh,
        total_bytes: int,
        total_unsharded_bytes: int,
        module: nn.Module,
        reshard_after_forward: bool = True,
    ) -> None:
        if byte_storage.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 storage, got {byte_storage.dtype}")
        self._byte_storage = byte_storage
        self._param_infos = param_infos
        self._mesh = mesh
        self._total_bytes = total_bytes
        self._total_unsharded_bytes = total_unsharded_bytes
        self._module = module
        self._state = ShardedState.SHARDED
        self._reshard_after_forward = reshard_after_forward

        # Unsharded buffer (allocated on demand)
        self._unsharded_byte_storage: torch.Tensor | None = None

        # Cache sharded parameters for reduce_grad
        self._sharded_params: dict[str, nn.Parameter] = {}
        for fqn in param_infos:
            parts = fqn.split(".")
            mod = module
            for part in parts[:-1]:
                mod = getattr(mod, part)
            self._sharded_params[fqn] = getattr(mod, parts[-1])

    @property
    def byte_storage(self) -> torch.Tensor:
        """The underlying unified byte storage tensor (sharded)."""
        return self._byte_storage

    @property
    def flat_storage(self) -> torch.Tensor:
        """Alias for byte_storage for backwards compatibility."""
        return self._byte_storage

    @property
    def total_bytes(self) -> int:
        """Total bytes in the sharded storage."""
        return self._total_bytes

    @property
    def total_unsharded_bytes(self) -> int:
        """Total bytes needed for unsharded storage."""
        return self._total_unsharded_bytes

    @property
    def numel(self) -> int:
        """Total number of bytes (for compatibility, returns byte count)."""
        return self._byte_storage.numel()

    @property
    def param_infos(self) -> dict[str, ParamInfo]:
        """Metadata for each parameter."""
        return self._param_infos

    @property
    def state(self) -> ShardedState:
        """Current state (SHARDED or UNSHARDED)."""
        return self._state

    @property
    def world_size(self) -> int:
        """World size of the mesh."""
        return self._mesh.size()

    def get_local_view(self, fqn: str) -> torch.Tensor:
        """Get the local tensor view for a parameter by FQN (from sharded storage)."""
        info = self._param_infos[fqn]
        num_bytes = info.local_numel * info.dtype.itemsize
        byte_view = self._byte_storage[info.byte_offset : info.byte_offset + num_bytes]
        typed_flat = byte_view.view(info.dtype)
        return typed_flat.view(info.local_shape)

    def get_unsharded_view(self, fqn: str) -> torch.Tensor:
        """Get the unsharded tensor view for a parameter by FQN."""
        if self._unsharded_byte_storage is None:
            raise RuntimeError("Unsharded storage not allocated. Call unshard() first.")
        info = self._param_infos[fqn]
        num_bytes = info.global_numel * info.dtype.itemsize
        byte_view = self._unsharded_byte_storage[
            info.unsharded_byte_offset : info.unsharded_byte_offset + num_bytes
        ]
        typed_flat = byte_view.view(info.dtype)
        return typed_flat.view(info.global_shape)

    def unshard(self) -> None:
        """
        All-gather local shards and register unsharded parameters on the module.

        After calling this, model.parameters() returns unsharded tensors for forward/backward.
        """
        if self._state == ShardedState.UNSHARDED:
            return  # Already unsharded

        # Allocate unsharded buffer if needed
        if self._unsharded_byte_storage is None:
            self._unsharded_byte_storage = torch.empty(
                self._total_unsharded_bytes,
                dtype=torch.uint8,
                device=self._byte_storage.device,
            )

        # Gather via Placement.unshard()
        infos = list(self._param_infos.values())
        ptype = type(infos[0].placements[0])
        local_shards = [self._sharded_params[info.fqn].data for info in infos]
        full_params = ptype.unshard(local_shards, infos, self._mesh)

        for info, full_param in zip(infos, full_params, strict=True):
            num_bytes = info.global_numel * info.dtype.itemsize
            dest = self._unsharded_byte_storage[
                info.unsharded_byte_offset : info.unsharded_byte_offset + num_bytes
            ]
            dest.copy_(full_param.reshape(-1).view(torch.uint8))

        # Register unsharded parameters on the module
        for fqn, info in self._param_infos.items():
            unsharded_view = self.get_unsharded_view(fqn)
            unsharded_param = nn.Parameter(
                unsharded_view, requires_grad=info.requires_grad
            )
            _set_param_on_module(self._module, fqn, unsharded_param)

        self._state = ShardedState.UNSHARDED

    def _sync_unsharded_to_storage(self) -> None:
        """
        Copy data from unsharded buffer back to sharded byte_storage.

        This is useful after calling reset_parameters() on unsharded params,
        to ensure the initialized values are persisted in the sharded storage.
        Must be called while in UNSHARDED state, before reduce_grad().
        """
        if self._state != ShardedState.UNSHARDED:
            raise RuntimeError("Must be in UNSHARDED state to sync to storage")
        if self._unsharded_byte_storage is None:
            raise RuntimeError("Unsharded storage not allocated")

        my_rank = self._mesh.get_local_rank()
        world_size = self.world_size

        for fqn, info in self._param_infos.items():
            if info.local_numel == 0:
                continue
            unsharded_view = self.get_unsharded_view(fqn)
            shard = info.placements[0].extract_local_shard(
                unsharded_view, my_rank, world_size
            )
            if shard.numel() > 0:
                nbytes = shard.numel() * shard.element_size()
                self._byte_storage[info.byte_offset : info.byte_offset + nbytes].copy_(
                    shard.reshape(-1).view(torch.uint8)
                )

    def _sync_sharded_to_storage(self, device: torch.device | None = None) -> None:
        """
        Copy data from sharded parameter tensors to byte_storage.

        This is useful after calling to_empty() and reset_parameters() on a model
        that was sharded on meta device. The parameter tensors have been
        materialized and initialized, but byte_storage may still be on meta or
        have stale data.

        Args:
            device: Target device for byte_storage. If None, uses the device of
                    the first sharded parameter.

        Must be called while in SHARDED state.
        """
        if self._state != ShardedState.SHARDED:
            raise RuntimeError("Must be in SHARDED state to sync to storage")

        # Get target device from first param if not specified
        if device is None:
            for fqn, sharded_param in self._sharded_params.items():
                device = sharded_param.device
                break

        if device is None:
            raise RuntimeError("No parameters found to determine target device")

        # Materialize byte_storage if on meta device
        if self._byte_storage.device == torch.device("meta"):
            self._byte_storage = torch.empty(
                self._byte_storage.shape,
                dtype=self._byte_storage.dtype,
                device=device,
            )

        # Copy from each sharded param to byte_storage
        for fqn, info in self._param_infos.items():
            sharded_param = self._sharded_params[fqn]

            local_data = sharded_param.data

            if info.local_numel == 0:
                continue  # No data for this rank

            # Get the view into byte_storage for this param
            byte_offset = info.byte_offset
            num_bytes = info.local_numel * info.dtype.itemsize
            storage_slice = self._byte_storage[byte_offset : byte_offset + num_bytes]

            # View as the param's dtype and copy
            storage_view = storage_slice.view(info.dtype)
            storage_view.copy_(local_data.view(-1))

    def reduce_grad(self) -> None:
        """
        Reduce gradients, free unsharded buffer, and restore sharded parameters.

        Gradients from the unsharded parameters are reduced via
        Placement.reduce_grad(), and the resulting sharded gradients
        are stored on the sharded parameters.

        After calling this, model.parameters() returns sharded tensors with gradients.
        """
        if self._state == ShardedState.SHARDED:
            return  # Already sharded

        # Reduce gradients via Placement.reduce_grad()
        infos_with_grads: list[ParamInfo] = []
        grads: list[torch.Tensor] = []
        for fqn, info in self._param_infos.items():
            unsharded_param = _get_param_from_module(self._module, fqn)
            if unsharded_param.grad is not None:
                infos_with_grads.append(info)
                grads.append(unsharded_param.grad.contiguous())

        if grads:
            ptype = type(infos_with_grads[0].placements[0])
            sharded_grads = ptype.reduce_grad(grads, infos_with_grads, self._mesh)

            for info, sharded_grad in zip(infos_with_grads, sharded_grads, strict=True):
                if info.local_numel > 0:
                    self._sharded_params[info.fqn].grad = sharded_grad

        # Restore sharded parameters
        for fqn, sharded_param in self._sharded_params.items():
            _set_param_on_module(self._module, fqn, sharded_param)

        # Free unsharded buffer
        if self._unsharded_byte_storage is not None:
            self._unsharded_byte_storage = None

        self._state = ShardedState.SHARDED

    @contextmanager
    def unsharded(self):
        """
        Context manager for automatic unshard/reduce_grad around forward.

        Usage:
            with storage.unsharded():
                output = model(input)
        """
        self.unshard()
        try:
            yield
        finally:
            self.reduce_grad()


def _storage_requires_eager_batched_unshard(storage: DStorage) -> bool:
    """Return whether eager execution must use pre-gathered bucket tensors."""
    infos = list(storage._param_infos.values())
    if not infos:
        return False
    ptype = type(infos[0].placements[0])
    return (
        ptype is not _OwnedFullCopy
        and ptype is not Owned
        and ptype is not FlatShard
        and storage.byte_storage.device.type != "cpu"
    )


def _storage_uses_eager_autograd_unshard(storage: DStorage) -> bool:
    """Return whether eager RAF should use the custom bucket autograd path."""
    infos = list(storage._param_infos.values())
    if not infos:
        return False
    ptype = type(infos[0].placements[0])
    return (
        storage._reshard_after_forward
        and ptype is Shard
        and storage.byte_storage.device.type == "cuda"
    )


def _compute_local_info(
    global_shape: torch.Size,
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
) -> tuple[torch.Size, int]:
    """Compute local shape and numel for a parameter on current rank."""
    rank = mesh.get_local_rank()
    world_size = mesh.size()
    placement = placements[0]
    local_shape = placement.compute_local_shape(global_shape, rank, world_size)
    local_numel = placement.compute_local_numel(global_shape, rank, world_size)
    return local_shape, local_numel


def per_param_placements(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Placement, ...]]:
    """Shard(0) per parameter (FSDP2-style)."""
    # ep_size <> world_size, Shard(0) or Shard(1)
    return {fqn: (Shard(0),) for fqn, _ in named_params}


def flat_shard_placements(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Placement, ...]]:
    """Flatten all params into one 1D tensor divided evenly across ranks (FSDP1-style)."""
    total_flat_numel = sum(p.numel() for _, p in named_params)
    result: dict[str, tuple[Placement, ...]] = {}
    flat_offset = 0
    for fqn, param in named_params:
        result[fqn] = (FlatShard(flat_offset, param.numel(), total_flat_numel),)
        flat_offset += param.numel()
    return result


def param_boundary_placements(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Placement, ...]]:
    """Assign each parameter to one rank via greedy bin-packing (veScale-FSDP-style)."""
    assignments = _assign_params_to_ranks(named_params, mesh.size())
    return {fqn: (Owned(assignments[fqn]),) for fqn, _ in named_params}


def auto_buckets(module: nn.Module) -> list[BucketSpec]:
    """Generate one bucket per direct child module.

    Returns a list of ``BucketSpec`` objects suitable for the ``buckets``
    parameter of :func:`flex_shard`. Each bucket contains a single
    ``"child_name.*"`` pattern matching all parameters under that child.

    Example::

        >>> buckets = auto_buckets(model)
        >>> flex_shard(
        ...     model,
        ...     mesh,
        ...     dp_mesh_dims,
        ...     shard_placement_fn=per_param_placements,
        ...     buckets=buckets,
        ... )
    """
    children = list(module.named_children())
    if not children:
        return [BucketSpec(["*"])]
    return [BucketSpec([f"{name}.*"]) for name, _ in children]


def _create_param_infos(
    named_params: list[tuple[str, nn.Parameter]],
    mesh_info: FlexShardMeshInfo,
    param_placements: dict[str, tuple[Placement, ...]],
) -> tuple[dict[str, ParamInfo], int, int]:
    """
    Create ParamInfo for each parameter, computing local shapes and byte offsets.

    Placement-agnostic: works with any placement type (Shard, FlatShard, Owned, etc.).
    Parameters are laid out sequentially in the byte buffer with proper alignment.

    Args:
        named_params: List of (fqn, param) tuples
        mesh_info: Mesh metadata for sharding
        param_placements: Dict mapping FQN to placement tuple for each parameter

    Returns:
        param_infos: dict mapping FQN to ParamInfo
        total_bytes: total bytes needed for the sharded buffer
        total_unsharded_bytes: total bytes needed for the unsharded buffer
    """
    param_infos: dict[str, ParamInfo] = {}
    current_byte_offset = 0
    current_unsharded_byte_offset = 0

    for fqn, param in named_params:
        placements = param_placements[fqn]
        # FlexShard operates on the tensor local to non-DP mesh dims. DTensor
        # params already expose that through to_local(); non-materialized
        # params derive it from the recorded SPMD placements.
        spmd_mesh, spmd_placements = _get_param_spmd_metadata(param)
        param_for_shape = _get_param_tensor_for_flex_shard(
            param, mesh_info, fqn, contiguous=False
        )
        global_shape = param_for_shape.shape
        global_stride = make_contiguous_strides_for(global_shape)
        local_shape, local_numel = _compute_local_info(
            global_shape, mesh_info.dp_shard_mesh, placements
        )
        dtype = param.dtype
        global_numel = param_for_shape.numel()

        alignment = _get_dtype_alignment(dtype)

        # Sharded buffer: only allocate if this rank has data
        if local_numel > 0:
            aligned_offset = _align_offset(current_byte_offset, alignment)
            byte_offset = aligned_offset
            current_byte_offset = aligned_offset + local_numel * dtype.itemsize
        else:
            byte_offset = 0

        # Unsharded buffer: all ranks need space for all params
        aligned_unsharded_offset = _align_offset(
            current_unsharded_byte_offset, alignment
        )
        current_unsharded_byte_offset = (
            aligned_unsharded_offset + global_numel * dtype.itemsize
        )

        info = ParamInfo(
            fqn=fqn,
            global_shape=global_shape,
            global_stride=tuple(global_stride),
            dtype=dtype,
            requires_grad=param.requires_grad,
            placements=placements,
            local_shape=local_shape,
            local_numel=local_numel,
            byte_offset=byte_offset,
            global_numel=global_numel,
            unsharded_byte_offset=aligned_unsharded_offset,
            spmd_mesh=spmd_mesh,
            spmd_placements=spmd_placements,
            spmd_dp_dim_indices=mesh_info.dp_dim_indices,
        )
        param_infos[fqn] = info

    return param_infos, current_byte_offset, current_unsharded_byte_offset


def _create_sharded_view(
    local_view: torch.Tensor,
    info: ParamInfo,
    mesh_info: FlexShardMeshInfo,
) -> torch.Tensor:
    """Annotate a local tensor view with placement metadata."""
    set_sharding_info(
        local_view,
        placements=info.placements,
        global_shape=info.global_shape,
        global_stride=info.global_stride,
        mesh=mesh_info.dp_shard_mesh,
    )
    if info.spmd_mesh is not None and info.spmd_placements is not None:
        setattr(local_view, _SPMD_MESH_ATTR, info.spmd_mesh)
        setattr(local_view, _SPMD_PLACEMENTS_ATTR, info.spmd_placements)
        setattr(local_view, _SPMD_DP_DIM_INDICES_ATTR, info.spmd_dp_dim_indices)
    return local_view


def _set_param_on_module(
    root_module: nn.Module,
    fqn: str,
    param: nn.Parameter,
) -> None:
    """Navigate to submodule by FQN and set parameter."""
    parts = fqn.split(".")
    module = root_module
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], param)


def _get_param_from_module(
    root_module: nn.Module,
    fqn: str,
) -> nn.Parameter:
    """Navigate to submodule by FQN and get parameter."""
    parts = fqn.split(".")
    module = root_module
    for part in parts[:-1]:
        module = getattr(module, part)
    return getattr(module, parts[-1])


def _assign_params_to_ranks(
    named_params: list[tuple[str, nn.Parameter]],
    world_size: int,
) -> dict[str, int]:
    """
    Assign parameters to ranks using greedy bin-packing for balanced memory.

    Assigns larger parameters first to help balance the load.

    Returns:
        Dict mapping FQN to owner rank.
    """
    # Sort by size (descending) for better bin packing
    sorted_params = sorted(
        named_params,
        key=lambda x: x[1].numel() * x[1].element_size(),
        reverse=True,
    )

    rank_bytes: list[int] = [0] * world_size
    assignments: dict[str, int] = {}

    for fqn, param in sorted_params:
        # Assign to rank with least bytes
        target_rank = rank_bytes.index(min(rank_bytes))
        assignments[fqn] = target_rank
        rank_bytes[target_rank] += param.numel() * param.element_size()

    return assignments


def _write_params_to_dstorage(
    byte_storage: torch.Tensor,
    named_params: list[tuple[str, nn.Parameter]],
    param_infos: dict[str, ParamInfo],
    mesh_info: FlexShardMeshInfo,
) -> None:
    """Pack original parameter data into byte storage.

    Calls placement.extract_local_shard() to get each rank's typed local shard,
    then copies it as uint8 into the byte buffer.
    """
    mesh = mesh_info.dp_shard_mesh
    my_rank = mesh.get_local_rank()
    world_size = mesh.size()

    for fqn, param in named_params:
        info = param_infos[fqn]
        param_data = _get_param_tensor_for_flex_shard(
            param, mesh_info, fqn, contiguous=True
        )
        if param_data.device.type == "meta":
            continue
        shard = info.placements[0].extract_local_shard(param_data, my_rank, world_size)
        if shard.numel() > 0:
            nbytes = shard.numel() * shard.element_size()
            byte_storage[info.byte_offset : info.byte_offset + nbytes].copy_(
                shard.reshape(-1).view(torch.uint8)
            )


def _get_managed_named_params(
    module: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    """
    Collect parameters that should be managed by this module's DStorage.

    This excludes parameters from child modules that already have their own
    DStorage (i.e., already wrapped with flex_shard).

    Similar to FSDP2's _get_managed_modules/_get_managed_states pattern.
    """
    managed_params: list[tuple[str, nn.Parameter]] = []

    # Find child modules that already have DStorage
    wrapped_prefixes: set[str] = set()
    for name, child in module.named_modules():
        if name and getattr(child, _DSTORAGE_ATTR, None) is not None:
            # This child is already wrapped; skip its parameters
            wrapped_prefixes.add(name + ".")

    # Collect parameters not in wrapped submodules
    for fqn, param in module.named_parameters():
        is_wrapped = any(fqn.startswith(prefix) for prefix in wrapped_prefixes)
        if not is_wrapped:
            managed_params.append((fqn, param))

    return managed_params


PlacementFn = Callable[
    [list[tuple[str, nn.Parameter]], "DeviceMesh"],
    dict[str, tuple[Placement, ...]],
]

logger = logging.getLogger(__name__)


def flex_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    dp_mesh_dims: DataParallelMeshDims,
    shard_placement_fn: PlacementFn,
    buckets: list[BucketSpec],
) -> FlexShardModule:
    """
    Apply flat-storage FSDP sharding to a module.

    This function:
    1. Collects parameters from the module (excluding already-wrapped submodules)
    2. Groups parameters into communication buckets (one per bucket, or all in one)
    3. Creates a unified byte buffer per bucket for all its parameters
    4. Replaces each parameter with a plain tensor annotated with placement metadata
    5. Registers property-based parametrization for traceable communication
    6. Stores DStorages on the module (accessible via module.dstorages)

    Each bucket gets its own byte buffer and DStorage, enabling independent
    all-gather operations per bucket.

    Nested wrapping is supported: apply flex_shard to inner modules first,
    then to outer modules. The outer module's storage will exclude parameters
    from already-wrapped inner modules.

    Args:
        module: The module to shard. Can have real or meta device parameters.
        mesh: The full SPMD device mesh for sharding.
        dp_mesh_dims: Names for the data-parallel dimensions in ``mesh``.
            FlexShard derives its DP shard mesh from ``mesh`` and expects
            parameters to be DTensors on that full mesh with DP dims initially
            Replicate().
        shard_placement_fn: Required callable that maps
            ``(named_params, dp_shard_mesh)`` to per-parameter placements.
            Built-in callables include ``per_param_placements``,
            ``flat_shard_placements``, and ``param_boundary_placements``.
        buckets: Required list of bucket specifications. Use
            ``[BucketSpec(["*"])]`` for a single whole-module bucket or
            ``auto_buckets()`` to generate one bucket per direct child module.

    Returns:
        The module (mutated in-place). Use module.dstorages to access internals.

    Example::

        >>> mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        >>> model = Transformer(args)
        >>> model.parallelize(mesh, wrap_forward=False, distribute_buffers=False)
        >>> # Single bucket:
        >>> flex_shard(
        ...     model,
        ...     mesh,
        ...     DataParallelMeshDims(shard="fsdp"),
        ...     shard_placement_fn=per_param_placements,
        ...     buckets=[BucketSpec(["*"])],
        ... )
        >>> # Explicit buckets:
        >>> flex_shard(
        ...     model,
        ...     mesh,
        ...     DataParallelMeshDims(shard="fsdp"),
        ...     shard_placement_fn=per_param_placements,
        ...     buckets=[BucketSpec(["attn.*"]), BucketSpec(["ffn.*"])],
        ... )
        >>> # Auto buckets (one per child):
        >>> flex_shard(
        ...     model,
        ...     mesh,
        ...     DataParallelMeshDims(shard="fsdp"),
        ...     shard_placement_fn=per_param_placements,
        ...     buckets=auto_buckets(model),
        ... )

    Note:
        - Parameters of different dtypes are supported in a single unified buffer
        - Proper alignment is maintained for each dtype
        - Parameters on meta device will have uninitialized storage
        - Each bucket must have consistent placement types
    """
    # Check if module is already wrapped
    if getattr(module, _DSTORAGES_ATTR, None) is not None:
        raise ValueError(
            f"Module {type(module).__name__} already has DStorage. "
            "Cannot apply flex_shard twice to the same module."
        )

    # Collect parameters (excluding those from already-wrapped submodules)
    named_params = _get_managed_named_params(module)
    if not named_params:
        raise ValueError(
            f"Module {type(module).__name__} has no parameters to shard. "
            "All parameters may belong to already-wrapped submodules."
        )
    mesh_info = _get_flex_shard_mesh_info(mesh, dp_mesh_dims)
    shard_mesh = mesh_info.dp_shard_mesh

    # Determine device - use meta only if all params are meta, otherwise use mesh device.
    all_params_meta = all(
        _get_param_tensor_for_flex_shard(
            param, mesh_info, fqn, contiguous=False
        ).device.type
        == "meta"
        for fqn, param in named_params
    )
    if all_params_meta:
        device = torch.device("meta")
    else:
        device = _get_device_from_mesh(shard_mesh)
    _validate_global_spmd_params(
        named_params,
        mesh_info,
        expected_device=None if all_params_meta else device,
    )

    # Resolve placements for all params
    param_placements = shard_placement_fn(named_params, shard_mesh)
    expected_fqns = {fqn for fqn, _ in named_params}
    actual_fqns = set(param_placements)
    missing_fqns = expected_fqns - actual_fqns
    extra_fqns = actual_fqns - expected_fqns
    if missing_fqns or extra_fqns:
        msg_parts = []
        if missing_fqns:
            msg_parts.append(f"missing placements for {sorted(missing_fqns)}")
        if extra_fqns:
            msg_parts.append(f"unexpected placements for {sorted(extra_fqns)}")
        raise ValueError(
            "shard_placement_fn must return placements for exactly the managed "
            f"parameters; {', '.join(msg_parts)}."
        )
    _validate_placements_for_tracing(
        param_placements, named_params, mesh_info, shard_mesh
    )

    if not buckets:
        raise ValueError("flex_shard requires at least one BucketSpec in buckets.")
    if not all(isinstance(bucket, BucketSpec) for bucket in buckets):
        raise TypeError("flex_shard buckets must be a list of BucketSpec objects.")

    param_fqns = [fqn for fqn, _ in named_params]
    bucket_assignments = _assign_params_to_buckets(param_fqns, buckets)
    _validate_bucket_placements(bucket_assignments, param_placements, buckets)

    # Log bucket coverage
    if logger.isEnabledFor(logging.DEBUG):
        lines = ["flex_shard bucket coverage:"]
        total_params = 0
        for i, fqns in enumerate(bucket_assignments):
            patterns = buckets[i].patterns
            lines.append(f"  bucket {i} {patterns}: {len(fqns)} params")
            total_params += len(fqns)
        lines.append(f"  total: {total_params} params across {len(buckets)} buckets")
        logger.debug("\n".join(lines))

    # In parametrization mode, adjust placements for traceability:
    # - FlatShard: decompose bucket-level into per-parameter flat sharding
    # - Owned: store full param on all ranks (broadcast ensures consistency)
    named_params_dict_tmp = dict(named_params)
    for fqn in param_placements:
        placement = param_placements[fqn][0]
        if isinstance(placement, FlatShard):
            numel = _get_param_tensor_for_flex_shard(
                named_params_dict_tmp[fqn], mesh_info, fqn, contiguous=False
            ).numel()
            param_placements[fqn] = (FlatShard(0, numel, numel),)
        elif isinstance(placement, Owned):
            # In parametrization mode, all ranks store the full param.
            # Replace with _OwnedFullCopy so compute_local_shape/numel
            # and extract_local_shard return the full param for all ranks.
            param_placements[fqn] = (_OwnedFullCopy(placement.owner_rank),)

    # Per-bucket: create param_infos, byte buffer, replace params, create DStorage
    named_params_dict = dict(named_params)
    storages: list[DStorage] = []
    fqn_to_mp_policy: dict[str, MixedPrecisionPolicy | None] = {}
    fqn_to_offload_policy: dict[str, OffloadPolicy | None] = {}
    fqn_to_reshard_after_forward: dict[str, bool] = {}

    for bucket_idx, bucket_fqns in enumerate(bucket_assignments):
        if not bucket_fqns:
            continue

        # Extract per-bucket policies
        bucket_mp_policy = None
        bucket_offload_policy = None
        bucket_spec = buckets[bucket_idx]
        bucket_mp_policy = bucket_spec.mp_policy
        bucket_offload_policy = bucket_spec.offload_policy
        bucket_reshard_after_forward = bucket_spec.reshard_after_forward
        for fqn in bucket_fqns:
            fqn_to_mp_policy[fqn] = bucket_mp_policy
            fqn_to_offload_policy[fqn] = bucket_offload_policy
            fqn_to_reshard_after_forward[fqn] = bucket_reshard_after_forward

        bucket_named_params = [(fqn, named_params_dict[fqn]) for fqn in bucket_fqns]
        bucket_placements = {fqn: param_placements[fqn] for fqn in bucket_fqns}

        param_infos, total_bytes, total_unsharded_bytes = _create_param_infos(
            bucket_named_params, mesh_info, bucket_placements
        )

        if bucket_offload_policy is not None:
            byte_storage = torch.empty(
                total_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=bucket_offload_policy.pin_memory,
            )
        else:
            byte_storage = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        _write_params_to_dstorage(
            byte_storage, bucket_named_params, param_infos, mesh_info
        )

        for fqn, info in param_infos.items():
            local_view = byte_storage[
                info.byte_offset : info.byte_offset
                + info.local_numel * info.dtype.itemsize
            ]
            typed_view = local_view.view(info.dtype).view(info.local_shape)
            new_param = nn.Parameter(typed_view, requires_grad=info.requires_grad)
            expected_param_device = (
                torch.device("cpu") if bucket_offload_policy is not None else device
            )
            if new_param.device != expected_param_device:
                raise AssertionError(
                    f"Expected sharded parameter {fqn!r} on "
                    f"{expected_param_device}, but got {new_param.device}"
                )
            _create_sharded_view(new_param, info, mesh_info)
            _set_param_on_module(module, fqn, new_param)

        storage = DStorage(
            byte_storage,
            param_infos,
            shard_mesh,
            total_bytes,
            total_unsharded_bytes,
            module,
            reshard_after_forward=bucket_reshard_after_forward,
        )
        storages.append(storage)

    # Store DStorages on module
    setattr(module, _DSTORAGES_ATTR, storages)
    setattr(module, _DSTORAGE_ATTR, storages[0] if storages else None)
    setattr(module, _EAGER_COMM_CONTEXTS_ATTR, {})

    # Change module class to include FlexShardModule mixin
    cls = type(module)
    if not issubclass(cls, FlexShardModule):
        module.__class__ = type(cls.__name__, (cls, FlexShardModule), {})

    # Register property-based parametrization.
    group_name = shard_mesh.get_group().group_name
    world_size = shard_mesh.size()

    # Group parametrizations by leaf module (across all buckets)
    module_param_map: dict[nn.Module, dict[str, nn.Module]] = {}

    for s in storages:
        requires_eager_batched_unshard = _storage_requires_eager_batched_unshard(s)
        uses_eager_autograd_unshard = _storage_uses_eager_autograd_unshard(s)
        bucket_fqn = _get_storage_debug_fqn(s)
        for fqn, info in s._param_infos.items():
            mp = fqn_to_mp_policy.get(fqn)
            offload = fqn_to_offload_policy.get(fqn)
            param_dtype = mp.param_dtype if mp else None
            reduce_dtype = mp.reduce_dtype if mp else None
            compute_device = torch.device(device) if offload is not None else None
            reshard = fqn_to_reshard_after_forward[fqn]
            placement = info.placements[0]
            p = placement.create_parametrization(
                info,
                group_name,
                world_size,
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                compute_device=compute_device,
                reshard_after_forward=reshard,
            )

            # Wrap for DTensor outputs needed by non-DP composition.
            if info.spmd_mesh is not None and info.spmd_placements is not None:
                p = DTensorAwareParametrization(
                    p,
                    mesh=info.spmd_mesh,
                    placements=info.spmd_placements,
                    dp_dim_indices=info.spmd_dp_dim_indices,
                )

            setattr(
                p,
                _REQUIRES_EAGER_BATCHED_UNSHARD_ATTR,
                requires_eager_batched_unshard,
            )
            setattr(
                p,
                _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR,
                uses_eager_autograd_unshard,
            )
            setattr(p, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, False)
            setattr(p, _PARAM_FQN_ATTR, fqn)
            setattr(p, _BUCKET_FQN_ATTR, bucket_fqn)

            # Find the leaf module owning this param
            parts = fqn.split(".")
            leaf_mod = module
            for part in parts[:-1]:
                leaf_mod = getattr(leaf_mod, part)
            local_name = parts[-1]

            if leaf_mod not in module_param_map:
                module_param_map[leaf_mod] = {}
            module_param_map[leaf_mod][local_name] = p

    for mod, param_map in module_param_map.items():
        _register_parametrization(mod, param_map)

    # Reshard-after-forward: free unsharded params after each layer's
    # forward, recompute (re-all-gather) in backward.
    # In compiled modes (JIT/AOT), the graph pass handles this.
    # In eager mode, we wrap each layer in checkpoint with a selective
    # policy that recomputes only collective ops (all-gather, broadcast),
    # saving all compute ops (matmul, attention) to avoid redundant work.
    reshard_storages = [s for s in storages if s._reshard_after_forward]
    if _reshard_checkpoint_enabled.get() and reshard_storages:
        _apply_reshard_checkpoint(module, reshard_storages)

    # Install batched all-gather hooks for eager mode when the storage layout
    # supports the batched Placement.unshard() path.
    _install_batched_allgather_hooks(storages, module_param_map)

    return module


def _wrap_with_reshard(child: nn.Module) -> nn.Module:
    """Wrap a single module with reshard checkpoint, composing with AC if present.

    If the child is already wrapped by AC's CheckpointWrapper, unwraps it,
    merges the AC policy with FlexShard's reshard policy, and re-wraps once.
    If no AC wrapper exists, wraps with reshard-only policy.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointWrapper,
    )
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    def _reshard_only_context_fn():
        forward_ctx, recompute_ctx = create_selective_checkpoint_contexts(
            _flex_shard_reshard_policy
        )
        return forward_ctx, _mark_reshard_checkpoint_recompute(recompute_ctx)

    if isinstance(child, CheckpointWrapper):
        # AC already applied — unwrap, merge policies, re-wrap
        inner = child._checkpoint_wrapped_module
        ac_kwargs = dict(child.checkpoint_fn.keywords)
        ac_kwargs.pop("use_reentrant", None)
        ac_context_fn = ac_kwargs.pop("context_fn", None)
        if ac_context_fn is not None:
            # Selective AC — merge with reshard policy
            merged_fn = _compose_reshard_with_ac_policy(ac_context_fn)
        else:
            # Full AC — add reshard policy via selective context
            merged_fn = _reshard_only_context_fn
        return checkpoint_wrapper(inner, context_fn=merged_fn, **ac_kwargs)

    # No AC — reshard-only wrapping
    return checkpoint_wrapper(child, context_fn=_reshard_only_context_fn)


def _module_path_common_prefix(paths: list[str]) -> str:
    """Return the common module path prefix for parameter-owner module paths."""
    if not paths:
        return ""
    common_parts = paths[0].split(".") if paths[0] else []
    for path in paths[1:]:
        parts = path.split(".") if path else []
        limit = min(len(common_parts), len(parts))
        i = 0
        while i < limit and common_parts[i] == parts[i]:
            i += 1
        common_parts = common_parts[:i]
        if not common_parts:
            break
    return ".".join(common_parts)


def _get_module_by_path(module: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted module path from a root module."""
    result = module
    for part in path.split("."):
        if part:
            result = getattr(result, part)
    return result


def _set_module_by_path(module: nn.Module, path: str, child: nn.Module) -> None:
    """Set a dotted module path on a root module."""
    parts = path.split(".")
    parent = (
        _get_module_by_path(module, ".".join(parts[:-1])) if len(parts) > 1 else module
    )
    name = parts[-1]
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and name.isdigit():
        parent[int(name)] = child
    elif isinstance(parent, nn.ModuleDict):
        parent[name] = child
    else:
        setattr(parent, name, child)


def _top_level_owner_path(module: nn.Module, owner_path: str) -> str:
    """Choose the outer module to checkpoint for a parameter owner path."""
    parts = owner_path.split(".")
    if not parts or not parts[0]:
        return ""
    child = getattr(module, parts[0])
    if (
        isinstance(child, (nn.ModuleDict, nn.ModuleList, nn.Sequential))
        and len(parts) > 1
    ):
        return ".".join(parts[:2])
    return parts[0]


def _get_storage_reshard_module_paths(storage: DStorage) -> list[str]:
    """Return module paths to checkpoint for one resharding bucket."""
    owner_paths = sorted(
        {".".join(fqn.split(".")[:-1]) for fqn in storage._param_infos if "." in fqn}
    )
    if not owner_paths:
        return []

    common = _module_path_common_prefix(owner_paths)
    if common:
        target = _get_module_by_path(storage._module, common)
        if isinstance(target, (nn.ModuleDict, nn.ModuleList)):
            return sorted(
                {
                    _top_level_owner_path(storage._module, owner_path)
                    for owner_path in owner_paths
                }
            )
        return [common]
    return sorted(
        {
            _top_level_owner_path(storage._module, owner_path)
            for owner_path in owner_paths
        }
    )


def _get_storage_debug_fqn(storage: DStorage) -> str | None:
    """Return a concise module/bucket FQN for profiler annotations."""
    owner_paths = sorted(
        {".".join(fqn.split(".")[:-1]) for fqn in storage._param_infos}
    )
    if not owner_paths:
        return None
    common = _module_path_common_prefix(owner_paths)
    if common:
        return common
    top_level_paths = sorted(
        {
            _top_level_owner_path(storage._module, owner_path)
            for owner_path in owner_paths
        }
    )
    top_level_paths = [path for path in top_level_paths if path]
    if not top_level_paths:
        return None
    return ", ".join(top_level_paths)


def _apply_reshard_checkpoint(
    module: nn.Module,
    reshard_storages: list[DStorage],
) -> None:
    """Wrap FlexShard-managed bucket modules in checkpoint for reshard.

    Each selected bucket's owning module gets wrapped with a checkpoint policy that marks
    collective ops (all-gather, broadcast, wait_tensor) as MUST_RECOMPUTE
    so unsharded params are freed after each layer's forward.

    Composes with activation checkpointing: if a child is already wrapped
    by AC's CheckpointWrapper, the two policies are merged into a single
    wrapper (FlexShard collectives → MUST_RECOMPUTE, AC compute ops →
    MUST_SAVE, everything else → PREFER_RECOMPUTE).
    """
    paths: set[str] = set()
    for storage in reshard_storages:
        storage_paths = _get_storage_reshard_module_paths(storage)
        if storage_paths:
            paths.update(storage_paths)
        else:
            paths.update(name for name, _ in module.named_children())

    for path in sorted(paths, key=lambda p: (p.count("."), p)):
        child = _get_module_by_path(module, path)
        _set_module_by_path(module, path, _wrap_with_reshard(child))


def _install_batched_allgather_hooks(
    storages: list,
    module_param_map: dict[nn.Module, dict[str, nn.Module]],
) -> None:
    """Install pre/post forward hooks for batched per-bucket all-gather.

    In eager mode, each DStorage's pre-forward hook runs a single batched
    Placement.unshard() call (one NCCL collective per bucket), then sets
    _pre_gathered on each parametrization module so the property getter
    skips the per-param all-gather.

    Skipped under torch.compile — compiled modes use per-param
    _c10d_functional ops which the compiler rebatches via graph passes.
    """
    for storage in storages:
        infos = list(storage._param_infos.values())
        if not infos:
            continue

        ptype = type(infos[0].placements[0])
        # Skip batching when the placement/storage does not support the eager
        # batched path. Parametrization remains the source of truth.
        if not _storage_requires_eager_batched_unshard(storage):
            continue

        # Pre-compute (leaf_module, param_name, parametrization, info) for
        # each param in this bucket. Captured at flex_shard() time (before
        # checkpoint wrapping changes the module tree).
        param_entries: list[tuple[nn.Module, str, nn.Module, ParamInfo]] = []
        for info in infos:
            parts = info.fqn.split(".")
            leaf_mod = storage._module
            for part in parts[:-1]:
                child = getattr(leaf_mod, part, None)
                if child is None:
                    wrapped = getattr(leaf_mod, "_checkpoint_wrapped_module", None)
                    if wrapped is not None:
                        leaf_mod = getattr(wrapped, part)
                    else:
                        leaf_mod = getattr(leaf_mod, part)
                else:
                    leaf_mod = child
            local_name = parts[-1]
            # Unwrap CheckpointWrapper to find the original module
            # that's in module_param_map
            if hasattr(leaf_mod, "_checkpoint_wrapped_module"):
                leaf_mod = leaf_mod._checkpoint_wrapped_module
            if leaf_mod in module_param_map:
                param_p = module_param_map[leaf_mod].get(local_name)
                if param_p is not None:
                    param_entries.append((leaf_mod, local_name, param_p, info))

        logger.debug(f"Batched hooks: {len(param_entries)}/{len(infos)} params matched")
        if not param_entries:
            continue

        ag_context = None
        if ptype is Shard and storage.byte_storage.device.type == "cuda":
            device = storage.byte_storage.device
            ag_context = _get_or_create_eager_comm_context(storage._module, device)
        ag_bucket = None
        if ag_context is not None:
            ag_bucket = EagerAllGatherBucket(
                storage=storage,
                entries=param_entries,
                infos=infos,
                debug_fqn=_get_storage_debug_fqn(storage),
                use_autograd_unshard=_storage_uses_eager_autograd_unshard(storage),
            )
        use_autograd_unshard = _storage_uses_eager_autograd_unshard(storage)

        def make_hooks(
            s,
            entries,
            pt,
            all_gather_context,
            all_gather_bucket,
            use_autograd_bucket,
        ):
            # Collected grads from AccumulateGrad hooks (indexed by position)
            collected_grads: dict[int, torch.Tensor] = {}

            def _begin_bucket_unshard(bucket):
                local_shards = [
                    leaf._parameters[name].data for leaf, name, _, _ in bucket.entries
                ]
                return Shard.begin_unshard(
                    local_shards,
                    bucket.infos,
                    bucket.storage._mesh,
                    all_gather_context.all_gather_stream,
                    debug_fqn=bucket.debug_fqn,
                )

            def _wait_bucket_unshard(result):
                Shard.wait_for_unshard(result)

            def _prefetch_next_bucket():
                if all_gather_context.pending is not None:
                    return
                prefetch_order = all_gather_context.buckets
                if _reshard_checkpoint_recompute.get():
                    prefetch_order = prefetch_order[::-1]
                for idx, bucket in enumerate(prefetch_order):
                    if bucket is all_gather_bucket:
                        break
                else:
                    return
                next_idx = idx + 1
                if next_idx >= len(prefetch_order):
                    return
                next_bucket = prefetch_order[next_idx]
                all_gather_context.pending = PendingEagerAllGather(
                    bucket=next_bucket,
                    result=_begin_bucket_unshard(next_bucket),
                    recompute=_reshard_checkpoint_recompute.get(),
                )

            def _take_pending_for_current_bucket():
                pending = all_gather_context.pending
                if pending is None:
                    return None
                if (
                    pending.bucket is all_gather_bucket
                    and pending.recompute == _reshard_checkpoint_recompute.get()
                ):
                    all_gather_context.pending = None
                    return pending.result
                _wait_bucket_unshard(pending.result)
                all_gather_context.pending = None
                return None

            def _reduce_fn():
                """Batched reduce-scatter using collected grads."""
                grads, valid = [], []
                for idx, (leaf, name, param_p, info) in enumerate(entries):
                    g = collected_grads.pop(idx, None)
                    if g is not None:
                        grads.append(g)
                        valid.append((leaf, name, info))
                    param_p._unsharded_for_reduce = None
                if not grads:
                    return
                valid_infos = [i for _, _, i in valid]
                with torch.no_grad():
                    if (
                        all_gather_context is not None
                        and all_gather_bucket is not None
                        and pt is Shard
                    ):
                        _wait_and_clear_reduce_scatter_states(
                            all_gather_context,
                            all_gather_bucket.debug_fqn,
                        )
                        result = Shard.begin_reduce_grad(
                            grads,
                            valid_infos,
                            s._mesh,
                            all_gather_context.reduce_scatter_stream,
                            debug_fqn=all_gather_bucket.debug_fqn,
                        )
                        stored_grads: list[torch.Tensor] = []
                        with torch.cuda.stream(
                            all_gather_context.reduce_scatter_stream
                        ):
                            for (leaf, name, _), rg in zip(
                                valid, result.sharded_grads, strict=True
                            ):
                                param = leaf._parameters[name]
                                if rg.dtype != param.dtype:
                                    rg = rg.to(param.dtype)
                                stored_grads.append(rg)
                                if param.grad is None:
                                    param.grad = rg
                                else:
                                    param.grad += rg
                            result.sharded_grads = stored_grads
                            result.event = torch.cuda.Event()
                            result.event.record(
                                all_gather_context.reduce_scatter_stream
                            )
                        all_gather_context.reduce_scatter_states.append(result)
                        _queue_reduce_scatter_wait(all_gather_context)
                    else:
                        reduced = pt.reduce_grad(grads, valid_infos, s._mesh)
                        for (leaf, name, _), rg in zip(valid, reduced, strict=True):
                            param = leaf._parameters[name]
                            if rg.dtype != param.dtype:
                                rg = rg.to(param.dtype)
                            if param.grad is None:
                                param.grad = rg
                            else:
                                param.grad += rg

            def pre_forward_hook(mod, args):
                if torch.compiler.is_compiling():
                    return
                # Batched all-gather
                local_shards = [
                    (
                        leaf._parameters[name]
                        if use_autograd_bucket
                        else leaf._parameters[name].data
                    )
                    for leaf, name, _, _ in entries
                ]
                entry_infos = [info for _, _, _, info in entries]
                if (
                    all_gather_context is not None
                    and all_gather_bucket is not None
                    and pt is Shard
                ):
                    if use_autograd_bucket:
                        prefetched_result = _take_pending_for_current_bucket()
                        runtime = EagerBucketAllGatherRuntime(
                            prefetched_result=prefetched_result,
                            infos=entry_infos,
                            param_refs=[(leaf, name) for leaf, name, _, _ in entries],
                            mesh=s._mesh,
                            context=all_gather_context,
                            debug_fqn=all_gather_bucket.debug_fqn,
                        )
                        full_params = list(
                            _EagerBucketAllGather.apply(runtime, *local_shards)
                        )
                        _prefetch_next_bucket()
                    else:
                        with torch.no_grad():
                            result = _take_pending_for_current_bucket()
                            if result is None:
                                result = Shard.begin_unshard(
                                    local_shards,
                                    entry_infos,
                                    s._mesh,
                                    all_gather_context.all_gather_stream,
                                    debug_fqn=all_gather_bucket.debug_fqn,
                                )
                            full_params = Shard.finish_unshard(result)
                            _prefetch_next_bucket()
                else:
                    with torch.no_grad():
                        full_params = pt.unshard(local_shards, entry_infos, s._mesh)
                for (_, _, param_p, _), full_param in zip(
                    entries, full_params, strict=True
                ):
                    param_p._pre_gathered = full_param

            def post_forward_hook(mod, args, output):
                if torch.compiler.is_compiling():
                    return
                for _, _, param_p, _ in entries:
                    param_p._pre_gathered = None
                if use_autograd_bucket:
                    return

                # Register AccumulateGrad hooks on detached leaf params.
                # Each hook stores its grad by index. The last hook fires
                # _reduce_fn with all collected grads (u.grad is None at
                # hook time — grads must be captured from the hook argument).
                if torch.is_grad_enabled():
                    collected_grads.clear()
                    leaf_indices = []
                    for idx, (_, _, param_p, _) in enumerate(entries):
                        u = getattr(param_p, "_unsharded_for_reduce", None)
                        if u is not None and u.requires_grad:
                            leaf_indices.append((idx, u))

                    if leaf_indices:
                        grad_count = [0]
                        n = len(leaf_indices)

                        def _make_hook(i):
                            def _on_grad(grad):
                                collected_grads[i] = grad
                                grad_count[0] += 1
                                if grad_count[0] >= n:
                                    _reduce_fn()
                                    grad_count[0] = 0

                            return _on_grad

                        for idx, leaf in leaf_indices:
                            leaf.register_hook(_make_hook(idx))

            return pre_forward_hook, post_forward_hook

        pre_hook, post_hook = make_hooks(
            storage,
            param_entries,
            ptype,
            ag_context,
            ag_bucket,
            use_autograd_unshard,
        )

        # Register hooks on the bucket's child module (not root) so they
        # fire during checkpoint recomputation in backward too.
        # Navigate through CheckpointWrapper to the inner module.
        target = _get_bucket_module(storage)
        if (
            storage._reshard_after_forward
            and target is storage._module
            and any(
                hasattr(child, "_checkpoint_wrapped_module")
                for child in storage._module.modules()
                if child is not storage._module
            )
        ):
            logger.debug(
                "Skipping root-level batched all-gather hook because child "
                "checkpoint recomputation would not replay the root hook.",
            )
            continue
        inner = getattr(target, "_checkpoint_wrapped_module", target)
        inner.register_forward_pre_hook(pre_hook)
        inner.register_forward_hook(post_hook)
        if ag_context is not None and ag_bucket is not None:
            ag_context.buckets.append(ag_bucket)
        for _, _, param_p, _ in param_entries:
            setattr(param_p, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, True)


def _get_bucket_module(storage) -> nn.Module:
    """Find the deepest common ancestor module for a bucket's params.

    For bucket "layers.0.*", returns model.layers[0].
    For bucket "norm.*, output.*" (no common prefix), returns root.
    """
    fqns = list(storage._param_infos.keys())
    # Get module-level prefixes (strip param name)
    prefixes = [".".join(fqn.split(".")[:-1]) for fqn in fqns]
    # Find common prefix
    if not prefixes:
        return storage._module
    common = prefixes[0]
    for p in prefixes[1:]:
        # Find common prefix character by character, then trim to last "."
        i = 0
        while i < len(common) and i < len(p) and common[i] == p[i]:
            i += 1
        common = common[:i]
    # Trim to last complete component (don't split mid-name)
    if "." in common:
        common = common[: common.rfind(".") + 1].rstrip(".")
    elif common and common not in prefixes:
        # Partial match — not a complete component
        common = ""
    if not common:
        return storage._module
    mod = storage._module
    for part in common.split("."):
        mod = getattr(mod, part)
    return mod
