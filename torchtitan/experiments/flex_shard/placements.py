# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for

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


def _with_fqn(label: str, fqn: str | None) -> str:
    """Append a module/bucket FQN to profiler labels, matching FSDP style."""
    if fqn:
        return f"{label} ({fqn})"
    return label


_active_parametrization = True


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


class _MixedPrecisionCast(torch.autograd.Function):
    """Cast with decoupled forward/backward dtype control."""

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


class ShardParametrization(nn.Module):
    """Parametrization for Shard placement."""

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

    def forward(self, local_shard: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return local_shard
        if (
            self.compute_device is not None
            and local_shard.device != self.compute_device
        ):
            local_shard = local_shard.to(self.compute_device, non_blocking=True)

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
        if full.requires_grad and torch.is_grad_enabled():
            full.register_hook(lambda grad: grad / self.world_size)
        if self.shard_dim != 0:
            chunks = full.chunk(self.world_size, dim=0)
            full = torch.cat(chunks, dim=self.shard_dim)

        if self.global_dim_size is not None:
            full = full.narrow(self.shard_dim, 0, self.global_dim_size)

        if self.param_dtype is not None or self.reduce_dtype is not None:
            full = _MixedPrecisionCast.apply(full, self.param_dtype, self.reduce_dtype)
        return full


class FlatShardParametrization(nn.Module):
    """Parametrization for FlatShard placement."""

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

    def forward(self, flat_shard: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return flat_shard
        if self.compute_device is not None and flat_shard.device != self.compute_device:
            flat_shard = flat_shard.to(self.compute_device, non_blocking=True)

        if self.padded_shard_size is not None:
            pad_size = self.padded_shard_size - flat_shard.numel()
            if pad_size > 0:
                padding = flat_shard.new_zeros(pad_size)
                flat_shard = torch.cat([flat_shard, padding])

        full_flat = torch.ops._c10d_functional.all_gather_into_tensor(
            flat_shard, self.world_size, self.group_name
        )
        full_flat = torch.ops._c10d_functional.wait_tensor(full_flat)
        if full_flat.requires_grad and torch.is_grad_enabled():
            full_flat.register_hook(lambda grad: grad / self.world_size)

        if self.global_numel is not None:
            full_flat = full_flat[: self.global_numel]

        full = full_flat.view(self.original_shape)
        if self.param_dtype is not None or self.reduce_dtype is not None:
            full = _MixedPrecisionCast.apply(full, self.param_dtype, self.reduce_dtype)
        return full


class _OwnedBroadcast(torch.autograd.Function):
    """Differentiable broadcast for Owned placement."""

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
    """Parametrization for Owned placement."""

    def __init__(
        self,
        owner_rank: int,
        group_name: str,
        world_size: int,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = None,
        compute_device: torch.device | None = None,
    ):
        super().__init__()
        self.owner_rank = owner_rank
        self.group_name = group_name
        self.world_size = world_size
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.compute_device = compute_device

    def forward(self, param: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return param
        if self.compute_device is not None and param.device != self.compute_device:
            param = param.to(self.compute_device, non_blocking=True)
        full = _OwnedBroadcast.apply(
            param, self.owner_rank, self.group_name, self.world_size
        )
        if self.param_dtype is not None or self.reduce_dtype is not None:
            full = _MixedPrecisionCast.apply(full, self.param_dtype, self.reduce_dtype)
        return full


class RaggedShardParametrization(nn.Module):
    """Parametrization for RaggedShard placement."""

    def __init__(
        self,
        shard_dim: int,
        split_sizes: list[int],
        group_name: str,
        world_size: int,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = None,
        compute_device: torch.device | None = None,
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

    def forward(self, local_shard: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return local_shard
        if (
            self.compute_device is not None
            and local_shard.device != self.compute_device
        ):
            local_shard = local_shard.to(self.compute_device, non_blocking=True)

        local_size = local_shard.shape[self.shard_dim]
        pad_size = self.max_shard_size - local_size
        if pad_size > 0:
            pad_shape = list(local_shard.shape)
            pad_shape[self.shard_dim] = pad_size
            padding = local_shard.new_zeros(pad_shape)
            local_shard = torch.cat([local_shard, padding], dim=self.shard_dim)

        full = torch.ops._c10d_functional.all_gather_into_tensor(
            local_shard, self.world_size, self.group_name
        )
        full = torch.ops._c10d_functional.wait_tensor(full)
        if full.requires_grad and torch.is_grad_enabled():
            full.register_hook(lambda grad: grad / self.world_size)

        chunks = full.chunk(self.world_size, dim=0)
        real_chunks = [
            chunk.narrow(self.shard_dim, 0, self.split_sizes[r])
            for r, chunk in enumerate(chunks)
        ]
        full = torch.cat(real_chunks, dim=self.shard_dim)
        if self.param_dtype is not None or self.reduce_dtype is not None:
            full = _MixedPrecisionCast.apply(full, self.param_dtype, self.reduce_dtype)
        return full


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
        """Create an eager parametrization module for this placement.

        Called by flex_shard() to build the property-getter parametrization
        used for module parameter access. Subclasses override to return their
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
                del per_rank_shards
            cls.release_unshard_buffers(result)
            return results

    @classmethod
    def wait_for_unshard(cls, result: EagerAllGatherResult) -> None:
        device = result.gathered[0].device
        if device.type == "cuda" and result.event is not None:
            torch.cuda.current_stream(device).wait_event(result.event)

    @classmethod
    def release_unshard_buffers(cls, result: EagerAllGatherResult) -> None:
        """Release raw all-gather buffers after current-stream work is queued."""
        if not result.gathered and result.send_buf is None:
            return
        device = (
            result.gathered[0].device if result.gathered else result.send_buf.device  # type: ignore[union-attr]
        )
        if device.type != "cuda":
            result.gathered.clear()
            result.send_buf = None
            return

        stream = torch.cuda.current_stream(device)
        event = torch.cuda.Event()
        event.record(stream)
        handoffs: list[StreamHandoff] = []
        if result.send_buf is not None:
            handoffs.append(StreamHandoff(result.send_buf, event, stream))
            result.send_buf = None
        while result.gathered:
            handoffs.append(StreamHandoff(result.gathered.pop(), event, stream))
        for handoff in handoffs:
            handoff.release()

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
        device = (
            result.recv_buf.device
            if result.recv_buf is not None
            else result.sharded_grads[0].device
        )
        if device.type == "cuda" and result.event is not None:
            torch.cuda.current_stream(device).wait_event(result.event)

    @classmethod
    def release_reduce_grad_buffers(
        cls,
        result: EagerReduceScatterResult,
        release_sharded_grads: bool,
    ) -> None:
        """Release pending reduce-scatter buffers after its completion wait."""
        tensors: list[torch.Tensor] = []
        if result.send_buf is not None:
            tensors.append(result.send_buf)
            result.send_buf = None
        if result.recv_buf is not None:
            tensors.append(result.recv_buf)
            result.recv_buf = None
        if release_sharded_grads:
            tensors.extend(result.sharded_grads)
            result.sharded_grads.clear()
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
    This enables eager broadcast from owner_rank via OwnedParametrization.
    The memory overhead is acceptable since Owned is typically used for small
    params (biases, norms).
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


__all__ = [
    "disable_active_parametrization",
    "FlatShard",
    "FlatShardParametrization",
    "flat_shard_placements",
    "Owned",
    "OwnedParametrization",
    "param_boundary_placements",
    "per_param_placements",
    "Placement",
    "RaggedShard",
    "RaggedShardParametrization",
    "Shard",
    "ShardParametrization",
]
