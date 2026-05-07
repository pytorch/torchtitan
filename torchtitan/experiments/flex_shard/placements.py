# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle
from torch._prims_common import make_contiguous_strides_for
from typing_extensions import override

from .utils import _with_fqn

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_storage import ParamInfo


@dataclass(frozen=True)
class LocalStorageLayout:
    """Placement-owned local storage layout for one parameter."""

    local_shape: torch.Size
    local_numel: int
    storage_nbytes: int


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


class Placement:
    """Base class for FlexShard placement strategies.

    Each subclass implements per-param sharding (extract_local_shard,
    assemble_from_shards) and batched communication (unshard, reduce_grad).
    """

    def validate_param(self, fqn: str, param: nn.Parameter) -> None:
        """Validate that this placement can manage one parameter."""

    def validate_bucket(
        self,
        bucket_idx: int,
        bucket_patterns: list[str],
        fqn: str,
        param: nn.Parameter,
        bucket_named_params: list[tuple[str, nn.Parameter]],
    ) -> None:
        """Validate that this placement can share a bucket with these params."""

    def bucket_compatibility_key(self) -> Hashable:
        """Return a key for deciding if placements can share one bucket."""
        return (type(self), repr(self))

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

    def local_storage_layout(
        self,
        global_shape: torch.Size,
        dtype: torch.dtype,
        rank: int,
        world_size: int,
    ) -> LocalStorageLayout:
        """Return the local storage layout for one parameter."""
        local_shape = self.compute_local_shape(global_shape, rank, world_size)
        local_numel = self.compute_local_numel(global_shape, rank, world_size)
        return LocalStorageLayout(
            local_shape=local_shape,
            local_numel=local_numel,
            storage_nbytes=local_numel * dtype.itemsize,
        )

    def copy_param_to_storage(
        self,
        byte_storage: torch.Tensor,
        info: ParamInfo,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        """Pack one full parameter into its placement-owned local storage."""
        param_data = param.detach()
        if param_data.device.type == "meta":
            return
        if not param_data.is_contiguous():
            param_data = param_data.contiguous()
        shard = self.extract_local_shard(param_data, rank, world_size)
        if shard.numel() == 0:
            return
        nbytes = shard.numel() * shard.element_size()
        if nbytes > info.storage_nbytes:
            raise ValueError(
                f"Placement {self!r} produced {nbytes} bytes for {info.fqn!r}, "
                f"but its storage layout only reserved {info.storage_nbytes} bytes."
            )
        byte_storage[info.byte_offset : info.byte_offset + nbytes].copy_(
            shard.reshape(-1).view(torch.uint8)
        )

    def make_local_storage_view(
        self,
        byte_storage: torch.Tensor,
        info: ParamInfo,
    ) -> torch.Tensor:
        """Return the exposed local parameter view from bucket storage."""
        nbytes = info.local_numel * info.dtype.itemsize
        byte_view = byte_storage[info.byte_offset : info.byte_offset + nbytes]
        return byte_view.view(info.dtype).view(info.local_shape)

    def unshard(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        """Convenience wrapper for a same-stream unshard."""
        device_handle = _get_device_handle(tensors[0].device.type)
        result = self.begin_unshard(
            tensors,
            infos,
            mesh,
            device_handle.current_stream(tensors[0].device),
            debug_fqn=None,
        )
        return result.finish()

    def reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        """Convenience wrapper for a same-stream gradient reduction."""
        device_handle = _get_device_handle(tensors[0].device.type)
        result = self.begin_reduce_grad(
            tensors,
            infos,
            mesh,
            device_handle.current_stream(tensors[0].device),
            debug_fqn=None,
        )
        return result.finish()

    def begin_unshard(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        all_gather_stream: torch.Stream,
        debug_fqn: str | None = None,
    ) -> PlacementUnshardResult:
        """Begin an unshard operation, possibly asynchronously."""
        raise NotImplementedError

    def begin_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        reduce_scatter_stream: torch.Stream,
        debug_fqn: str | None = None,
    ) -> PlacementReduceGradResult:
        """Begin a gradient reduction operation, possibly asynchronously."""
        raise NotImplementedError


def _get_single_placement(placements: tuple[Placement, ...]) -> Placement:
    """Return the only placement supported by the minimal eager path."""
    if len(placements) != 1:
        raise ValueError(
            "FlexShard eager mode currently supports exactly one placement "
            f"per parameter, but got {len(placements)} placements."
        )
    return placements[0]


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

    @override
    def validate_param(self, fqn: str, param: nn.Parameter) -> None:
        if self.dim >= param.ndim:
            raise ValueError(
                f"Parameter {fqn!r} has {param.ndim} dimensions but "
                f"Shard(dim={self.dim}) is out of range."
            )

    @override
    def validate_bucket(
        self,
        bucket_idx: int,
        bucket_patterns: list[str],
        fqn: str,
        param: nn.Parameter,
        bucket_named_params: list[tuple[str, nn.Parameter]],
    ) -> None:
        if self.dim != 0:
            raise ValueError(
                f"Bucket {bucket_idx} "
                f"{bucket_patterns} "
                f"parameter {fqn!r} uses {self!r}. "
                "FlexShard eager mode currently supports only Shard(0) "
                "placements."
            )

    @override
    def bucket_compatibility_key(self) -> Hashable:
        return (Shard, self.dim)

    @override
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

    @override
    def compute_local_numel(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> int:
        local_shape = self.compute_local_shape(global_shape, rank, world_size)
        numel = 1
        for d in local_shape:
            numel *= d
        return numel

    @override
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

    @override
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

    @override
    def begin_unshard(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        all_gather_stream: torch.Stream,
        debug_fqn: str | None = None,
    ) -> PlacementUnshardResult:
        from .placement_results import AsyncAllGatherResult

        ws = mesh.size()
        pg = mesh.get_group()
        dtype = tensors[0].dtype
        device = tensors[0].device
        device_handle = _get_device_handle(device.type)

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
                    offset += info.placement.compute_local_numel(
                        info.global_shape, r, ws
                    )
                per_rank_sizes.append(offset)
                per_rank_param_offsets.append(offsets_r)

            # Variable-size all_gather outputs
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

    @override
    def begin_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        reduce_scatter_stream: torch.Stream,
        debug_fqn: str | None = None,
    ) -> PlacementReduceGradResult:
        from .placement_results import AsyncReduceScatterResult

        ws = mesh.size()
        rank = mesh.get_local_rank()
        pg = mesh.get_group()
        dtype = tensors[0].dtype
        device = tensors[0].device
        device_handle = _get_device_handle(device.type)

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
                    local_shape = info.placement.compute_local_shape(
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

        copy_in_done = device_handle.Event()
        copy_in_done.record(device_handle.current_stream(device))
        recv_buf: torch.Tensor
        with device_handle.stream(reduce_scatter_stream):
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


def per_param_placements(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Placement, ...]]:
    """Shard(0) per parameter (FSDP2-style)."""
    return {fqn: (Shard(0),) for fqn, _ in named_params}


__all__ = [
    "LocalStorageLayout",
    "per_param_placements",
    "Placement",
    "PlacementReduceGradResult",
    "PlacementUnshardResult",
    "Shard",
]
