# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for

from .comm_buffer_lifetime import (
    AsyncAllGatherResult,
    AsyncReduceScatterResult,
    StreamHandoff,
)
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

    def supports_graph_capture(self, device: torch.device) -> bool:
        """Return whether this placement supports graph capture."""
        return False

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
        """Batched gather communication for all params in a storage unit."""
        raise NotImplementedError

    def reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        """Batched reduce communication for all param gradients in a storage unit."""
        raise NotImplementedError

    def begin_unshard(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        all_gather_stream: torch.Stream | None,
        debug_fqn: str | None = None,
    ) -> Any:
        """Begin an unshard operation, possibly asynchronously."""
        raise NotImplementedError

    def finish_unshard(self, result: Any) -> list[torch.Tensor]:
        """Finish a result returned by begin_unshard()."""
        raise NotImplementedError

    def wait_for_unshard(self, result: Any) -> None:
        """Wait for a begin_unshard() result to become usable."""

    def release_unshard_buffers(self, result: Any) -> None:
        """Release temporary buffers owned by an unshard result."""

    def begin_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        reduce_scatter_stream: torch.Stream | None,
        debug_fqn: str | None = None,
    ) -> Any:
        """Begin a gradient reduction operation, possibly asynchronously."""
        raise NotImplementedError

    def finish_reduce_grad(self, result: Any) -> list[torch.Tensor]:
        """Finish a result returned by begin_reduce_grad()."""
        raise NotImplementedError

    def wait_for_reduce_grad(self, result: Any) -> None:
        """Wait for a begin_reduce_grad() result to become usable."""

    def release_reduce_grad_buffers(
        self,
        result: Any,
        release_sharded_grads: bool,
    ) -> None:
        """Release temporary buffers owned by a reduce-grad result."""


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

    def validate_param(self, fqn: str, param: nn.Parameter) -> None:
        if self.dim >= param.ndim:
            raise ValueError(
                f"Parameter {fqn!r} has {param.ndim} dimensions but "
                f"Shard(dim={self.dim}) is out of range."
            )

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

    def bucket_compatibility_key(self) -> Hashable:
        return (Shard, self.dim)

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

    def unshard(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        result = self.begin_unshard(
            tensors,
            infos,
            mesh,
            all_gather_stream=None,
            debug_fqn=None,
        )
        return self.finish_unshard(result)

    def begin_unshard(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        all_gather_stream: torch.Stream | None,
        debug_fqn: str | None = None,
    ) -> AsyncAllGatherResult:
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

        if all_gather_stream is None or device.type != "cuda":
            label = _with_fqn("FlexShard::all_gather", debug_fqn)
            with dist.record_comm(label):
                dist.all_gather(gathered, send_buf, group=pg)
            event = None
            if device.type == "cuda":
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
            return AsyncAllGatherResult(
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
        return AsyncAllGatherResult(
            gathered=gathered,
            infos=infos,
            mesh=mesh,
            per_rank_param_offsets=per_rank_param_offsets,
            event=event,
            send_buf=send_buf,
            debug_fqn=debug_fqn,
        )

    def finish_unshard(self, result: AsyncAllGatherResult) -> list[torch.Tensor]:
        self.wait_for_unshard(result)
        ws = result.mesh.size()
        device = result.gathered[0].device
        # Unpack: per param, extract shard from each rank, assemble_from_shards
        with torch.profiler.record_function(
            _with_fqn("FlexShard::all_gather_copy_out", result.debug_fqn)
        ):
            results: list[torch.Tensor] = []
            for i, info in enumerate(result.infos):
                p = info.placement
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
            self.release_unshard_buffers(result)
            return results

    def wait_for_unshard(self, result: AsyncAllGatherResult) -> None:
        device = result.gathered[0].device
        if device.type == "cuda" and result.event is not None:
            torch.cuda.current_stream(device).wait_event(result.event)

    def release_unshard_buffers(self, result: AsyncAllGatherResult) -> None:
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

    def reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
    ) -> list[torch.Tensor]:
        result = self.begin_reduce_grad(
            tensors,
            infos,
            mesh,
            reduce_scatter_stream=None,
            debug_fqn=None,
        )
        return self.finish_reduce_grad(result)

    def begin_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        reduce_scatter_stream: torch.Stream | None,
        debug_fqn: str | None = None,
    ) -> AsyncReduceScatterResult:
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
            return AsyncReduceScatterResult(
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
        return AsyncReduceScatterResult(
            sharded_grads=sharded_grads,
            event=event,
            send_buf=send_buf,
            recv_buf=recv_buf,
            debug_fqn=debug_fqn,
        )

    def finish_reduce_grad(
        self, result: AsyncReduceScatterResult
    ) -> list[torch.Tensor]:
        self.wait_for_reduce_grad(result)
        return result.sharded_grads

    def wait_for_reduce_grad(self, result: AsyncReduceScatterResult) -> None:
        device = (
            result.recv_buf.device
            if result.recv_buf is not None
            else result.sharded_grads[0].device
        )
        if device.type == "cuda" and result.event is not None:
            torch.cuda.current_stream(device).wait_event(result.event)

    def release_reduce_grad_buffers(
        self,
        result: AsyncReduceScatterResult,
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
    "Shard",
]
