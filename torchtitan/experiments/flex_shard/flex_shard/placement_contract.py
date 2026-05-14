# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist

from .utils import _record_comm_if_eager, _record_function_if_eager

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_storage import ParamInfo


@dataclass(frozen=True)
class LocalStorageLayout:
    """Placement-owned local storage layout for one parameter."""

    local_shape: torch.Size
    local_numel: int
    storage_nbytes: int


@dataclass
class PlacementUnshardResult:
    """Placement-owned unshard result and temporary buffers."""

    full_params: list[torch.Tensor]
    buffers: list[torch.Tensor] = field(default_factory=list)


@dataclass
class PlacementPreparedReduceGrad:
    """Placement-owned packed gradient reduction request."""

    placement: Placement
    infos: list[ParamInfo]
    layout: Any
    rank: int
    world_size: int
    pg: Any
    buffers: list[torch.Tensor]
    debug_fqn: str | None


@dataclass
class PlacementReduceGradResult:
    """Placement-owned reduced local gradients and temporary buffers."""

    sharded_grads: list[torch.Tensor]
    buffers: list[torch.Tensor] = field(default_factory=list)


class Placement:
    """Base class for FlexShard placement strategies.

    Each subclass implements per-param sharding, storage layout, full-parameter
    assembly, and gradient-reduction packing. The placement also owns the
    collective pattern for bucket unshard and gradient reduction. For example,
    Shard uses all-gather and reduce-scatter, while Owned uses broadcast and
    reduce-to-owner.
    """

    def compute_local_numel(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> int:
        """How many elements this rank holds for a param with global_shape."""
        return math.prod(self.compute_local_shape(global_shape, rank, world_size))

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
        """Extract this rank's local shard from the full parameter."""
        raise NotImplementedError

    def assemble_from_shards(
        self,
        per_rank_shards: list[torch.Tensor],
        global_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reconstruct the full parameter from typed per-rank shards."""
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

    def pack_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        world_size: int,
    ) -> tuple[torch.Tensor, Any]:
        """Pack full gradients into a flat reduce-scatter input buffer."""
        raise NotImplementedError

    def unpack_reduced_grad(
        self,
        recv_buf: torch.Tensor,
        infos: list[ParamInfo],
        layout: Any,
        rank: int,
        world_size: int,
    ) -> list[torch.Tensor]:
        """Unpack a flat reduce-scatter output buffer into local grad shards."""
        raise NotImplementedError

    def unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementUnshardResult:
        """Unshard one bucket using the placement's default all-gather path."""
        ws = mesh.size()
        pg = mesh.get_group()
        dtype = tensors[0].dtype
        device = tensors[0].device

        with _record_function_if_eager("FlexShard::all_gather_copy_in", debug_fqn):
            send_buf = torch.cat([t.reshape(-1) for t in tensors])

            per_rank_sizes: list[int] = []
            per_rank_param_offsets: list[list[int]] = []
            for r in range(ws):
                offset = 0
                offsets_r: list[int] = []
                for info in infos:
                    offsets_r.append(offset)
                    offset += self.compute_local_numel(info.global_shape, r, ws)
                per_rank_sizes.append(offset)
                per_rank_param_offsets.append(offsets_r)

            gathered = [
                torch.empty(per_rank_sizes[r], dtype=dtype, device=device)
                for r in range(ws)
            ]

        with _record_comm_if_eager("FlexShard::all_gather", debug_fqn):
            dist.all_gather(gathered, send_buf, group=pg)

        with _record_function_if_eager("FlexShard::all_gather_copy_out", debug_fqn):
            full_params = []
            for i, info in enumerate(infos):
                per_rank_shards: list[torch.Tensor] = []
                for r in range(ws):
                    numel = self.compute_local_numel(info.global_shape, r, ws)
                    shape = self.compute_local_shape(info.global_shape, r, ws)
                    if numel > 0:
                        offset = per_rank_param_offsets[r][i]
                        per_rank_shards.append(
                            gathered[r][offset : offset + numel].view(shape)
                        )
                    else:
                        per_rank_shards.append(
                            torch.empty(shape, dtype=info.dtype, device=device)
                        )
                full_params.append(
                    self.assemble_from_shards(
                        per_rank_shards, info.global_shape, info.dtype
                    )
                )
                del per_rank_shards

        return PlacementUnshardResult(full_params, [send_buf, *gathered])

    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        """Pack full gradients for the placement's default reduce-scatter path."""
        ws = mesh.size()
        rank = mesh.get_local_rank()
        pg = mesh.get_group()
        with _record_function_if_eager("FlexShard::reduce_scatter_copy_in", debug_fqn):
            send_buf, layout = self.pack_reduce_grad(tensors, infos, ws)
            if send_buf.numel() % ws != 0:
                raise AssertionError(
                    f"Packed reduce-scatter buffer has {send_buf.numel()} elements, "
                    f"which is not divisible by world size {ws}."
                )
        return PlacementPreparedReduceGrad(
            placement=self,
            infos=infos,
            layout=layout,
            rank=rank,
            world_size=ws,
            pg=pg,
            buffers=[send_buf],
            debug_fqn=debug_fqn,
        )

    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
    ) -> PlacementReduceGradResult:
        """Reduce a prepared gradient request using reduce-scatter."""
        send_buf = prepared.buffers[0]
        recv_buf = torch.empty(
            send_buf.numel() // prepared.world_size,
            dtype=send_buf.dtype,
            device=send_buf.device,
        )
        with _record_comm_if_eager("FlexShard::reduce_scatter", prepared.debug_fqn):
            # TODO: Plumb the reduction/scaling policy from SPMD gradient semantics.
            # AVG is a convenient default, but delayed grad scaling may need SUM
            # plus an explicit scale at a different point in the training step.
            dist.reduce_scatter_tensor(
                output=recv_buf,
                input=send_buf,
                op=dist.ReduceOp.AVG,
                group=prepared.pg,
            )
        with _record_function_if_eager(
            "FlexShard::reduce_scatter_copy_out",
            prepared.debug_fqn,
        ):
            sharded_grads = self.unpack_reduced_grad(
                recv_buf,
                prepared.infos,
                prepared.layout,
                prepared.rank,
                prepared.world_size,
            )
        return PlacementReduceGradResult(sharded_grads, [recv_buf])


__all__ = [
    "LocalStorageLayout",
    "Placement",
    "PlacementPreparedReduceGrad",
    "PlacementReduceGradResult",
    "PlacementUnshardResult",
]
