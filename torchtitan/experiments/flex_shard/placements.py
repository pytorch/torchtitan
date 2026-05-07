# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for
from typing_extensions import override

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
    assemble_from_shards), storage layout, and gradient-reduction packing.
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


@dataclass(frozen=True)
class _ShardReduceGradLayout:
    padded_sizes: list[torch.Size]


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
    def pack_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        world_size: int,
    ) -> tuple[torch.Tensor, _ShardReduceGradLayout]:
        dtype = tensors[0].dtype
        device = tensors[0].device
        padded_sizes: list[torch.Size] = []
        for tensor in tensors:
            padded_size = list(tensor.shape)
            padded_size[self.dim] = (
                (tensor.size(self.dim) + world_size - 1) // world_size
            ) * world_size
            padded_sizes.append(torch.Size(padded_size))

        input_numel = sum(s.numel() for s in padded_sizes)
        send_buf = torch.empty(input_numel, dtype=dtype, device=device)
        send_buf_2d = send_buf.view(world_size, -1)
        torch._chunk_cat(tensors, dim=self.dim, num_chunks=world_size, out=send_buf_2d)
        return send_buf, _ShardReduceGradLayout(padded_sizes)

    @override
    def unpack_reduced_grad(
        self,
        recv_buf: torch.Tensor,
        infos: list[ParamInfo],
        layout: Any,
        rank: int,
        world_size: int,
    ) -> list[torch.Tensor]:
        if not isinstance(layout, _ShardReduceGradLayout):
            raise AssertionError(
                f"Expected _ShardReduceGradLayout, got {type(layout).__name__}"
            )
        results: list[torch.Tensor] = []
        flat_offset = 0
        for info, padded_size in zip(infos, layout.padded_sizes, strict=True):
            local_shape = info.placement.compute_local_shape(
                info.global_shape, rank, world_size
            )
            stride = make_contiguous_strides_for(local_shape)
            shard = torch.as_strided(
                recv_buf,
                size=local_shape,
                stride=stride,
                storage_offset=flat_offset,
            ).contiguous()
            results.append(shard)
            flat_offset += padded_size.numel() // world_size
        return results


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
