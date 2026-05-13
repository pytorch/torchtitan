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

from ..flex_shard.placement_contract import Placement

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import ParamInfo


@dataclass(frozen=True)
class _ShardReduceGradLayout:
    padded_sizes: list[torch.Size]


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
    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        chunks = list(torch.chunk(param, world_size, dim=self.dim))
        empty_shape = list(param.shape)
        empty_shape[self.dim] = 0
        while len(chunks) < world_size:
            chunks.append(param.new_empty(empty_shape))
        return chunks[rank].contiguous()

    @override
    def assemble_from_shards(
        self,
        per_rank_shards: list[torch.Tensor],
        global_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not per_rank_shards:
            raise AssertionError("Expected at least one shard to assemble.")
        return torch.cat(per_rank_shards, dim=self.dim)

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
    "per_param_placements",
    "Shard",
]
