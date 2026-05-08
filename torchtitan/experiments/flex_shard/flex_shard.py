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
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for
from torch.autograd import Variable
from torch.profiler import record_function
from torch.utils._pytree import tree_flatten


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


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
        ws = mesh.size()
        pg = mesh.get_group()
        dtype = tensors[0].dtype
        device = tensors[0].device

        # Pack local shards into one flat buffer
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

        # Variable-size all_gather
        gathered = [
            torch.empty(per_rank_sizes[r], dtype=dtype, device=device)
            for r in range(ws)
        ]
        dist.all_gather(gathered, send_buf, group=pg)

        # Unpack: per param, extract shard from each rank, assemble_from_shards
        results: list[torch.Tensor] = []
        for i, info in enumerate(infos):
            p = info.placements[0]
            per_rank_shards: list[torch.Tensor] = []
            for r in range(ws):
                numel = p.compute_local_numel(info.global_shape, r, ws)
                shape = p.compute_local_shape(info.global_shape, r, ws)
                if numel > 0:
                    off = per_rank_param_offsets[r][i]
                    per_rank_shards.append(gathered[r][off : off + numel].view(shape))
                else:
                    per_rank_shards.append(
                        torch.empty(shape, dtype=dtype, device=device)
                    )
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
        dtype = tensors[0].dtype
        device = tensors[0].device

        padded_sizes = []
        for tensor in tensors:
            padded_dim0 = ((tensor.size(0) + ws - 1) // ws) * ws
            padded_sizes.append(torch.Size([padded_dim0] + list(tensor.shape[1:])))

        input_numel = sum(s.numel() for s in padded_sizes)
        output_numel = input_numel // ws

        send_buf = torch.empty(input_numel, dtype=dtype, device=device)
        send_buf_2d = send_buf.view(ws, -1)
        torch._chunk_cat(tensors, dim=0, num_chunks=ws, out=send_buf_2d)

        recv_buf = torch.empty(output_numel, dtype=dtype, device=device)

        dist.reduce_scatter_tensor(
            output=recv_buf,
            input=send_buf,
            op=dist.ReduceOp.AVG,
            group=pg,
        )

        # Unpack per param from recv_buf
        results: list[torch.Tensor] = []
        flat_offset = 0
        for info, padded_size in zip(infos, padded_sizes):
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
    "flat_shard_placements",
    "FlatShard",
    "FlatShardParametrization",
    "flex_shard",
    "FlexShardModule",
    "get_global_shape",
    "get_placements",
    "is_flex_shard_param",
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


def _validate_placements_for_tracing(
    param_placements: dict[str, tuple[Placement, ...]],
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> None:
    """Validate that placements are compatible with FX tracing (Phase 1).

    Raises ValueError if any placement uses features not yet supported by
    the traceable parametrization classes (Owned, RaggedShard, uneven splits,
    or multi-dimensional meshes).
    """
    if mesh.ndim != 1:
        raise ValueError(
            f"Traceable FlexShard requires a 1D mesh, got {mesh.ndim}D. "
            "Multi-dimensional mesh support is planned for Phase 5+."
        )

    param_dict = dict(named_params)
    world_size = mesh.size()

    for fqn, placements in param_placements.items():
        for placement in placements:
            if isinstance(placement, Owned):
                raise ValueError(
                    f"Parameter '{fqn}' uses Owned placement, which is not "
                    "supported for tracing. Owned placement requires "
                    "rank-dependent control flow that cannot be captured in "
                    "an FX graph. Support is planned for Phase 5+."
                )
            if isinstance(placement, RaggedShard):
                raise ValueError(
                    f"Parameter '{fqn}' uses RaggedShard placement, which is "
                    "not supported for tracing. RaggedShard requires "
                    "variable-size collectives that cannot be traced. "
                    "Support is planned for Phase 5+."
                )
            if isinstance(placement, Shard):
                param = param_dict[fqn]
                dim_size = param.shape[placement.dim]
                if dim_size % world_size != 0:
                    raise ValueError(
                        f"Parameter '{fqn}' has shape {tuple(param.shape)} with "
                        f"dim {placement.dim} size {dim_size}, which is not "
                        f"evenly divisible by world_size={world_size}. "
                        "Use FlatShard or pad the parameter."
                    )
            if isinstance(placement, FlatShard):
                param = param_dict[fqn]
                if param.numel() % world_size != 0:
                    raise ValueError(
                        f"Parameter '{fqn}' has {param.numel()} elements, "
                        f"which is not evenly divisible by "
                        f"world_size={world_size}. Pad the parameter."
                    )


# ---------------------------------------------------------------------------
# Phase 2b: Bucket assignment and validation
# ---------------------------------------------------------------------------


def _assign_params_to_buckets(
    param_fqns: list[str],
    buckets: list[list[str] | BucketSpec],
) -> list[list[str]]:
    """Assign each param FQN to exactly one bucket via fnmatch.

    Returns:
        List of lists: assignments[i] = [fqn, ...] for bucket i.

    Raises:
        ValueError: if any param matches zero or multiple buckets.
    """
    param_to_buckets: dict[str, list[int]] = {fqn: [] for fqn in param_fqns}
    for bucket_idx, bucket in enumerate(buckets):
        patterns = _get_bucket_patterns(bucket)
        for fqn in param_fqns:
            for pattern in patterns:
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
            bucket_descs = ", ".join(
                f"bucket {i} {_get_bucket_patterns(buckets[i])}" for i in idxs
            )
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
    buckets: list[list[str] | BucketSpec],
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
                    f"{_get_bucket_patterns(buckets[bucket_idx])} "
                    f"has mixed placement types: '{fqns[0]}' uses "
                    f"{type(reference_placement).__name__} but '{fqn}' uses "
                    f"{type(placement).__name__}. "
                    "All params in a bucket must share the same placement type."
                )
            if isinstance(placement, Shard) and isinstance(reference_placement, Shard):
                if placement.dim != reference_placement.dim:
                    raise ValueError(
                        f"Bucket {bucket_idx} "
                        f"{_get_bucket_patterns(buckets[bucket_idx])} "
                        f"has mixed shard dimensions: '{fqns[0]}' uses "
                        f"Shard({reference_placement.dim}) but '{fqn}' uses "
                        f"Shard({placement.dim}). "
                        "All Shard params in a bucket must share the same "
                        "dimension."
                    )


# ---------------------------------------------------------------------------
# Phase 2a: Parametrization guard and property-based registration
# ---------------------------------------------------------------------------

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
    param_name_to_property = {
        param_name: property(
            lambda self, pn=param_name, p=param: p(self._parameters[pn])
        )
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


@dataclass
class BucketSpec:
    """Specification for a parameter communication bucket.

    Args:
        patterns: fnmatch glob patterns matched against parameter FQNs.
            A parameter matches this bucket if its FQN matches any pattern.
        mp_policy: Mixed precision policy for this bucket (Phase 2c).
        offload_policy: CPU offload policy for this bucket (Phase 2c).
    """

    patterns: list[str]
    mp_policy: Any = None
    offload_policy: Any = None


def _get_bucket_patterns(bucket: list | BucketSpec) -> list[str]:
    """Extract patterns from a bucket (list[str] or BucketSpec)."""
    if isinstance(bucket, BucketSpec):
        return bucket.patterns
    return bucket


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
        for tensor, info in zip(tensors, infos):
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
        for tensor, info in zip(tensors, infos):
            p = info.placements[0]
            send = tensor.contiguous()
            dist.reduce(send, dst=p.owner_rank, op=dist.ReduceOp.AVG, group=pg)
            if rank == p.owner_rank:
                results.append(send)
            else:
                results.append(tensor.new_empty(0))
        return results


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
        for tensor, info in zip(tensors, infos):
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
        for tensor, info in zip(tensors, infos):
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
    """

    def __init__(self, shard_dim: int, group_name: str, world_size: int):
        super().__init__()
        self.shard_dim = shard_dim
        self.group_name = group_name
        self.world_size = world_size

    def forward(self, local_shard: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return local_shard
        full = torch.ops._c10d_functional.all_gather_into_tensor(
            local_shard, self.world_size, self.group_name
        )
        full = torch.ops._c10d_functional.wait_tensor(full)
        if self.shard_dim != 0:
            chunks = full.chunk(self.world_size, dim=0)
            full = torch.cat(chunks, dim=self.shard_dim)
        return full


class FlatShardParametrization(nn.Module):
    """FX-traceable parametrization for FlatShard placement.

    Reconstructs the full parameter from a flat 1D shard using
    _c10d_functional ops. The flat shard has shape (numel // world_size,).
    After all-gather, the result is reshaped to the original parameter shape.
    """

    def __init__(self, group_name: str, world_size: int, original_shape: torch.Size):
        super().__init__()
        self.group_name = group_name
        self.world_size = world_size
        self.original_shape = original_shape

    def forward(self, flat_shard: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return flat_shard
        full_flat = torch.ops._c10d_functional.all_gather_into_tensor(
            flat_shard, self.world_size, self.group_name
        )
        full_flat = torch.ops._c10d_functional.wait_tensor(full_flat)
        return full_flat.view(self.original_shape)


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
    Placement.reduce_grad(), which handle batching per placement type.

    Lifecycle (automatic with hooks):
        1. SHARDED state: Parameters are sharded tensors with placement metadata
        2. Forward pre-hook: unshard() - Placement.unshard() to gather full params
        3. Forward: compute with unsharded params
        4. Forward post-hook: register backward hooks
        5. Backward: compute gradients with unsharded params
        6. Post-backward: reduce_grad() - Placement.reduce_grad() to reduce gradients,
           then restore sharded parameters
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
        register_hooks: bool = True,
        module_fqn: str = "",
    ) -> None:
        if byte_storage.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 storage, got {byte_storage.dtype}")
        self._byte_storage = byte_storage
        self._param_infos = param_infos
        self._mesh = mesh
        self._total_bytes = total_bytes
        self._total_unsharded_bytes = total_unsharded_bytes
        self._module = module
        self._module_fqn = module_fqn or type(module).__name__
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

        # Hook handles
        self._pre_forward_hook_handle: RemovableHandle | None = None
        self._post_forward_hook_handle: RemovableHandle | None = None

        # Track if post_backward has been called this iteration
        self._post_backward_called = False

        # Register forward hooks if requested
        if register_hooks:
            self._register_forward_hooks()

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

        with record_function(f"unshard({self._module_fqn})"):
            full_params = ptype.unshard(local_shards, infos, self._mesh)

        for info, full_param in zip(infos, full_params):
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

            with record_function(f"reduce_grad({self._module_fqn})"):
                sharded_grads = ptype.reduce_grad(grads, infos_with_grads, self._mesh)

            for info, sharded_grad in zip(infos_with_grads, sharded_grads):
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

    # ==================== Hook-based Scheduling ====================

    def _register_forward_hooks(self) -> None:
        """Register forward pre/post hooks on the module."""
        self._pre_forward_hook_handle = self._module.register_forward_pre_hook(
            self._pre_forward, prepend=True, with_kwargs=True
        )
        self._post_forward_hook_handle = self._module.register_forward_hook(
            self._post_forward, prepend=False
        )

    def _pre_forward(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Forward pre-hook: unshard parameters."""
        self.unshard()
        return args, kwargs

    def _post_forward(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        output: Any,
    ) -> Any:
        """Forward post-hook: register backward hooks."""
        # Register backward hooks on output tensors
        output = self._register_pre_backward_hooks(output)

        # Reset post_backward flag for this iteration
        self._post_backward_called = False

        # NOTE: We do NOT reshard after forward even if reshard_after_forward=True
        # This is because the autograd graph references the unsharded params,
        # and we need those same param objects to receive gradients in backward.
        # Memory savings from reshard_after_forward would require more complex
        # tracking of unsharded params (like FSDP2's FSDPParam).

        return output

    def _register_pre_backward_hooks(self, output: Any) -> Any:
        """Register hooks on output tensors to trigger pre_backward."""
        if not torch.is_grad_enabled():
            return output

        flat_outputs, _ = tree_flatten(output)
        for t in flat_outputs:
            if torch.is_tensor(t) and t.requires_grad:
                t.register_hook(self._pre_backward)

        return output

    def _pre_backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward pre-hook: register post-backward callback."""
        # Register post-backward callback (must be done during backward)
        self._register_post_backward_callback()
        # Params are already unsharded from forward, no need to unshard again
        return grad

    def _register_post_backward_callback(self) -> None:
        """Register callback to run after backward completes."""
        if self._post_backward_called:
            return
        Variable._execution_engine.queue_callback(self._post_backward)

    def _post_backward(self) -> None:
        """Post-backward callback: reduce gradients and restore sharded params."""
        # Ensure we only run once per backward pass
        if self._post_backward_called:
            return
        self._post_backward_called = True

        # Force children to reduce first (handles LIFO callback ordering)
        for name, child in self._module.named_modules():
            if name and hasattr(child, _DSTORAGE_ATTR):
                child_ds = getattr(child, _DSTORAGE_ATTR)
                if child_ds.state == ShardedState.UNSHARDED:
                    child_ds._post_backward()

        # Only reduce if currently unsharded
        if self._state == ShardedState.UNSHARDED:
            self.reduce_grad()

    def _reshard_params_only(self) -> None:
        """
        Reshard parameters without reduce-scatter (for use after forward).

        This restores sharded parameters but does NOT reduce-scatter
        gradients (since there are none yet after forward).
        """
        if self._state == ShardedState.SHARDED:
            return

        # Restore sharded parameters
        for fqn, sharded_param in self._sharded_params.items():
            _set_param_on_module(self._module, fqn, sharded_param)

        # Free unsharded buffer
        if self._unsharded_byte_storage is not None:
            self._unsharded_byte_storage = None

        self._state = ShardedState.SHARDED

    def remove_hooks(self) -> None:
        """Remove registered forward hooks."""
        if self._pre_forward_hook_handle is not None:
            self._pre_forward_hook_handle.remove()
            self._pre_forward_hook_handle = None
        if self._post_forward_hook_handle is not None:
            self._post_forward_hook_handle.remove()
            self._post_forward_hook_handle = None


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


def auto_buckets(module: nn.Module) -> list[list[str]]:
    """Generate one bucket per direct child module.

    Returns a list of bucket patterns suitable for the ``buckets`` parameter
    of :func:`flex_shard`. Each bucket contains a single ``"child_name.*"``
    pattern matching all parameters under that child.

    Example::

        >>> buckets = auto_buckets(model)
        >>> flex_shard(model, mesh, buckets=buckets)
    """
    children = list(module.named_children())
    if not children:
        return [["*"]]
    return [[f"{name}.*"] for name, _ in children]


def _create_param_infos(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
    param_placements: dict[str, tuple[Placement, ...]],
) -> tuple[dict[str, ParamInfo], int, int]:
    """
    Create ParamInfo for each parameter, computing local shapes and byte offsets.

    Placement-agnostic: works with any placement type (Shard, FlatShard, Owned, etc.).
    Parameters are laid out sequentially in the byte buffer with proper alignment.

    Args:
        named_params: List of (fqn, param) tuples
        mesh: Device mesh for sharding
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
        global_shape = param.shape
        global_stride = make_contiguous_strides_for(global_shape)
        local_shape, local_numel = _compute_local_info(global_shape, mesh, placements)
        dtype = param.dtype
        global_numel = param.numel()

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
        )
        param_infos[fqn] = info

    return param_infos, current_byte_offset, current_unsharded_byte_offset


def _create_sharded_view(
    local_view: torch.Tensor,
    info: ParamInfo,
    mesh: DeviceMesh,
) -> torch.Tensor:
    """Annotate a local tensor view with placement metadata."""
    set_sharding_info(
        local_view,
        placements=info.placements,
        global_shape=info.global_shape,
        global_stride=info.global_stride,
        mesh=mesh,
    )
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
    mesh: DeviceMesh,
) -> None:
    """Pack original parameter data into byte storage.

    Calls placement.extract_local_shard() to get each rank's typed local shard,
    then copies it as uint8 into the byte buffer.
    """
    my_rank = mesh.get_local_rank()
    world_size = mesh.size()

    for fqn, param in named_params:
        info = param_infos[fqn]
        if param.device.type == "meta":
            continue
        shard = info.placements[0].extract_local_shard(param.data, my_rank, world_size)
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


def _resolve_placement_fn(
    shard_placement_fn: PlacementFn | dict[str, Placement] | None,
) -> PlacementFn:
    """Normalize shard_placement_fn to a callable.

    Accepts:
    - None: defaults to per_param_placements (Shard(0) per param)
    - dict[str, Placement]: fnmatch patterns, first match wins
    - Callable: used directly
    """
    if shard_placement_fn is None:
        return per_param_placements
    if isinstance(shard_placement_fn, dict):
        pattern_map = shard_placement_fn

        def fn(
            named_params: list[tuple[str, nn.Parameter]],
            mesh: DeviceMesh,
        ) -> dict[str, tuple[Placement, ...]]:
            result: dict[str, tuple[Placement, ...]] = {}
            for fqn, _ in named_params:
                for pattern, placement in pattern_map.items():
                    if fnmatch.fnmatch(fqn, pattern):
                        result[fqn] = (placement,)
                        break
                else:
                    raise ValueError(
                        f"Parameter '{fqn}' does not match any placement pattern. "
                        f'Add a catch-all pattern: {{"*": Shard(0)}}'
                    )
            return result

        return fn
    return shard_placement_fn


def flex_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    shard_placement_fn: PlacementFn | dict[str, Placement] | None = None,
    *,
    buckets: list[list[str] | BucketSpec] | None = None,
    reshard_after_forward: bool = True,
    register_hooks: bool = False,
    module_fqn: str = "",
) -> FlexShardModule:
    """
    Apply flat-storage FSDP sharding to a module.

    This function:
    1. Collects parameters from the module (excluding already-wrapped submodules)
    2. Groups parameters into communication buckets (one per bucket, or all in one)
    3. Creates a unified byte buffer per bucket for all its parameters
    4. Replaces each parameter with a plain tensor annotated with placement metadata
    5. Optionally registers forward/backward hooks for automatic unshard/reduce_grad
    6. Stores DStorages on the module (accessible via module.dstorages)

    Each bucket gets its own byte buffer and DStorage, enabling independent
    all-gather operations per bucket.

    Nested wrapping is supported: apply flex_shard to inner modules first,
    then to outer modules. The outer module's storage will exclude parameters
    from already-wrapped inner modules.

    Args:
        module: The module to shard. Can have real or meta device parameters.
        mesh: The device mesh for sharding. Currently only 1D mesh is supported.
        shard_placement_fn: Determines per-parameter placements. Accepts:
            - None (default): uses per_param_placements (Shard(0) per param)
            - dict[str, Placement]: fnmatch patterns, first match wins
            - Callable: (named_params, mesh) -> dict of per-param placements
            Built-in callables: per_param_placements, flat_shard_placements,
            param_boundary_placements.
        buckets: Optional list of bucket specifications. Each bucket is either
            a list of fnmatch patterns or a BucketSpec. When None (default),
            all parameters go into a single bucket. Use auto_buckets() to
            generate one bucket per direct child module.
        reshard_after_forward: If True (default), reshard parameters after forward
            to save memory. Parameters will be re-unsharded in backward.
            If False, keep parameters unsharded between forward and backward.
        register_hooks: If True, register forward/backward hooks for
            automatic unshard/reduce_grad. If False (default), uses
            property-based parametrization instead.
        module_fqn: Fully qualified name prefix for profiling labels.

    Returns:
        The module (mutated in-place). Use module.dstorages to access internals.

    Example::

        >>> mesh = init_device_mesh("cuda", (world_size,))
        >>> model = Transformer(args)
        >>> # Single bucket (default):
        >>> flex_shard(model, mesh)
        >>> # Explicit buckets:
        >>> flex_shard(model, mesh, buckets=[["attn.*"], ["ffn.*"]])
        >>> # Auto buckets (one per child):
        >>> flex_shard(model, mesh, buckets=auto_buckets(model))

    Note:
        - Parameters of different dtypes are supported in a single unified buffer
        - Proper alignment is maintained for each dtype
        - Parameters on meta device will have uninitialized storage
        - Each bucket must have consistent placement types
    """
    resolved_fn = _resolve_placement_fn(shard_placement_fn)

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

    # Determine device - use param device if meta, otherwise use mesh device
    first_param = named_params[0][1]
    if first_param.device.type == "meta":
        device = torch.device("meta")
    else:
        device = mesh.device_type
        if device == "cuda":
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device(device)

    # Resolve placements for all params
    param_placements = resolved_fn(named_params, mesh)
    _validate_placements_for_tracing(param_placements, named_params, mesh)

    # Resolve buckets
    if buckets is None:
        buckets_resolved: list[list[str] | BucketSpec] = [["*"]]
    else:
        buckets_resolved = buckets

    param_fqns = [fqn for fqn, _ in named_params]
    bucket_assignments = _assign_params_to_buckets(param_fqns, buckets_resolved)
    _validate_bucket_placements(bucket_assignments, param_placements, buckets_resolved)

    # Log bucket coverage
    if logger.isEnabledFor(logging.DEBUG):
        lines = ["flex_shard bucket coverage:"]
        total_params = 0
        for i, fqns in enumerate(bucket_assignments):
            patterns = _get_bucket_patterns(buckets_resolved[i])
            lines.append(f"  bucket {i} {patterns}: {len(fqns)} params")
            total_params += len(fqns)
        lines.append(
            f"  total: {total_params} params across " f"{len(buckets_resolved)} buckets"
        )
        logger.debug("\n".join(lines))

    # Per-bucket: create param_infos, byte buffer, replace params, create DStorage
    named_params_dict = dict(named_params)
    storages: list[DStorage] = []

    for bucket_idx, bucket_fqns in enumerate(bucket_assignments):
        if not bucket_fqns:
            continue

        bucket_named_params = [(fqn, named_params_dict[fqn]) for fqn in bucket_fqns]
        bucket_placements = {fqn: param_placements[fqn] for fqn in bucket_fqns}

        param_infos, total_bytes, total_unsharded_bytes = _create_param_infos(
            bucket_named_params, mesh, bucket_placements
        )

        byte_storage = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        _write_params_to_dstorage(byte_storage, bucket_named_params, param_infos, mesh)

        for fqn, info in param_infos.items():
            local_view = byte_storage[
                info.byte_offset : info.byte_offset
                + info.local_numel * info.dtype.itemsize
            ]
            typed_view = local_view.view(info.dtype).view(info.local_shape)
            new_param = nn.Parameter(typed_view, requires_grad=info.requires_grad)
            _create_sharded_view(new_param, info, mesh)
            _set_param_on_module(module, fqn, new_param)

        bucket_fqn = (
            f"{module_fqn}_bucket{bucket_idx}" if module_fqn else f"bucket{bucket_idx}"
        )
        storage = DStorage(
            byte_storage,
            param_infos,
            mesh,
            total_bytes,
            total_unsharded_bytes,
            module,
            reshard_after_forward=reshard_after_forward,
            register_hooks=register_hooks,
            module_fqn=bucket_fqn,
        )
        storages.append(storage)

    # Store DStorages on module
    setattr(module, _DSTORAGES_ATTR, storages)
    setattr(module, _DSTORAGE_ATTR, storages[0] if storages else None)

    # Change module class to include FlexShardModule mixin
    cls = type(module)
    if not issubclass(cls, FlexShardModule):
        module.__class__ = type(cls.__name__, (cls, FlexShardModule), {})

    # Register property-based parametrization (Phase 2a)
    if not register_hooks:
        group_name = mesh.get_group().group_name
        world_size = mesh.size()

        # Group parametrizations by leaf module (across all buckets)
        module_param_map: dict[nn.Module, dict[str, nn.Module]] = {}

        for s in storages:
            for fqn, info in s._param_infos.items():
                placement = info.placements[0]
                if isinstance(placement, Shard):
                    p = ShardParametrization(
                        shard_dim=placement.dim,
                        group_name=group_name,
                        world_size=world_size,
                    )
                elif isinstance(placement, FlatShard):
                    p = FlatShardParametrization(
                        group_name=group_name,
                        world_size=world_size,
                        original_shape=info.global_shape,
                    )
                else:
                    raise ValueError(
                        f"Unsupported placement for parametrization: " f"{placement}"
                    )

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

    return module
