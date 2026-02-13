# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, Literal, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.hooks import RemovableHandle

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for
from torch.autograd import Variable
from torch.utils._pytree import tree_flatten


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


class Placement:
    """Base class for FlexShard placement strategies.

    DStorage issues one all_gather for the entire byte buffer, then calls
    unpack_unshard per param to reconstruct the full tensor. For gradient
    reduction, DStorage groups params by placement type and calls comm_reduce.
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

    # Shard: extract local shard from full param and pack into byte buffer
    def pack_unshard(
        self,
        param: torch.Tensor,
        buffer: torch.Tensor,
        byte_offset: int,
        rank: int,
        world_size: int,
    ) -> None:
        """Extract this rank's local shard from full param and copy into buffer.

        Called during flex_shard() to populate byte_storage from original params.

        Args:
            param: the full (unsharded) parameter tensor
            buffer: the byte storage buffer
            byte_offset: where to write in the buffer
            rank: this rank's index
            world_size: total number of ranks
        """
        raise NotImplementedError

    # Unshard: extract full param from gathered per-rank buffers
    def unpack_unshard(
        self,
        gathered_buffers: list[torch.Tensor],
        per_rank_byte_offsets: list[int],
        global_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Extract unsharded param from gathered buffers.

        Args:
            gathered_buffers: list of per-rank byte buffers from all_gather
            per_rank_byte_offsets: byte offset of this param in each rank's buffer
            global_shape: the full unsharded shape
            dtype: parameter dtype
        """
        raise NotImplementedError

    # Reduce: full grad → local reduced grad
    def comm_reduce(
        self, send_buf: torch.Tensor, recv_buf: torch.Tensor, mesh: DeviceMesh
    ) -> None:
        raise NotImplementedError

    def unpack_reduce(
        self,
        buffer: torch.Tensor,
        offset: int,
        local_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
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

    def pack_unshard(
        self,
        param: torch.Tensor,
        buffer: torch.Tensor,
        byte_offset: int,
        rank: int,
        world_size: int,
    ) -> None:
        chunks = list(torch.chunk(param, world_size, dim=self.dim))
        while len(chunks) < world_size:
            chunks.append(chunks[0].new_empty(0))
        local_data = chunks[rank]
        if local_data.numel() > 0:
            nbytes = local_data.numel() * local_data.element_size()
            dest = buffer[byte_offset : byte_offset + nbytes].view(local_data.dtype)
            dest.copy_(local_data.reshape(-1))

    def unpack_unshard(
        self,
        gathered_buffers: list[torch.Tensor],
        per_rank_byte_offsets: list[int],
        global_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ws = len(gathered_buffers)
        chunks = []
        for rank in range(ws):
            local_shape = self.compute_local_shape(global_shape, rank, ws)
            local_numel = 1
            for d in local_shape:
                local_numel *= d
            if local_numel > 0:
                nbytes = local_numel * dtype.itemsize
                offset = per_rank_byte_offsets[rank]
                chunk = gathered_buffers[rank][offset : offset + nbytes]
                chunks.append(chunk.view(dtype).view(local_shape))
        if chunks:
            return torch.cat(chunks, dim=self.dim)
        return torch.empty(global_shape, dtype=dtype, device=gathered_buffers[0].device)

    def comm_reduce(
        self, send_buf: torch.Tensor, recv_buf: torch.Tensor, mesh: DeviceMesh
    ) -> None:
        dist.reduce_scatter_tensor(
            output=recv_buf,
            input=send_buf,
            op=dist.ReduceOp.AVG,
            group=mesh.get_group(),
        )

    def unpack_reduce(
        self,
        buffer: torch.Tensor,
        offset: int,
        local_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        stride = make_contiguous_strides_for(local_shape)
        return torch.as_strided(
            buffer, size=local_shape, stride=stride, storage_offset=offset
        ).contiguous()


@dataclass
class FlatShard(Placement):
    """Flat-concat sharding — contiguous 1D slice of the flattened parameter."""

    flat_offset: int = 0
    numel: int = 0


__all__ = [
    "DStorage",
    "FlatShard",
    "flex_shard",
    "FlexShardModule",
    "get_global_shape",
    "get_placements",
    "is_flex_shard_param",
    "Owned",
    "Placement",
    "RaggedShard",
    "set_sharding_info",
    "Shard",
]


# Module attribute name for storing DStorage
_DSTORAGE_ATTR = "_dstorage"

# Hidden attribute names for placement metadata on plain tensors
_PLACEMENTS_ATTR = "_placements"
_GLOBAL_SHAPE_ATTR = "_global_shape"
_GLOBAL_STRIDE_ATTR = "_global_stride"
_MESH_ATTR = "_mesh"


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
    """Mixin added to modules after flex_shard(). Provides sharding methods directly on the module."""

    def unshard(self) -> None:
        getattr(self, _DSTORAGE_ATTR).unshard()

    def shard(self) -> None:
        getattr(self, _DSTORAGE_ATTR).reshard()

    @property
    def dstorage(self) -> DStorage:
        return getattr(self, _DSTORAGE_ATTR)


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

    def pack_unshard(
        self,
        param: torch.Tensor,
        buffer: torch.Tensor,
        byte_offset: int,
        rank: int,
        world_size: int,
    ) -> None:
        if rank == self.owner_rank:
            nbytes = param.numel() * param.element_size()
            dest = buffer[byte_offset : byte_offset + nbytes].view(param.dtype)
            dest.copy_(param.reshape(-1))

    def unpack_unshard(
        self,
        gathered_buffers: list[torch.Tensor],
        per_rank_byte_offsets: list[int],
        global_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        global_numel = 1
        for d in global_shape:
            global_numel *= d
        nbytes = global_numel * dtype.itemsize
        offset = per_rank_byte_offsets[self.owner_rank]
        return (
            gathered_buffers[self.owner_rank][offset : offset + nbytes]
            .view(dtype)
            .view(global_shape)
        )

    def comm_reduce(
        self, send_buf: torch.Tensor, recv_buf: torch.Tensor, mesh: DeviceMesh
    ) -> None:
        # Per-param reduce to owner
        dist.reduce(
            send_buf,
            dst=self.owner_rank,
            op=dist.ReduceOp.AVG,
            group=mesh.get_group(),
        )

    def unpack_reduce(
        self,
        buffer: torch.Tensor,
        offset: int,
        local_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return buffer  # Owner keeps the reduced grad as-is


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

    def __init__(
        self, dims: tuple[int, ...] = (0,), local_units: tuple[int, ...] = ()
    ):
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
                chunk = (remaining * self.local_units[r] + remaining_units - 1) // remaining_units
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

    def pack_unshard(
        self,
        param: torch.Tensor,
        buffer: torch.Tensor,
        byte_offset: int,
        rank: int,
        world_size: int,
    ) -> None:
        dim_size = param.shape[self.dim]
        splits = self._compute_dim_splits(dim_size)
        start = sum(splits[:rank])
        local_data = param.narrow(self.dim, start, splits[rank])
        if local_data.numel() > 0:
            nbytes = local_data.numel() * local_data.element_size()
            dest = buffer[byte_offset : byte_offset + nbytes].view(local_data.dtype)
            dest.copy_(local_data.reshape(-1))

    def unpack_unshard(
        self,
        gathered_buffers: list[torch.Tensor],
        per_rank_byte_offsets: list[int],
        global_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ws = len(gathered_buffers)
        dim_size = global_shape[self.dim]
        splits = self._compute_dim_splits(dim_size)
        chunks = []
        for rank in range(ws):
            local_shape = list(global_shape)
            local_shape[self.dim] = splits[rank]
            local_numel = 1
            for d in local_shape:
                local_numel *= d
            if local_numel > 0:
                nbytes = local_numel * dtype.itemsize
                offset = per_rank_byte_offsets[rank]
                chunk = gathered_buffers[rank][offset : offset + nbytes]
                chunks.append(chunk.view(dtype).view(local_shape))
        if chunks:
            return torch.cat(chunks, dim=self.dim)
        return torch.empty(global_shape, dtype=dtype, device=gathered_buffers[0].device)

    def comm_reduce(
        self, send_buf: torch.Tensor, recv_buf: torch.Tensor, mesh: DeviceMesh
    ) -> None:
        # send_buf is the full unsharded grad.
        # Split along self.dim into per-rank chunks based on local_units,
        # then reduce_scatter so each rank gets its reduced chunk.
        pg = mesh.get_group()
        dim_size = send_buf.shape[self.dim]
        splits = self._compute_dim_splits(dim_size)
        # Split grad into variable-size chunks for each rank
        input_list = list(torch.split(send_buf, splits, dim=self.dim))
        # Flatten each chunk for reduce_scatter
        flat_inputs = [chunk.contiguous().view(-1) for chunk in input_list]
        flat_output = recv_buf.view(-1)
        dist.reduce_scatter(
            flat_output,
            flat_inputs,
            op=dist.ReduceOp.AVG,
            group=pg,
        )

    def unpack_reduce(
        self,
        buffer: torch.Tensor,
        offset: int,
        local_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return buffer.view(local_shape)


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
    owner_rank: int | None = (
        None  # for param-boundary sharding: which rank owns this param
    )
    padded_local_numel: int = (
        0  # padded size for uniform buffer layout (max across ranks)
    )


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

    This enables a single all-gather operation for the entire parameter group.

    Lifecycle (automatic with hooks):
        1. SHARDED state: Parameters are sharded tensors with placement metadata
        2. Forward pre-hook: unshard() - all-gather to get full params
        3. Forward: compute with unsharded params
        4. Forward post-hook: register backward hooks, optionally reshard
        5. Backward pre-hook: unshard() if resharded after forward
        6. Backward: compute gradients with unsharded params
        7. Post-backward: reshard() with reduce-scatter gradients
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
        region_info: dict[str, int] | None = None,
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

        # Region info for batched collectives (Shard → Ragged → Owned)
        self._region_info = region_info or {
            "shard_region_start": 0,
            "shard_region_end": total_bytes,
            "ragged_region_start": total_bytes,
            "ragged_region_end": total_bytes,
            "owned_region_start": total_bytes,
            "owned_region_end": total_bytes,
        }

        # Unsharded buffer (allocated on demand)
        self._unsharded_byte_storage: torch.Tensor | None = None

        # Cache sharded parameters for reshard
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

        # Cache for per-rank byte offsets (computed lazily in all_gather)
        self._cached_per_rank_offsets: (
            tuple[dict[str, list[int]], list[int]] | None
        ) = None

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

    def _compute_per_rank_byte_offsets(
        self,
    ) -> tuple[dict[str, list[int]], list[int]]:
        """Compute byte offsets for each param in each rank's buffer.

        Since placements and global shapes are known to all ranks, each rank
        can locally reconstruct any other rank's buffer layout without
        communication.

        Returns:
            offsets: dict mapping FQN to list of per-rank byte offsets
            sizes: list of total buffer size per rank
        """
        if self._cached_per_rank_offsets is not None:
            return self._cached_per_rank_offsets

        ws = self.world_size
        offsets: dict[str, list[int]] = {fqn: [0] * ws for fqn in self._param_infos}
        sizes = [0] * ws

        for r in range(ws):
            offset = 0
            for fqn, info in self._param_infos.items():
                placement = info.placements[0]
                alignment = _get_dtype_alignment(info.dtype)

                if isinstance(placement, Shard):
                    # Shard uses padded sizes — uniform across ranks
                    aligned = _align_offset(offset, alignment)
                    offsets[fqn][r] = aligned
                    offset = aligned + info.padded_local_numel * info.dtype.itemsize
                elif isinstance(placement, Owned):
                    local_numel = placement.compute_local_numel(
                        info.global_shape, r, ws
                    )
                    if local_numel > 0:
                        aligned = _align_offset(offset, alignment)
                        offsets[fqn][r] = aligned
                        offset = aligned + local_numel * info.dtype.itemsize
                else:
                    # RaggedShard or custom — actual local size per rank
                    local_numel = placement.compute_local_numel(
                        info.global_shape, r, ws
                    )
                    aligned = _align_offset(offset, alignment)
                    offsets[fqn][r] = aligned
                    offset = aligned + local_numel * info.dtype.itemsize

            sizes[r] = offset

        self._cached_per_rank_offsets = (offsets, sizes)
        return offsets, sizes

    def all_gather(self) -> torch.Tensor:
        """
        All-gather the sharded byte buffer to get the full unsharded buffer.

        Issues one dist.all_gather for the entire byte_storage, then calls
        placement.unpack_unshard() per param to reconstruct full tensors.

        Returns:
            Unsharded byte buffer containing all parameter data.
        """
        # Allocate unsharded buffer if needed
        if self._unsharded_byte_storage is None:
            self._unsharded_byte_storage = torch.empty(
                self._total_unsharded_bytes,
                dtype=torch.uint8,
                device=self._byte_storage.device,
            )

        pg = self._mesh.get_group()
        ws = self._mesh.size()

        # Compute per-rank buffer sizes and param offsets (no communication)
        per_rank_offsets, per_rank_sizes = self._compute_per_rank_byte_offsets()

        # One all_gather for the entire byte buffer
        gathered_buffers = [
            torch.empty(per_rank_sizes[r], dtype=torch.uint8, device=self._byte_storage.device)
            for r in range(ws)
        ]
        dist.all_gather(gathered_buffers, self._byte_storage.contiguous(), group=pg)

        # Unpack per param
        for fqn, info in self._param_infos.items():
            unsharded = info.placements[0].unpack_unshard(
                gathered_buffers,
                per_rank_offsets[fqn],
                info.global_shape,
                info.dtype,
            )

            # Copy to unsharded buffer
            num_bytes = info.global_numel * info.dtype.itemsize
            dest = self._unsharded_byte_storage[
                info.unsharded_byte_offset : info.unsharded_byte_offset + num_bytes
            ]
            dest.copy_(unsharded.view(-1).view(torch.uint8))

        return self._unsharded_byte_storage

    def unshard(self) -> None:
        """
        All-gather the byte buffer and register unsharded parameters on the module.

        After calling this, model.parameters() returns unsharded tensors for forward/backward.
        """
        if self._state == ShardedState.UNSHARDED:
            return  # Already unsharded

        # All-gather the byte buffer
        self.all_gather()

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
        Must be called while in UNSHARDED state, before reshard().
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
            info.placements[0].pack_unshard(
                unsharded_view, self._byte_storage, info.byte_offset, my_rank, world_size
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

    def reshard(self) -> None:
        """
        Reduce-scatter gradients, free unsharded buffer, and restore sharded parameters.

        Gradients from the unsharded parameters are reduce-scattered across ranks,
        and the resulting sharded gradients are stored on the sharded parameters.

        After calling this, model.parameters() returns sharded tensors with gradients.
        """
        if self._state == ShardedState.SHARDED:
            return  # Already sharded

        # Reduce-scatter gradients before swapping parameters
        self._reduce_scatter_grads()

        # Restore sharded parameters
        for fqn, sharded_param in self._sharded_params.items():
            _set_param_on_module(self._module, fqn, sharded_param)

        # Free unsharded buffer
        if self._unsharded_byte_storage is not None:
            self._unsharded_byte_storage = None

        self._state = ShardedState.SHARDED

    def _reduce_scatter_grads(self) -> None:
        """
        Reduce gradients to sharded params.

        Groups params by placement type. Each placement's comm_reduce handles
        the collective (reduce-scatter for Shard, reduce for Owned).
        """
        world_size = self.world_size
        my_rank = self._mesh.get_local_rank()

        # Group params with grads by placement type
        groups: dict[type, list[tuple[str, ParamInfo, torch.Tensor]]] = {}
        for fqn, info in self._param_infos.items():
            unsharded_param = _get_param_from_module(self._module, fqn)
            if unsharded_param.grad is None:
                continue
            grad = unsharded_param.grad.contiguous()
            ptype = type(info.placements[0])
            if ptype not in groups:
                groups[ptype] = []
            groups[ptype].append((fqn, info, grad))

        for ptype, param_list in groups.items():
            placement = param_list[0][1].placements[0]

            if isinstance(placement, Shard):
                # Batched reduce-scatter for Shard params
                grads = [g for _, _, g in param_list]
                infos = [info for _, info, _ in param_list]
                padded_sizes = []
                for grad in grads:
                    padded_dim0 = (
                        (grad.size(0) + world_size - 1) // world_size
                    ) * world_size
                    padded_sizes.append(
                        torch.Size([padded_dim0] + list(grad.shape[1:]))
                    )

                input_numel = sum(s.numel() for s in padded_sizes)
                output_numel = input_numel // world_size
                grad_dtype = grads[0].dtype
                device = grads[0].device

                send_buf = torch.empty(input_numel, dtype=grad_dtype, device=device)
                send_buf_2d = send_buf.view(world_size, -1)
                torch._chunk_cat(grads, dim=0, num_chunks=world_size, out=send_buf_2d)

                recv_buf = torch.empty(output_numel, dtype=grad_dtype, device=device)
                placement.comm_reduce(send_buf, recv_buf, self._mesh)

                # Unpack per param
                flat_offset = 0
                for info, padded_size in zip(infos, padded_sizes, strict=True):
                    padded_sharded_numel = padded_size.numel() // world_size
                    if info.local_numel > 0:
                        sharded_grad = placement.unpack_reduce(
                            recv_buf, flat_offset, info.local_shape, info.dtype
                        )
                        self._sharded_params[info.fqn].grad = sharded_grad
                    flat_offset += padded_sharded_numel
            elif isinstance(placement, RaggedShard):
                # Per-param variable-size reduce-scatter for RaggedShard
                for fqn, info, grad in param_list:
                    placement = info.placements[0]
                    local_shape = placement.compute_local_shape(
                        info.global_shape, my_rank, world_size
                    )
                    recv_buf = torch.empty(
                        local_shape, dtype=grad.dtype, device=grad.device
                    )
                    placement.comm_reduce(grad, recv_buf, self._mesh)
                    if info.local_numel > 0:
                        self._sharded_params[fqn].grad = recv_buf
            else:
                # Per-param reduce for other placements (e.g., Owned)
                for fqn, info, grad in param_list:
                    placement = info.placements[0]
                    placement.comm_reduce(grad, grad, self._mesh)
                    if info.local_numel > 0 and my_rank == info.owner_rank:
                        self._sharded_params[fqn].grad = grad

        torch.cuda.synchronize()

    @contextmanager
    def unsharded(self):
        """
        Context manager for automatic unshard/reshard around forward.

        Usage:
            with storage.unsharded():
                output = model(input)
        """
        self.unshard()
        try:
            yield
        finally:
            self.reshard()

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
        """Post-backward callback: reshard and reduce-scatter gradients."""
        # Ensure we only run once per backward pass
        if self._post_backward_called:
            return
        self._post_backward_called = True

        # Only reshard if currently unsharded
        if self._state == ShardedState.UNSHARDED:
            self.reshard()

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


def _compute_max_local_numel(
    global_shape: torch.Size,
    placements: tuple[Placement, ...],
    world_size: int,
) -> int:
    """
    Compute the max local numel across all ranks for a placement.

    Rank 0 gets the max chunk in ceil-based sharding.
    """
    return placements[0].compute_local_numel(global_shape, 0, world_size)


def _create_param_infos(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
) -> tuple[dict[str, ParamInfo], int, int, dict[str, int]]:
    """
    Create ParamInfo for each parameter, computing local shapes and byte offsets.

    Parameters are laid out sequentially in the byte buffer with proper alignment.
    Uses padded sizes for uniform buffer layout across ranks (enables batched all-gather).

    Returns:
        param_infos: dict mapping FQN to ParamInfo
        total_bytes: total bytes needed for the sharded buffer
        total_unsharded_bytes: total bytes needed for the unsharded buffer
        region_info: dict with shard/owned region boundaries
    """
    world_size = mesh.size()
    param_infos: dict[str, ParamInfo] = {}
    current_byte_offset = 0
    current_unsharded_byte_offset = 0

    for fqn, param in named_params:
        global_shape = param.shape
        global_stride = make_contiguous_strides_for(global_shape)
        local_shape, local_numel = _compute_local_info(global_shape, mesh, placements)
        dtype = param.dtype

        # Compute global numel
        global_numel = 1
        for dim in global_shape:
            global_numel *= dim

        # Compute padded local numel (max across all ranks for uniform layout)
        padded_local_numel = _compute_max_local_numel(
            global_shape, placements, world_size
        )

        # Align offset for this dtype (sharded buffer)
        alignment = _get_dtype_alignment(dtype)
        aligned_offset = _align_offset(current_byte_offset, alignment)

        # Align offset for unsharded buffer
        aligned_unsharded_offset = _align_offset(
            current_unsharded_byte_offset, alignment
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
            byte_offset=aligned_offset,
            global_numel=global_numel,
            unsharded_byte_offset=aligned_unsharded_offset,
            padded_local_numel=padded_local_numel,
        )
        param_infos[fqn] = info

        # Move offsets past this parameter's PADDED bytes (for uniform layout)
        param_bytes = padded_local_numel * dtype.itemsize
        current_byte_offset = aligned_offset + param_bytes

        unsharded_param_bytes = global_numel * dtype.itemsize
        current_unsharded_byte_offset = aligned_unsharded_offset + unsharded_param_bytes

    # For Shard-only: entire buffer is shard region
    region_info = {
        "shard_region_start": 0,
        "shard_region_end": current_byte_offset,
        "ragged_region_start": current_byte_offset,
        "ragged_region_end": current_byte_offset,
        "owned_region_start": current_byte_offset,
        "owned_region_end": current_byte_offset,
    }

    return param_infos, current_byte_offset, current_unsharded_byte_offset, region_info


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


def _create_param_infos_param_boundary(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
    assignments: dict[str, int],
) -> tuple[dict[str, ParamInfo], int, int, dict[str, int]]:
    """
    Create ParamInfo for parameter-boundary sharding.

    Each parameter is assigned to one rank. The owner has full data,
    non-owners have empty storage for this param.

    Returns:
        param_infos: dict mapping FQN to ParamInfo
        total_bytes: total bytes needed for the sharded buffer (this rank's owned params)
        total_unsharded_bytes: total bytes needed for the unsharded buffer (all params)
        region_info: dict with shard/owned region boundaries
    """
    my_rank = mesh.get_local_rank()
    param_infos: dict[str, ParamInfo] = {}
    current_byte_offset = 0
    current_unsharded_byte_offset = 0

    for fqn, param in named_params:
        global_shape = param.shape
        global_stride = make_contiguous_strides_for(global_shape)
        dtype = param.dtype
        owner_rank = assignments[fqn]

        global_numel = param.numel()

        owned = Owned(owner_rank)
        local_shape = owned.compute_local_shape(global_shape, my_rank, mesh.size())
        local_numel = owned.compute_local_numel(global_shape, my_rank, mesh.size())

        placement = (owned,)

        # Align offset for this dtype (sharded buffer - only for owned params)
        alignment = _get_dtype_alignment(dtype)

        # Only allocate space in sharded buffer if this rank owns the param
        if my_rank == owner_rank:
            aligned_offset = _align_offset(current_byte_offset, alignment)
            byte_offset = aligned_offset
            param_bytes = local_numel * dtype.itemsize
            current_byte_offset = aligned_offset + param_bytes
        else:
            byte_offset = 0  # Not stored locally

        # Unsharded buffer offset (all ranks need space for all params)
        aligned_unsharded_offset = _align_offset(
            current_unsharded_byte_offset, alignment
        )
        unsharded_param_bytes = global_numel * dtype.itemsize
        current_unsharded_byte_offset = aligned_unsharded_offset + unsharded_param_bytes

        info = ParamInfo(
            fqn=fqn,
            global_shape=global_shape,
            global_stride=tuple(global_stride),
            dtype=dtype,
            requires_grad=param.requires_grad,
            placements=placement,
            local_shape=local_shape,
            local_numel=local_numel,
            byte_offset=byte_offset,
            global_numel=global_numel,
            unsharded_byte_offset=aligned_unsharded_offset,
            owner_rank=owner_rank,
        )
        param_infos[fqn] = info

    # For param_boundary: no shard region (all params are Owned)
    region_info = {
        "shard_region_start": 0,
        "shard_region_end": 0,
        "ragged_region_start": 0,
        "ragged_region_end": 0,
        "owned_region_start": 0,
        "owned_region_end": current_byte_offset,
    }

    return param_infos, current_byte_offset, current_unsharded_byte_offset, region_info


def _create_param_infos_with_placement_fn(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
    shard_placement_fn: Callable[
        [str, nn.Parameter], Shard | Owned | RaggedShard | None
    ],
    default_placement: Shard | Owned | RaggedShard = Shard(0),
) -> tuple[dict[str, ParamInfo], int, int, dict[str, int]]:
    """
    Create ParamInfo for each parameter using a per-parameter placement function.

    Supports mixed Shard, RaggedShard, and Owned placements in the same storage.
    Uses padded sizes for Shard params to enable batched all-gather.
    RaggedShard uses actual (non-padded) sizes since buffer sizes differ per rank.

    Args:
        named_params: List of (fqn, param) tuples
        mesh: Device mesh for sharding
        shard_placement_fn: Function mapping (fqn, param) -> placement or None
            Returns None for default placement
        default_placement: Default placement when function returns None

    Returns:
        param_infos: dict mapping FQN to ParamInfo
        total_bytes: total bytes needed for the sharded buffer
        total_unsharded_bytes: total bytes needed for the unsharded buffer
        region_info: dict with shard/ragged/owned region boundaries
    """
    my_rank = mesh.get_local_rank()
    world_size = mesh.size()

    # First pass: categorize params by placement type
    shard_params: list[tuple[str, nn.Parameter, Shard]] = []
    ragged_params: list[tuple[str, nn.Parameter, RaggedShard]] = []
    owned_params: list[tuple[str, nn.Parameter, Owned]] = []

    for fqn, param in named_params:
        placement = shard_placement_fn(fqn, param)
        if placement is None:
            placement = default_placement

        # Normalize negative dims for Shard
        if isinstance(placement, Shard) and placement.dim < 0:
            placement = Shard(placement.dim + param.ndim)

        if isinstance(placement, Owned):
            owned_params.append((fqn, param, placement))
        elif isinstance(placement, RaggedShard):
            ragged_params.append((fqn, param, placement))
        else:
            assert isinstance(placement, Shard)
            shard_params.append((fqn, param, placement))

    # Second pass: layout buffer — Shard → Ragged → Owned
    param_infos: dict[str, ParamInfo] = {}
    current_byte_offset = 0
    current_unsharded_byte_offset = 0

    # Layout Shard params first (contiguous region for batched all-gather)
    shard_region_start = 0
    for fqn, param, placement in shard_params:
        global_shape = param.shape
        global_stride = make_contiguous_strides_for(global_shape)
        dtype = param.dtype
        global_numel = param.numel()

        placements_tuple = (placement,)
        local_shape, local_numel = _compute_local_info(
            global_shape, mesh, placements_tuple
        )

        # Compute padded local numel for uniform buffer layout
        padded_local_numel = _compute_max_local_numel(
            global_shape, placements_tuple, world_size
        )

        alignment = _get_dtype_alignment(dtype)
        aligned_offset = _align_offset(current_byte_offset, alignment)
        byte_offset = aligned_offset
        # Use PADDED size for buffer allocation
        param_bytes = padded_local_numel * dtype.itemsize
        current_byte_offset = aligned_offset + param_bytes

        # Unsharded buffer offset
        aligned_unsharded_offset = _align_offset(
            current_unsharded_byte_offset, alignment
        )
        unsharded_param_bytes = global_numel * dtype.itemsize
        current_unsharded_byte_offset = aligned_unsharded_offset + unsharded_param_bytes

        info = ParamInfo(
            fqn=fqn,
            global_shape=global_shape,
            global_stride=tuple(global_stride),
            dtype=dtype,
            requires_grad=param.requires_grad,
            placements=placements_tuple,
            local_shape=local_shape,
            local_numel=local_numel,
            byte_offset=byte_offset,
            global_numel=global_numel,
            unsharded_byte_offset=aligned_unsharded_offset,
            owner_rank=None,
            padded_local_numel=padded_local_numel,
        )
        param_infos[fqn] = info

    shard_region_end = current_byte_offset

    # Layout RaggedShard params (variable sizes per rank, NOT padded)
    ragged_region_start = current_byte_offset
    for fqn, param, placement in ragged_params:
        global_shape = param.shape
        global_stride = make_contiguous_strides_for(global_shape)
        dtype = param.dtype
        global_numel = param.numel()

        local_shape = placement.compute_local_shape(global_shape, my_rank, world_size)
        local_numel = placement.compute_local_numel(global_shape, my_rank, world_size)

        placements_tuple = (placement,)

        # Use actual local_numel (not padded) — buffer sizes differ per rank
        alignment = _get_dtype_alignment(dtype)
        aligned_offset = _align_offset(current_byte_offset, alignment)
        byte_offset = aligned_offset
        param_bytes = local_numel * dtype.itemsize
        current_byte_offset = aligned_offset + param_bytes

        # Unsharded buffer offset
        aligned_unsharded_offset = _align_offset(
            current_unsharded_byte_offset, alignment
        )
        unsharded_param_bytes = global_numel * dtype.itemsize
        current_unsharded_byte_offset = aligned_unsharded_offset + unsharded_param_bytes

        info = ParamInfo(
            fqn=fqn,
            global_shape=global_shape,
            global_stride=tuple(global_stride),
            dtype=dtype,
            requires_grad=param.requires_grad,
            placements=placements_tuple,
            local_shape=local_shape,
            local_numel=local_numel,
            byte_offset=byte_offset,
            global_numel=global_numel,
            unsharded_byte_offset=aligned_unsharded_offset,
        )
        param_infos[fqn] = info

    ragged_region_end = current_byte_offset

    # Layout Owned params (only owner stores data)
    owned_region_start = current_byte_offset
    for fqn, param, placement in owned_params:
        global_shape = param.shape
        global_stride = make_contiguous_strides_for(global_shape)
        dtype = param.dtype
        global_numel = param.numel()
        owner_rank = placement.owner_rank

        local_shape = placement.compute_local_shape(global_shape, my_rank, world_size)
        local_numel = placement.compute_local_numel(global_shape, my_rank, world_size)

        if local_numel > 0:
            alignment = _get_dtype_alignment(dtype)
            aligned_offset = _align_offset(current_byte_offset, alignment)
            byte_offset = aligned_offset
            param_bytes = local_numel * dtype.itemsize
            current_byte_offset = aligned_offset + param_bytes
        else:
            byte_offset = 0  # Not stored locally

        placements_tuple = (placement,)

        # Unsharded buffer offset
        alignment = _get_dtype_alignment(dtype)
        aligned_unsharded_offset = _align_offset(
            current_unsharded_byte_offset, alignment
        )
        unsharded_param_bytes = global_numel * dtype.itemsize
        current_unsharded_byte_offset = aligned_unsharded_offset + unsharded_param_bytes

        info = ParamInfo(
            fqn=fqn,
            global_shape=global_shape,
            global_stride=tuple(global_stride),
            dtype=dtype,
            requires_grad=param.requires_grad,
            placements=placements_tuple,
            local_shape=local_shape,
            local_numel=local_numel,
            byte_offset=byte_offset,
            global_numel=global_numel,
            unsharded_byte_offset=aligned_unsharded_offset,
            owner_rank=owner_rank,
        )
        param_infos[fqn] = info

    owned_region_end = current_byte_offset

    # Return param_infos, total bytes, total unsharded bytes, and region info
    region_info = {
        "shard_region_start": shard_region_start,
        "shard_region_end": shard_region_end,
        "ragged_region_start": ragged_region_start,
        "ragged_region_end": ragged_region_end,
        "owned_region_start": owned_region_start,
        "owned_region_end": owned_region_end,
    }
    return param_infos, current_byte_offset, current_unsharded_byte_offset, region_info


def _pack_original_data(
    byte_storage: torch.Tensor,
    named_params: list[tuple[str, nn.Parameter]],
    param_infos: dict[str, ParamInfo],
    mesh: DeviceMesh,
) -> None:
    """Pack original parameter data into byte storage via placement.pack_unshard().

    Each placement knows how to extract its rank's local shard from the full
    param and copy it into byte_storage at the correct offset.
    """
    my_rank = mesh.get_local_rank()
    world_size = mesh.size()

    for fqn, param in named_params:
        info = param_infos[fqn]
        if param.device.type == "meta":
            continue
        info.placements[0].pack_unshard(
            param.data, byte_storage, info.byte_offset, my_rank, world_size
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


def flex_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    placements: tuple[Placement, ...] | None = None,
    reshard_after_forward: bool = True,
    register_hooks: bool = True,
    shard_strategy: Literal["per_param", "param_boundary"] = "per_param",
    shard_placement_fn: Callable[
        [str, nn.Parameter], Shard | Owned | RaggedShard | None
    ]
    | None = None,
) -> FlexShardModule:
    """
    Apply flat-storage FSDP sharding to a module.

    This function:
    1. Collects parameters from the module (excluding already-wrapped submodules)
    2. Creates a single unified byte buffer for all parameters (regardless of dtype)
    3. Replaces each parameter with a plain tensor annotated with placement metadata
       (a typed view into the byte buffer at the appropriate offset)
    4. Optionally registers forward/backward hooks for automatic unshard/reshard
    5. Stores DStorage on the module (accessible via module.dstorage)

    The unified byte buffer enables a single all-gather operation for all parameters.

    Nested wrapping is supported: apply flex_shard to inner modules first,
    then to outer modules. The outer module's storage will exclude parameters
    from already-wrapped inner modules.

    Args:
        module: The module to shard. Can have real or meta device parameters.
        mesh: The device mesh for sharding. Currently only 1D mesh is supported.
        placements: The default sharding placements. Defaults to (Shard(0),).
            Used when shard_placement_fn is None or returns None for a param.
            Ignored when shard_strategy="param_boundary".
        reshard_after_forward: If True (default), reshard parameters after forward
            to save memory. Parameters will be re-unsharded in backward.
            If False, keep parameters unsharded between forward and backward.
        register_hooks: If True (default), register forward/backward hooks for
            automatic unshard/reshard. If False, caller must manually call
            unshard()/reshard().
        shard_strategy: The default sharding strategy (ignored if shard_placement_fn is provided):
            - "per_param" (default): Each parameter is sharded across all ranks
              along the specified dimension. Uses Shard placement.
            - "param_boundary": Each parameter is assigned to one rank (owner).
              The owner has full parameter data, others have empty tensors.
              Uses Owned placement and greedy bin-packing for balanced memory.
        shard_placement_fn: Optional callable for per-parameter placement control.
            Takes (fqn, param) and returns Shard | Owned | None.
            - Shard(dim): Shard this parameter along dimension dim
            - Owned(rank): Assign this parameter to the specified rank
            - None: Use default placement from `placements` parameter
            This enables mixed Shard/Owned placements in a single DStorage.

    Returns:
        The module (mutated in-place). Use module.storage to access internals.

    Example::

        >>> mesh = init_device_mesh("cuda", (world_size,))
        >>> model = Transformer(args)
        >>> # Nested wrapping: wrap layers first, then root
        >>> for layer in model.layers:
        ...     flex_shard(layer, mesh)
        >>> storage = flex_shard(model, mesh)  # Only wraps non-layer params
        >>> # Forward/backward now work automatically with hooks
        >>> output = model(input)
        >>> output.sum().backward()

    Example with per-parameter placement::

        >>> def placement_fn(fqn, param):
        ...     if "embed" in fqn:
        ...         return Owned(0)  # Embeddings owned by rank 0
        ...     return Shard(0)  # Other params sharded
        >>> storage = flex_shard(model, mesh, shard_placement_fn=placement_fn)

    Note:
        - Parameters of different dtypes are supported in a single unified buffer
        - Proper alignment is maintained for each dtype
        - Parameters on meta device will have uninitialized storage
        - The returned DStorage is also stored on module._dstorage
        - Forward/backward hooks are automatically registered for unshard/reshard
    """
    if placements is None:
        placements = (Shard(0),)

    # Check if module is already wrapped
    if getattr(module, _DSTORAGE_ATTR, None) is not None:
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

    # Create parameter infos with local shapes and byte offsets
    region_info = None
    if shard_placement_fn is not None:
        # Per-parameter placement function provided
        default_placement = placements[0] if placements else Shard(0)
        param_infos, total_bytes, total_unsharded_bytes, region_info = (
            _create_param_infos_with_placement_fn(
                named_params, mesh, shard_placement_fn, default_placement
            )
        )
    elif shard_strategy == "param_boundary":
        # Assign params to ranks using bin-packing
        assignments = _assign_params_to_ranks(named_params, mesh.size())
        param_infos, total_bytes, total_unsharded_bytes, region_info = (
            _create_param_infos_param_boundary(named_params, mesh, assignments)
        )
    else:
        param_infos, total_bytes, total_unsharded_bytes, region_info = (
            _create_param_infos(named_params, mesh, placements)
        )

    # Allocate unified byte storage
    byte_storage = torch.empty(total_bytes, dtype=torch.uint8, device=device)

    # Pack original data into byte storage via placement.pack_unshard()
    _pack_original_data(byte_storage, named_params, param_infos, mesh)

    # Replace each parameter with annotated plain tensor (before creating DStorage)
    for fqn, info in param_infos.items():
        local_view = byte_storage[
            info.byte_offset : info.byte_offset + info.local_numel * info.dtype.itemsize
        ]
        typed_view = local_view.view(info.dtype).view(info.local_shape)
        new_param = nn.Parameter(typed_view, requires_grad=info.requires_grad)
        _create_sharded_view(new_param, info, mesh)
        _set_param_on_module(module, fqn, new_param)

    # Create DStorage (after sharded params are registered)
    # This also registers forward/backward hooks if requested
    storage = DStorage(
        byte_storage,
        param_infos,
        mesh,
        total_bytes,
        total_unsharded_bytes,
        module,
        reshard_after_forward=reshard_after_forward,
        register_hooks=register_hooks,
        region_info=region_info,
    )

    # Store DStorage on module (accessible via module.dstorage)
    setattr(module, _DSTORAGE_ATTR, storage)

    # Change module class to include FlexShardModule mixin
    cls = type(module)
    if not issubclass(cls, FlexShardModule):
        module.__class__ = type(cls.__name__, (cls, FlexShardModule), {})

    return module
