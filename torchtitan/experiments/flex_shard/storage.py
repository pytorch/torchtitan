# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for

from .metadata import set_sharding_info

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .placements import Placement


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


class DStorage:
    """
    Manages a byte buffer that backs one bucket of sharded parameters.

    All parameters in a storage must share the same dtype and use Shard(0)
    placement. Each parameter's local shard is a typed view into this buffer at
    its sequential byte offset.

    Communication is delegated to eager hooks and parametrization modules; this
    storage object owns buffer layout and metadata.
    """

    def __init__(
        self,
        byte_storage: torch.Tensor,
        param_infos: dict[str, ParamInfo],
        mesh: DeviceMesh,
        total_bytes: int,
        module: nn.Module,
        reshard_after_forward: bool = True,
    ) -> None:
        if byte_storage.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 storage, got {byte_storage.dtype}")
        self._byte_storage = byte_storage
        self._param_infos = param_infos
        self._mesh = mesh
        self._total_bytes = total_bytes
        self._module = module
        self._reshard_after_forward = reshard_after_forward

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
    def numel(self) -> int:
        """Total number of bytes (for compatibility, returns byte count)."""
        return self._byte_storage.numel()

    @property
    def param_infos(self) -> dict[str, ParamInfo]:
        """Metadata for each parameter."""
        return self._param_infos

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


def _create_param_infos(
    named_params: list[tuple[str, nn.Parameter]],
    mesh_info: Any,
    param_placements: dict[str, tuple[Placement, ...]],
) -> tuple[dict[str, ParamInfo], int]:
    """
    Create ParamInfo for each parameter, computing local shapes and byte offsets.

    The caller validates that each bucket uses Shard(0) and a uniform dtype, so
    parameters are laid out sequentially in the byte buffer.

    Args:
        named_params: List of (fqn, param) tuples
        mesh_info: Mesh metadata for sharding
        param_placements: Dict mapping FQN to placement tuple for each parameter

    Returns:
        param_infos: dict mapping FQN to ParamInfo
        total_bytes: total bytes needed for the sharded buffer
    """
    param_infos: dict[str, ParamInfo] = {}
    current_byte_offset = 0

    for fqn, param in named_params:
        placements = param_placements[fqn]
        global_shape = param.shape
        global_stride = make_contiguous_strides_for(global_shape)
        local_shape, local_numel = _compute_local_info(
            global_shape, mesh_info.dp_shard_mesh, placements
        )
        dtype = param.dtype
        global_numel = param.numel()

        # Sharded buffer: only allocate if this rank has data
        if local_numel > 0:
            byte_offset = current_byte_offset
            current_byte_offset += local_numel * dtype.itemsize
        else:
            byte_offset = 0

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
        )
        param_infos[fqn] = info

    return param_infos, current_byte_offset


def _create_sharded_view(
    local_view: torch.Tensor,
    info: ParamInfo,
    mesh_info: Any,
) -> torch.Tensor:
    """Annotate a local tensor view with placement metadata."""
    set_sharding_info(
        local_view,
        placements=info.placements,
        global_shape=info.global_shape,
        global_stride=info.global_stride,
        mesh=mesh_info.dp_shard_mesh,
    )
    return local_view


def _write_params_to_dstorage(
    byte_storage: torch.Tensor,
    named_params: list[tuple[str, nn.Parameter]],
    param_infos: dict[str, ParamInfo],
    mesh_info: Any,
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
        param_data = param.detach()
        if param_data.device.type == "meta":
            continue
        if not param_data.is_contiguous():
            param_data = param_data.contiguous()
        shard = info.placements[0].extract_local_shard(param_data, my_rank, world_size)
        if shard.numel() > 0:
            nbytes = shard.numel() * shard.element_size()
            byte_storage[info.byte_offset : info.byte_offset + nbytes].copy_(
                shard.reshape(-1).view(torch.uint8)
            )
