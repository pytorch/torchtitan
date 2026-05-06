# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for

from .state import set_sharding_info
from .utils import _set_param_on_module

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .placements import Placement


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
    mesh: DeviceMesh,
    param_placements: dict[str, tuple[Placement, ...]],
) -> tuple[dict[str, ParamInfo], int]:
    """
    Create ParamInfo for each parameter, computing local shapes and byte offsets.

    The caller validates that each bucket uses Shard(0) and a uniform dtype, so
    parameters are laid out sequentially in the byte buffer.

    Args:
        named_params: List of (fqn, param) tuples
        mesh: DP shard mesh used for sharding
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
        local_shape, local_numel = _compute_local_info(global_shape, mesh, placements)
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


def _materialize_bucket_storages(
    module: nn.Module,
    named_params: list[tuple[str, nn.Parameter]],
    bucket_assignments: list[list[str]],
    buckets: list[BucketSpec],
    param_placements: dict[str, tuple[Placement, ...]],
    mesh: DeviceMesh,
    device: torch.device,
) -> tuple[list[DStorage], dict[str, BucketSpec]]:
    """Create DStorages for bucket assignments and install sharded parameters."""
    named_params_dict = dict(named_params)
    storages: list[DStorage] = []
    fqn_to_bucket_spec: dict[str, BucketSpec] = {}

    for bucket_idx, bucket_fqns in enumerate(bucket_assignments):
        if not bucket_fqns:
            continue

        bucket_spec = buckets[bucket_idx]
        for fqn in bucket_fqns:
            fqn_to_bucket_spec[fqn] = bucket_spec

        bucket_named_params = [(fqn, named_params_dict[fqn]) for fqn in bucket_fqns]
        bucket_placements = {fqn: param_placements[fqn] for fqn in bucket_fqns}
        param_infos, total_bytes = _create_param_infos(
            bucket_named_params, mesh, bucket_placements
        )

        if bucket_spec.offload_policy is not None:
            byte_storage = torch.empty(
                total_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=bucket_spec.offload_policy.pin_memory,
            )
            expected_param_device = torch.device("cpu")
        else:
            byte_storage = torch.empty(total_bytes, dtype=torch.uint8, device=device)
            expected_param_device = torch.device(device)

        _write_params_to_dstorage(byte_storage, bucket_named_params, param_infos, mesh)

        for fqn, info in param_infos.items():
            num_bytes = info.local_numel * info.dtype.itemsize
            local_view = byte_storage[info.byte_offset : info.byte_offset + num_bytes]
            typed_view = local_view.view(info.dtype).view(info.local_shape)
            new_param = nn.Parameter(typed_view, requires_grad=info.requires_grad)
            if new_param.device != expected_param_device:
                raise AssertionError(
                    f"Expected sharded parameter {fqn!r} on "
                    f"{expected_param_device}, but got {new_param.device}"
                )
            _create_sharded_view(new_param, info, mesh)
            _set_param_on_module(module, fqn, new_param)

        storages.append(
            DStorage(
                byte_storage,
                param_infos,
                mesh,
                total_bytes,
                module,
                reshard_after_forward=bucket_spec.reshard_after_forward,
            )
        )

    return storages, fqn_to_bucket_spec
