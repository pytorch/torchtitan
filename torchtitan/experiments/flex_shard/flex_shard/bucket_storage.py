# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for

from .sharded_param import set_sharding_info
from .utils import _get_single_placement, _set_param_on_module

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .placement_contract import Placement
    from .reshard_after_forward import _ReshardAfterForwardRecomputeState


BucketParamFQNsByIndex = list[list[str]]

PlacementFn = Callable[
    [list[tuple[str, nn.Parameter]], "DeviceMesh"],
    dict[str, tuple["Placement", ...]],
]


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

    This is a placeholder for future CPU offload support. The minimal eager
    FlexShard path currently rejects non-None offload policies before storage
    materialization.

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
        placement_fn: Required callable that maps this bucket's
            ``(named_params, mesh)`` to per-parameter placements.
            The minimal eager path expects one ``Placement`` per parameter.
        mp_policy: Mixed precision policy for this bucket. This currently
            covers parameter and gradient-reduction dtypes only.
            TODO: add module-boundary input/output casting separately from
            BucketSpec. FSDP2 exposes cast_forward_inputs and output_dtype, but
            those apply to a module's forward args/outputs, not to an individual
            parameter bucket. Keeping them out of BucketSpec avoids ambiguous
            behavior when multiple buckets share the same hooked module.
        offload_policy: CPU offload policy for this bucket. TODO: implement
            and test CPU offload before allowing this in flex_shard().
        reshard_after_forward: Whether to free this bucket's unsharded
            parameters after forward and recompute them in backward. This
            defaults to True. Buckets that reshard after forward must have
            hooks that run in both the original forward and activation
            checkpoint recomputation.
    """

    patterns: list[str]
    placement_fn: PlacementFn
    mp_policy: MixedPrecisionPolicy | None = None
    offload_policy: OffloadPolicy | None = None
    reshard_after_forward: bool = True


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
    storage_nbytes: int = 0  # bytes reserved for this param's local storage
    global_numel: int = 0  # total elements in unsharded param

    @property
    def placement(self) -> Placement:
        """The single placement supported by the minimal eager path."""
        return _get_single_placement(self.placements)


class ShardedBucketStorage:
    """
    Manages a byte buffer that backs one bucket of sharded parameters.

    All parameters in a bucket storage must share a dtype and placement-compatible
    local layout. Each placement owns its parameter's local storage layout and
    exposed tensor view; ShardedBucketStorage only places those layouts
    sequentially in one byte buffer.

    Communication is delegated to eager hooks and parameter accessors; this
    bucket storage object owns the byte buffer and metadata.
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
        self._reshard_after_forward_recompute_state: (
            _ReshardAfterForwardRecomputeState | None
        ) = None

    @classmethod
    def from_bucket(
        cls,
        module: nn.Module,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
        device: torch.device,
        bucket_spec: BucketSpec,
    ) -> ShardedBucketStorage:
        """Create storage metadata for one bucket and install sharded params."""
        param_infos, total_bytes = cls.create_param_infos(
            named_params,
            mesh,
            param_placements,
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

        bucket_storage = cls(
            byte_storage,
            param_infos,
            mesh,
            total_bytes,
            module,
            reshard_after_forward=bucket_spec.reshard_after_forward,
        )
        bucket_storage.copy_params_from(named_params)
        bucket_storage.install_sharded_params(expected_param_device)
        return bucket_storage

    @staticmethod
    def _compute_local_storage_layout(
        global_shape: torch.Size,
        dtype: torch.dtype,
        mesh: DeviceMesh,
        placements: tuple[Placement, ...],
    ):
        """Compute the placement-owned local storage layout."""
        rank = mesh.get_local_rank()
        world_size = mesh.size()
        placement = _get_single_placement(placements)
        return placement.local_storage_layout(global_shape, dtype, rank, world_size)

    @classmethod
    def create_param_infos(
        cls,
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
        param_placements: dict[str, tuple[Placement, ...]],
    ) -> tuple[dict[str, ParamInfo], int]:
        """
        Create ParamInfo for each parameter, computing local layout and byte offsets.

        The caller validates that each bucket uses compatible placements and a
        uniform dtype. Each placement owns its per-parameter local storage layout;
        bucket storage only places those layouts sequentially in the byte buffer.
        """
        param_infos: dict[str, ParamInfo] = {}
        current_byte_offset = 0

        for fqn, param in named_params:
            placements = param_placements[fqn]
            global_shape = param.shape
            global_stride = make_contiguous_strides_for(global_shape)
            dtype = param.dtype
            local_storage_layout = cls._compute_local_storage_layout(
                global_shape, dtype, mesh, placements
            )
            global_numel = param.numel()

            if local_storage_layout.storage_nbytes > 0:
                byte_offset = current_byte_offset
                current_byte_offset += local_storage_layout.storage_nbytes
            else:
                byte_offset = 0

            info = ParamInfo(
                fqn=fqn,
                global_shape=global_shape,
                global_stride=tuple(global_stride),
                dtype=dtype,
                requires_grad=param.requires_grad,
                placements=placements,
                local_shape=local_storage_layout.local_shape,
                local_numel=local_storage_layout.local_numel,
                byte_offset=byte_offset,
                storage_nbytes=local_storage_layout.storage_nbytes,
                global_numel=global_numel,
            )
            param_infos[fqn] = info

        return param_infos, current_byte_offset

    def copy_params_from(
        self,
        named_params: list[tuple[str, nn.Parameter]],
    ) -> None:
        """Pack original parameter data into byte storage."""
        my_rank = self._mesh.get_local_rank()
        world_size = self._mesh.size()

        for fqn, param in named_params:
            info = self._param_infos[fqn]
            info.placement.copy_param_to_storage(
                self._byte_storage,
                info,
                param,
                my_rank,
                world_size,
            )

    def install_sharded_params(
        self,
        expected_device: torch.device,
    ) -> None:
        """Replace original module parameters with local storage views."""
        for fqn, info in self._param_infos.items():
            typed_view = info.placement.make_local_storage_view(
                self._byte_storage,
                info,
            )
            new_param = nn.Parameter(typed_view, requires_grad=info.requires_grad)
            if new_param.device != expected_device:
                raise AssertionError(
                    f"Expected sharded parameter {fqn!r} on "
                    f"{expected_device}, but got {new_param.device}"
                )
            set_sharding_info(
                new_param,
                placements=info.placements,
                global_shape=info.global_shape,
                global_stride=info.global_stride,
                mesh=self._mesh,
            )
            _set_param_on_module(self._module, fqn, new_param)

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
        return info.placement.make_local_storage_view(self._byte_storage, info)


def _assign_params_to_buckets(
    param_fqns: list[str],
    buckets: list[BucketSpec],
) -> BucketParamFQNsByIndex:
    """Assign each param FQN to exactly one bucket via fnmatch.

    Returns:
        A list aligned with ``buckets``: result[bucket_idx] is the ordered list
        of parameter FQNs assigned to ``buckets[bucket_idx]``.

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
    assignments: BucketParamFQNsByIndex = [[] for _ in buckets]
    for fqn, idxs in param_to_buckets.items():
        assignments[idxs[0]].append(fqn)

    return assignments
