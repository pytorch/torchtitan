# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed.fsdp import DataParallelMeshDims

from .eager_runtime import (
    _create_eager_param_states,
    _install_batched_allgather_hooks,
)
from .sharding_metadata import (
    get_global_shape,
    get_placements,
    is_flex_shard_param,
    set_sharding_info,
)
from .module_wrapping import (
    _attach_flex_shard_module_state,
    _check_not_already_flex_sharded,
    _register_module_param_accessors,
    FlexShardModule,
)
from .reshard_after_forward import (
    _apply_reshard_after_forward,
)
from .storage import (
    _assign_params_to_buckets,
    _materialize_bucket_storages,
    BucketSpec,
    MixedPrecisionPolicy,
)
from .utils import (
    _get_device_from_mesh,
    _get_dp_shard_mesh,
    _get_managed_named_params,
    _validate_bucket_placements,
    _validate_eager_params,
    _validate_placements,
)

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .placements import Placement


__all__ = [
    "BucketSpec",
    "flex_shard",
    "get_global_shape",
    "get_placements",
    "is_flex_shard_param",
    "MixedPrecisionPolicy",
    "set_sharding_info",
]


PlacementFn = Callable[
    [list[tuple[str, nn.Parameter]], "DeviceMesh"],
    dict[str, tuple["Placement", ...]],
]


@dataclass(frozen=True)
class PreparedFlexShardInputs:
    """Validated inputs and derived setup state for flex_shard()."""

    named_params: list[tuple[str, nn.Parameter]]
    shard_mesh: DeviceMesh
    device: torch.device
    param_placements: dict[str, tuple[Placement, ...]]
    bucket_assignments: list[list[str]]


def _prepare_flex_shard_inputs(
    module: nn.Module,
    mesh: DeviceMesh,
    dp_mesh_dims: DataParallelMeshDims,
    shard_placement_fn: PlacementFn,
    buckets: list[BucketSpec],
) -> PreparedFlexShardInputs:
    """Validate inputs and derive setup state for flex_shard()."""
    _check_not_already_flex_sharded(module)

    named_params = _get_managed_named_params(module)
    if not named_params:
        raise ValueError(
            f"Module {type(module).__name__} has no parameters to shard. "
            "All parameters may belong to already-wrapped submodules."
        )

    shard_mesh = _get_dp_shard_mesh(mesh, dp_mesh_dims)
    all_params_meta = all(param.device.type == "meta" for _, param in named_params)
    device = (
        torch.device("meta") if all_params_meta else _get_device_from_mesh(shard_mesh)
    )
    _validate_eager_params(
        named_params,
        expected_device=None if all_params_meta else device,
    )

    param_placements = shard_placement_fn(named_params, shard_mesh)
    _validate_placements(param_placements, named_params, shard_mesh)

    if not buckets:
        raise ValueError("flex_shard requires at least one BucketSpec in buckets.")
    if not all(isinstance(bucket, BucketSpec) for bucket in buckets):
        raise TypeError("flex_shard buckets must be a list of BucketSpec objects.")

    param_fqns = [fqn for fqn, _ in named_params]
    bucket_assignments = _assign_params_to_buckets(param_fqns, buckets)
    _validate_bucket_placements(
        bucket_assignments,
        param_placements,
        buckets,
        named_params,
    )

    return PreparedFlexShardInputs(
        named_params=named_params,
        shard_mesh=shard_mesh,
        device=device,
        param_placements=param_placements,
        bucket_assignments=bucket_assignments,
    )


def flex_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    dp_mesh_dims: DataParallelMeshDims,
    shard_placement_fn: PlacementFn,
    buckets: list[BucketSpec],
) -> FlexShardModule:
    """
    Apply flat-storage FSDP sharding to a module.

    This function:
    1. Collects parameters from the module (excluding already-wrapped submodules)
    2. Groups parameters into communication buckets (one per bucket, or all in one)
    3. Creates a unified byte buffer per bucket for all its parameters
    4. Replaces each parameter with a plain tensor annotated with placement metadata
    5. Registers property-based accessors for eager parameter access
    6. Stores DStorages on the module (accessible via module.dstorages)

    Each bucket gets its own byte buffer and DStorage, enabling independent
    all-gather operations per bucket.

    Nested wrapping is supported: apply flex_shard to inner modules first,
    then to outer modules. The outer module's storage will exclude parameters
    from already-wrapped inner modules.

    Args:
        module: The module to shard. Can have real or meta device parameters.
        mesh: The named device mesh for sharding.
        dp_mesh_dims: Names for the data-parallel dimensions in ``mesh``.
            FlexShard derives its DP shard mesh from ``mesh``.
        shard_placement_fn: Required callable that maps
            ``(named_params, dp_shard_mesh)`` to per-parameter placements.
            The minimal eager path currently supports ``Shard(0)`` placements,
            for example via ``per_param_placements``.
        buckets: Required list of bucket specifications. Use
            ``[BucketSpec(["*"])]`` for a single whole-module bucket.

    Returns:
        The module (mutated in-place). Use module.dstorages to access internals.

    Example::

        >>> mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        >>> model = Transformer(args)
        >>> model.to("cuda")
        >>> # Single bucket:
        >>> flex_shard(
        ...     model,
        ...     mesh,
        ...     DataParallelMeshDims(shard="fsdp"),
        ...     shard_placement_fn=per_param_placements,
        ...     buckets=[BucketSpec(["*"])],
        ... )
        >>> # Explicit buckets:
        >>> flex_shard(
        ...     model,
        ...     mesh,
        ...     DataParallelMeshDims(shard="fsdp"),
        ...     shard_placement_fn=per_param_placements,
        ...     buckets=[BucketSpec(["attn.*"]), BucketSpec(["ffn.*"])],
        ... )
    Note:
        - Each bucket must contain only ``Shard(0)`` placements.
        - Each bucket must contain one original parameter dtype. Split mixed
          dtype parameters into separate buckets.
        - Parameters on meta device will have uninitialized storage
    """
    inputs = _prepare_flex_shard_inputs(
        module,
        mesh,
        dp_mesh_dims,
        shard_placement_fn,
        buckets,
    )

    storages, fqn_to_bucket_spec = _materialize_bucket_storages(
        module,
        inputs.named_params,
        inputs.bucket_assignments,
        buckets,
        inputs.param_placements,
        inputs.shard_mesh,
        inputs.device,
    )

    _attach_flex_shard_module_state(module, storages)

    module_param_map = _create_eager_param_states(
        module,
        storages,
        fqn_to_bucket_spec,
        inputs.device,
    )
    _register_module_param_accessors(module_param_map)

    # Reshard-after-forward: in eager mode, wrap each layer in checkpoint with
    # a selective policy that recomputes only collective ops (all-gather,
    # broadcast), saving compute ops to avoid redundant work.
    reshard_storages = [s for s in storages if s._reshard_after_forward]
    if reshard_storages:
        _apply_reshard_after_forward(module, reshard_storages)

    # Install batched all-gather hooks for eager mode when the storage layout
    # supports the batched Placement.unshard() path.
    _install_batched_allgather_hooks(storages, module_param_map)

    return module
