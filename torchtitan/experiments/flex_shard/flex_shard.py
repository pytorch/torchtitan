# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch.distributed.fsdp import DataParallelMeshDims

from .eager_runtime import (
    _create_eager_parametrizations,
    _install_batched_allgather_hooks,
)
from .metadata import (
    get_global_shape,
    get_placements,
    is_flex_shard_param,
    set_sharding_info,
)
from .module_wrapping import (
    _attach_flex_shard_module_state,
    _register_module_parametrizations,
    FlexShardModule,
)
from .planning import _prepare_flex_shard_plan, PlacementFn
from .reshard import (
    _apply_reshard_checkpoint,
    _reshard_checkpoint_enabled,
)
from .storage import (
    _materialize_bucket_storages,
    auto_buckets,
    BucketSpec,
    MixedPrecisionPolicy,
)
from .utils import (
    disable_active_parametrization,
)

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


__all__ = [
    "auto_buckets",
    "BucketSpec",
    "disable_active_parametrization",
    "flex_shard",
    "get_global_shape",
    "get_placements",
    "is_flex_shard_param",
    "MixedPrecisionPolicy",
    "set_sharding_info",
]


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
    5. Registers property-based parametrization for eager parameter access
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
            ``[BucketSpec(["*"])]`` for a single whole-module bucket or
            ``auto_buckets()`` to generate one bucket per direct child module.

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
        >>> # Auto buckets (one per child):
        >>> flex_shard(
        ...     model,
        ...     mesh,
        ...     DataParallelMeshDims(shard="fsdp"),
        ...     shard_placement_fn=per_param_placements,
        ...     buckets=auto_buckets(model),
        ... )

    Note:
        - Each bucket must contain only ``Shard(0)`` placements.
        - Each bucket must contain one original parameter dtype. Split mixed
          dtype parameters into separate buckets.
        - Parameters on meta device will have uninitialized storage
    """
    plan = _prepare_flex_shard_plan(
        module,
        mesh,
        dp_mesh_dims,
        shard_placement_fn,
        buckets,
    )

    storages, fqn_to_bucket_spec = _materialize_bucket_storages(
        module,
        plan.named_params,
        plan.bucket_assignments,
        buckets,
        plan.param_placements,
        plan.shard_mesh,
        plan.device,
    )

    _attach_flex_shard_module_state(module, storages)

    group_name = plan.shard_mesh.get_group().group_name
    world_size = plan.shard_mesh.size()
    module_param_map = _create_eager_parametrizations(
        module,
        storages,
        fqn_to_bucket_spec,
        group_name,
        world_size,
        plan.device,
    )
    _register_module_parametrizations(module_param_map)

    # Reshard-after-forward: in eager mode, wrap each layer in checkpoint with
    # a selective policy that recomputes only collective ops (all-gather,
    # broadcast), saving compute ops to avoid redundant work.
    reshard_storages = [s for s in storages if s._reshard_after_forward]
    if _reshard_checkpoint_enabled.get() and reshard_storages:
        _apply_reshard_checkpoint(module, reshard_storages)

    # Install batched all-gather hooks for eager mode when the storage layout
    # supports the batched Placement.unshard() path.
    _install_batched_allgather_hooks(storages, module_param_map)

    return module
