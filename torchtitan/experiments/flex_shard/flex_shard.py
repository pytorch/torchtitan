# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed.fsdp import DataParallelMeshDims

from .eager_runtime import (
    _install_batched_allgather_hooks,
    _storage_requires_eager_batched_unshard,
    _storage_uses_eager_autograd_unshard,
    EagerAllGatherContext,
)
from .metadata import (
    _BUCKET_FQN_ATTR,
    _DSTORAGE_ATTR,
    _DSTORAGES_ATTR,
    _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR,
    _EAGER_BATCHED_HOOK_REGISTERED_ATTR,
    _EAGER_COMM_CONTEXTS_ATTR,
    _PARAM_FQN_ATTR,
    _REQUIRES_EAGER_BATCHED_UNSHARD_ATTR,
    get_global_shape,
    get_placements,
    is_flex_shard_param,
    set_sharding_info,
)
from .placements import (
    _MixedPrecisionCast,
    Placement,
)
from .reshard import (
    _apply_reshard_checkpoint,
    _get_storage_debug_fqn,
    _reshard_checkpoint_enabled,
)
from .storage import (
    _assign_params_to_buckets,
    _materialize_bucket_storages,
    auto_buckets,
    BucketSpec,
    DStorage,
    MixedPrecisionPolicy,
)
from .utils import (
    _get_device_from_mesh,
    _get_managed_named_params,
    _get_dp_shard_mesh,
    _is_graph_capture_active,
    _raise_graph_capture_unsupported,
    _raise_missing_eager_batched_unshard,
    disable_active_parametrization,
    _validate_bucket_placements,
    _validate_eager_params,
    _validate_placements,
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


_parametrized_module_class_counter = 0


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
    global _parametrized_module_class_counter
    _parametrized_module_class_counter += 1

    def _make_flex_shard_param_getter(param_name, parametrization):
        def get_flex_shard_param(self):
            # In eager batched mode, _pre_gathered is set on the
            # parametrization by the batched all-gather pre_forward hook.
            pre = getattr(parametrization, "_pre_gathered", None)
            if pre is not None:
                parametrization._pre_gathered = None
                param_dtype = getattr(parametrization, "param_dtype", None)
                reduce_dtype = getattr(parametrization, "reduce_dtype", None)
                if getattr(parametrization, _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR, False):
                    if param_dtype is not None or reduce_dtype is not None:
                        pre = _MixedPrecisionCast.apply(pre, param_dtype, reduce_dtype)
                    return pre

                if param_dtype is not None and pre.dtype != param_dtype:
                    pre = pre.to(param_dtype)
                unsharded = pre.detach().requires_grad_(True)
                if (
                    torch.is_grad_enabled()
                    and getattr(parametrization, "_unsharded_for_reduce", None) is None
                ):
                    parametrization._unsharded_for_reduce = unsharded
                return unsharded
            if _is_graph_capture_active():
                _raise_graph_capture_unsupported()
            if getattr(
                parametrization, _REQUIRES_EAGER_BATCHED_UNSHARD_ATTR, False
            ) and not getattr(
                parametrization, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, False
            ):
                _raise_missing_eager_batched_unshard(parametrization)
            return parametrization(self._parameters[param_name])

        return get_flex_shard_param

    param_name_to_property = {
        param_name: property(_make_flex_shard_param_getter(param_name, param))
        for param_name, param in param_parametrizations.items()
    }
    module_cls = type(
        f"FlexShard{module.__class__.__name__}_{_parametrized_module_class_counter}",
        (module.__class__,),
        param_name_to_property,
    )
    module.__class__ = module_cls
    sys.modules[module_cls.__module__].__dict__[module_cls.__name__] = module_cls


class FlexShardModule:
    """Mixin added to modules after flex_shard()."""

    @property
    def dstorage(self) -> DStorage:
        """First (or only) DStorage. For multi-bucket, use .dstorages."""
        return getattr(self, _DSTORAGE_ATTR)

    @property
    def dstorages(self) -> list:
        """All DStorage instances (one per bucket)."""
        return getattr(self, _DSTORAGES_ATTR)

    @property
    def eager_comm_contexts(self) -> dict[torch.device, EagerAllGatherContext]:
        """Root-owned eager communication contexts keyed by CUDA device."""
        return getattr(self, _EAGER_COMM_CONTEXTS_ATTR)


PlacementFn = Callable[
    [list[tuple[str, nn.Parameter]], "DeviceMesh"],
    dict[str, tuple[Placement, ...]],
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
    shard_mesh = _get_dp_shard_mesh(mesh, dp_mesh_dims)

    # Determine device - use meta only if all params are meta, otherwise use mesh device.
    all_params_meta = all(param.device.type == "meta" for _, param in named_params)
    if all_params_meta:
        device = torch.device("meta")
    else:
        device = _get_device_from_mesh(shard_mesh)
    _validate_eager_params(
        named_params,
        expected_device=None if all_params_meta else device,
    )

    # Resolve placements for all params
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

    storages, fqn_to_bucket_spec = _materialize_bucket_storages(
        module,
        named_params,
        bucket_assignments,
        buckets,
        param_placements,
        shard_mesh,
        device,
    )

    # Store DStorages on module
    setattr(module, _DSTORAGES_ATTR, storages)
    setattr(module, _DSTORAGE_ATTR, storages[0] if storages else None)
    setattr(module, _EAGER_COMM_CONTEXTS_ATTR, {})

    # Change module class to include FlexShardModule mixin
    cls = type(module)
    if not issubclass(cls, FlexShardModule):
        module.__class__ = type(cls.__name__, (cls, FlexShardModule), {})

    # Register property-based parametrization.
    group_name = shard_mesh.get_group().group_name
    world_size = shard_mesh.size()

    # Group parametrizations by leaf module (across all buckets)
    module_param_map: dict[nn.Module, dict[str, nn.Module]] = {}

    for s in storages:
        requires_eager_batched_unshard = _storage_requires_eager_batched_unshard(s)
        uses_eager_autograd_unshard = _storage_uses_eager_autograd_unshard(s)
        bucket_fqn = _get_storage_debug_fqn(s)
        for fqn, info in s._param_infos.items():
            bucket_spec = fqn_to_bucket_spec[fqn]
            mp_policy = bucket_spec.mp_policy
            param_dtype = mp_policy.param_dtype if mp_policy else None
            reduce_dtype = mp_policy.reduce_dtype if mp_policy else None
            compute_device = (
                torch.device(device) if bucket_spec.offload_policy is not None else None
            )
            placement = info.placements[0]
            p = placement.create_parametrization(
                info,
                group_name,
                world_size,
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                compute_device=compute_device,
            )

            setattr(
                p,
                _REQUIRES_EAGER_BATCHED_UNSHARD_ATTR,
                requires_eager_batched_unshard,
            )
            setattr(
                p,
                _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR,
                uses_eager_autograd_unshard,
            )
            setattr(p, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, False)
            setattr(p, _PARAM_FQN_ATTR, fqn)
            setattr(p, _BUCKET_FQN_ATTR, bucket_fqn)

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
