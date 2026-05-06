# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
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
    disable_active_parametrization,
    Placement,
)
from .reshard import (
    _apply_reshard_checkpoint,
    _get_storage_debug_fqn,
    _reshard_checkpoint_enabled,
)
from .storage import (
    _assign_params_to_buckets,
    _create_param_infos,
    _create_sharded_view,
    _write_params_to_dstorage,
    auto_buckets,
    BucketSpec,
    DStorage,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from .utils import (
    _validate_bucket_placements,
    _validate_eager_params,
    _validate_flex_shard_mesh,
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


@dataclass(frozen=True)
class FlexShardMeshInfo:
    """Mesh metadata for FlexShard's data-parallel shard view.

    ``dp_shard_mesh`` is the one-dimensional mesh used for FlexShard
    collectives and storage sharding.
    """

    dp_shard_mesh: DeviceMesh


def _get_submesh(mesh: DeviceMesh, names: tuple[str, ...]) -> DeviceMesh:
    """Return one mesh dim or flatten several named mesh dims."""
    if len(names) == 1:
        return mesh[names[0]]
    return mesh[names]._flatten("_".join(names))


def _get_flex_shard_mesh_info(
    mesh: DeviceMesh,
    dp_mesh_dims: DataParallelMeshDims,
) -> FlexShardMeshInfo:
    """Derive FlexShard's DP shard mesh from the input mesh."""
    _validate_flex_shard_mesh(mesh, dp_mesh_dims)

    assert mesh.mesh_dim_names is not None
    shard_names = dp_mesh_dims.shard_names
    return FlexShardMeshInfo(
        dp_shard_mesh=_get_submesh(mesh, shard_names),
    )


def _get_device_from_mesh(mesh: DeviceMesh) -> torch.device:
    """Return the current rank's device for ``mesh``."""
    if mesh.device_type == "cpu":
        return torch.device("cpu")
    if mesh.device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    try:
        device_module = torch.get_device_module(mesh.device_type)
    except (AttributeError, RuntimeError):
        return torch.device(mesh.device_type)
    return torch.device(mesh.device_type, device_module.current_device())


def _is_graph_capture_active() -> bool:
    """Return whether unsupported graph capture is active."""
    if torch.compiler.is_compiling():
        return True
    try:
        return torch._guards.TracingContext.try_get() is not None
    except AttributeError:
        return False


def _raise_graph_capture_unsupported() -> None:
    raise ValueError(
        "FlexShard currently supports eager execution only; torch.compile and "
        "graph capture are not supported yet."
    )


def _raise_missing_eager_batched_unshard(parametrization: nn.Module) -> None:
    param_fqn = getattr(parametrization, _PARAM_FQN_ATTR, "<unknown>")
    bucket_fqn = getattr(parametrization, _BUCKET_FQN_ATTR, None)
    bucket_msg = f" in bucket {bucket_fqn!r}" if bucket_fqn else ""
    raise RuntimeError(
        "FlexShard eager mode would fall back to per-parameter "
        f"_c10d_functional collectives for parameter {param_fqn!r}{bucket_msg}, "
        "but eager Shard(0) parameters must be served by a batched "
        "all-gather hook. This usually means the BucketSpec boundary does not "
        "match the module hook/checkpoint execution unit. Split the bucket to "
        "match forward module boundaries."
    )


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

    def _make_getter(pn, p):
        def getter(self):
            # In eager batched mode, _pre_gathered is set on the
            # parametrization by the batched all-gather pre_forward hook.
            pre = getattr(p, "_pre_gathered", None)
            if pre is not None:
                p._pre_gathered = None
                param_dtype = getattr(p, "param_dtype", None)
                reduce_dtype = getattr(p, "reduce_dtype", None)
                if getattr(p, _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR, False):
                    if param_dtype is not None or reduce_dtype is not None:
                        pre = _MixedPrecisionCast.apply(pre, param_dtype, reduce_dtype)
                    return pre

                if param_dtype is not None and pre.dtype != param_dtype:
                    pre = pre.to(param_dtype)
                unsharded = pre.detach().requires_grad_(True)
                if (
                    torch.is_grad_enabled()
                    and getattr(p, "_unsharded_for_reduce", None) is None
                ):
                    p._unsharded_for_reduce = unsharded
                return unsharded
            if _is_graph_capture_active():
                _raise_graph_capture_unsupported()
            if getattr(p, _REQUIRES_EAGER_BATCHED_UNSHARD_ATTR, False) and not getattr(
                p, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, False
            ):
                _raise_missing_eager_batched_unshard(p)
            return p(self._parameters[pn])

        return getter

    param_name_to_property = {
        param_name: property(_make_getter(param_name, param))
        for param_name, param in param_parametrizations.items()
    }
    module_cls = type(
        f"FlexShard{module.__class__.__name__}_{_wrap_class_counter}",
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
    mesh_info = _get_flex_shard_mesh_info(mesh, dp_mesh_dims)
    shard_mesh = mesh_info.dp_shard_mesh

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
    expected_fqns = {fqn for fqn, _ in named_params}
    actual_fqns = set(param_placements)
    missing_fqns = expected_fqns - actual_fqns
    extra_fqns = actual_fqns - expected_fqns
    if missing_fqns or extra_fqns:
        msg_parts = []
        if missing_fqns:
            msg_parts.append(f"missing placements for {sorted(missing_fqns)}")
        if extra_fqns:
            msg_parts.append(f"unexpected placements for {sorted(extra_fqns)}")
        raise ValueError(
            "shard_placement_fn must return placements for exactly the managed "
            f"parameters; {', '.join(msg_parts)}."
        )
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

    # Log bucket coverage
    if logger.isEnabledFor(logging.DEBUG):
        lines = ["flex_shard bucket coverage:"]
        total_params = 0
        for i, fqns in enumerate(bucket_assignments):
            patterns = buckets[i].patterns
            lines.append(f"  bucket {i} {patterns}: {len(fqns)} params")
            total_params += len(fqns)
        lines.append(f"  total: {total_params} params across {len(buckets)} buckets")
        logger.debug("\n".join(lines))

    # Per-bucket: create param_infos, byte buffer, replace params, create DStorage
    named_params_dict = dict(named_params)
    storages: list[DStorage] = []
    fqn_to_mp_policy: dict[str, MixedPrecisionPolicy | None] = {}
    fqn_to_offload_policy: dict[str, OffloadPolicy | None] = {}

    for bucket_idx, bucket_fqns in enumerate(bucket_assignments):
        if not bucket_fqns:
            continue

        # Extract per-bucket policies
        bucket_spec = buckets[bucket_idx]
        bucket_mp_policy = bucket_spec.mp_policy
        bucket_offload_policy = bucket_spec.offload_policy
        bucket_reshard_after_forward = bucket_spec.reshard_after_forward
        for fqn in bucket_fqns:
            fqn_to_mp_policy[fqn] = bucket_mp_policy
            fqn_to_offload_policy[fqn] = bucket_offload_policy

        bucket_named_params = [(fqn, named_params_dict[fqn]) for fqn in bucket_fqns]
        bucket_placements = {fqn: param_placements[fqn] for fqn in bucket_fqns}

        param_infos, total_bytes = _create_param_infos(
            bucket_named_params, mesh_info, bucket_placements
        )

        if bucket_offload_policy is not None:
            byte_storage = torch.empty(
                total_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=bucket_offload_policy.pin_memory,
            )
        else:
            byte_storage = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        _write_params_to_dstorage(
            byte_storage, bucket_named_params, param_infos, mesh_info
        )

        for fqn, info in param_infos.items():
            local_view = byte_storage[
                info.byte_offset : info.byte_offset
                + info.local_numel * info.dtype.itemsize
            ]
            typed_view = local_view.view(info.dtype).view(info.local_shape)
            new_param = nn.Parameter(typed_view, requires_grad=info.requires_grad)
            expected_param_device = (
                torch.device("cpu") if bucket_offload_policy is not None else device
            )
            if new_param.device != expected_param_device:
                raise AssertionError(
                    f"Expected sharded parameter {fqn!r} on "
                    f"{expected_param_device}, but got {new_param.device}"
                )
            _create_sharded_view(new_param, info, mesh_info)
            _set_param_on_module(module, fqn, new_param)

        storage = DStorage(
            byte_storage,
            param_infos,
            shard_mesh,
            total_bytes,
            module,
            reshard_after_forward=bucket_reshard_after_forward,
        )
        storages.append(storage)

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
            mp = fqn_to_mp_policy.get(fqn)
            offload = fqn_to_offload_policy.get(fqn)
            param_dtype = mp.param_dtype if mp else None
            reduce_dtype = mp.reduce_dtype if mp else None
            compute_device = torch.device(device) if offload is not None else None
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
