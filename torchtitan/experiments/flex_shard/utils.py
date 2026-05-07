# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed.fsdp import DataParallelMeshDims

from .state import (
    _BUCKET_FQN_ATTR,
    _DSTORAGE_ATTR,
    _EAGER_BATCHED_HOOK_REGISTERED_ATTR,
    _PARAM_FQN_ATTR,
)

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .placements import Placement
    from .storage import DStorage


def _with_fqn(label: str, fqn: str | None) -> str:
    """Append a module/bucket FQN to profiler labels, matching FSDP style."""
    if fqn:
        return f"{label} ({fqn})"
    return label


def _module_path_common_prefix(paths: list[str]) -> str:
    """Return the common module path prefix for parameter-owner module paths."""
    if not paths:
        return ""
    common_parts = paths[0].split(".") if paths[0] else []
    for path in paths[1:]:
        parts = path.split(".") if path else []
        limit = min(len(common_parts), len(parts))
        i = 0
        while i < limit and common_parts[i] == parts[i]:
            i += 1
        common_parts = common_parts[:i]
        if not common_parts:
            break
    return ".".join(common_parts)


def _strip_checkpoint_wrapped_module_path(path: str) -> str:
    """Remove CheckpointWrapper internals from a dotted module path."""
    return ".".join(
        part for part in path.split(".") if part != "_checkpoint_wrapped_module"
    )


def _top_level_owner_path(module: nn.Module, owner_path: str) -> str:
    """Choose the outer module to checkpoint for a parameter owner path."""
    parts = owner_path.split(".")
    if not parts or not parts[0]:
        return ""
    child = getattr(module, parts[0])
    if (
        isinstance(child, (nn.ModuleDict, nn.ModuleList, nn.Sequential))
        and len(parts) > 1
    ):
        return ".".join(parts[:2])
    return parts[0]


def _get_storage_debug_fqn(storage: DStorage) -> str | None:
    """Return a concise module/bucket FQN for profiler annotations."""
    owner_paths = sorted(
        {
            _strip_checkpoint_wrapped_module_path(".".join(fqn.split(".")[:-1]))
            for fqn in storage._param_infos
        }
    )
    if not owner_paths:
        return None
    common = _module_path_common_prefix(owner_paths)
    if common:
        return common
    top_level_paths = sorted(
        {
            _top_level_owner_path(storage._module, owner_path)
            for owner_path in owner_paths
        }
    )
    top_level_paths = [path for path in top_level_paths if path]
    if not top_level_paths:
        return None
    return ", ".join(top_level_paths)


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


def _raise_missing_eager_batched_unshard(param_state: Any) -> None:
    param_fqn = getattr(param_state, _PARAM_FQN_ATTR, "<unknown>")
    bucket_fqn = getattr(param_state, _BUCKET_FQN_ATTR, None)
    hook_registered = getattr(param_state, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, False)
    bucket_msg = f" in bucket {bucket_fqn!r}" if bucket_fqn else ""
    hook_msg = (
        " The bucket hook was registered but did not run before parameter access."
        if hook_registered
        else " No bucket hook was registered for this parameter."
    )
    raise RuntimeError(
        "FlexShard eager mode requires pre-gathered parameter data from a "
        f"batched all-gather hook for parameter {param_fqn!r}{bucket_msg}."
        f"{hook_msg} This usually means the parameter was accessed outside "
        "the hooked module forward, or the BucketSpec boundary does not match "
        "the module hook/checkpoint execution unit. Split the bucket to match "
        "forward module boundaries."
    )


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


def _validate_flex_shard_mesh(
    mesh: DeviceMesh,
    dp_mesh_dims: DataParallelMeshDims,
) -> None:
    """Validate mesh inputs for FlexShard eager mode."""
    if dp_mesh_dims.shard is None:
        raise ValueError("flex_shard requires dp_mesh_dims.shard to be set")
    if dp_mesh_dims.replicate is not None:
        raise NotImplementedError(
            "flex_shard eager mode does not yet support dp_mesh_dims.replicate"
        )
    if mesh.mesh_dim_names is None:
        raise ValueError("mesh must have mesh_dim_names when dp_mesh_dims is provided")

    mesh_names = tuple(mesh.mesh_dim_names)
    axis_names = dp_mesh_dims.shard_names + dp_mesh_dims.replicate_names
    if len(set(axis_names)) != len(axis_names):
        raise ValueError(
            f"dp_mesh_dims contains duplicate mesh axis names: {axis_names}"
        )
    for name in axis_names:
        if name not in mesh_names:
            raise ValueError(
                f"Mesh axis name {name!r} not found in mesh.mesh_dim_names {mesh_names}"
            )


def _get_submesh(mesh: DeviceMesh, names: tuple[str, ...]) -> DeviceMesh:
    """Return one mesh axis or flatten several named mesh axes."""
    if len(names) == 1:
        return mesh[names[0]]
    return mesh[names]._flatten("_".join(names))


def _get_dp_shard_mesh(
    mesh: DeviceMesh,
    dp_mesh_dims: DataParallelMeshDims,
) -> DeviceMesh:
    """Derive FlexShard's DP shard mesh from the input mesh."""
    _validate_flex_shard_mesh(mesh, dp_mesh_dims)

    assert mesh.mesh_dim_names is not None
    return _get_submesh(mesh, dp_mesh_dims.shard_names)


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


def _validate_eager_params(
    named_params: list[tuple[str, nn.Parameter]],
    expected_device: torch.device | None = None,
) -> None:
    """Validate parameters supported by the eager-only path."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None

    for fqn, param in named_params:
        if DTensor is not None and isinstance(param, DTensor):
            raise ValueError(
                "FlexShard eager mode expects plain parameters; "
                f"{fqn!r} is a DTensor. DTensor composition is not supported yet."
            )
        if (
            expected_device is not None
            and param.device.type != "meta"
            and param.device != expected_device
        ):
            raise ValueError(
                f"Parameter {fqn!r} is on {param.device}, but FlexShard expected "
                f"{expected_device}. Move the module to the target mesh device "
                "before calling flex_shard()."
            )


def _validate_placements(
    param_placements: dict[str, tuple[Placement, ...]],
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> None:
    """Validate that placements are compatible with eager FlexShard."""
    from .placements import FlatShard, Owned, RaggedShard, Shard

    param_dict = dict(named_params)
    expected_fqns = set(param_dict)
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

    world_size = mesh.size()

    for fqn, placements in param_placements.items():
        for placement in placements:
            if isinstance(placement, Owned):
                if placement.owner_rank >= mesh.size():
                    raise ValueError(
                        f"Parameter {fqn!r} uses Owned({placement.owner_rank}) "
                        f"but world_size is {mesh.size()}."
                    )
            if isinstance(placement, RaggedShard):
                if len(placement.local_units) != world_size:
                    raise ValueError(
                        f"Parameter {fqn!r} uses RaggedShard with "
                        f"{len(placement.local_units)} local_units but "
                        f"world_size is {world_size}."
                    )
            if isinstance(placement, Shard):
                param = param_dict[fqn]
                if placement.dim >= param.ndim:
                    raise ValueError(
                        f"Parameter {fqn!r} has {param.ndim} dimensions but "
                        f"Shard(dim={placement.dim}) is out of range."
                    )
            if isinstance(placement, FlatShard):
                param = param_dict[fqn]
                if param.numel() == 0:
                    raise ValueError(
                        f"Parameter {fqn!r} has 0 elements, cannot apply FlatShard."
                    )


def _validate_bucket_placements(
    bucket_assignments: list[list[str]],
    param_placements: dict[str, tuple[Placement, ...]],
    buckets: list[Any],
    named_params: list[tuple[str, nn.Parameter]],
) -> None:
    """Validate minimal eager bucket constraints."""
    from .placements import Shard

    param_dict = dict(named_params)
    for bucket_idx, fqns in enumerate(bucket_assignments):
        if not fqns:
            continue
        reference_dtype = param_dict[fqns[0]].dtype
        for fqn in fqns:
            placements = param_placements[fqn]
            if len(placements) != 1:
                raise ValueError(
                    f"Bucket {bucket_idx} "
                    f"{buckets[bucket_idx].patterns} "
                    f"parameter {fqn!r} has {len(placements)} placements. "
                    "FlexShard eager mode currently supports exactly one "
                    "Shard(0) placement per parameter."
                )
            placement = placements[0]
            if not isinstance(placement, Shard) or placement.dim != 0:
                raise ValueError(
                    f"Bucket {bucket_idx} "
                    f"{buckets[bucket_idx].patterns} "
                    f"parameter {fqn!r} uses {placement!r}. "
                    "FlexShard eager mode currently supports only Shard(0) "
                    "placements."
                )

            dtype = param_dict[fqn].dtype
            if dtype != reference_dtype:
                raise ValueError(
                    f"Bucket {bucket_idx} "
                    f"{buckets[bucket_idx].patterns} "
                    f"has mixed parameter dtypes: {fqns[0]!r} uses "
                    f"{reference_dtype} but {fqn!r} uses {dtype}. "
                    "All params in a FlexShard storage must share the same "
                    "dtype."
                )
