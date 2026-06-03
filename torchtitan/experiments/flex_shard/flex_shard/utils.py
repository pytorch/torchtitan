# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_storage import BucketParamFQNsByIndex, ShardedBucketStorage
    from .placement_contract import Placement


def _with_fqn(label: str, fqn: str | None) -> str:
    """Append a module/bucket FQN to profiler labels, matching FSDP style."""
    if fqn:
        return f"{label} ({fqn})"
    return label


def _record_function_if_eager(
    label: str,
    fqn: str | None,
) -> AbstractContextManager[Any]:
    """Return a profiler range in eager and a no-op context during compile."""
    if torch.compiler.is_compiling():
        return nullcontext()
    return torch.profiler.record_function(_with_fqn(label, fqn))


def _record_comm_if_eager(
    label: str,
    fqn: str | None,
) -> AbstractContextManager[Any]:
    """Return a c10d profiler range in eager and a no-op during compile."""
    if torch.compiler.is_compiling():
        return nullcontext()
    return dist.record_comm(_with_fqn(label, fqn))


def _get_single_placement(placements: tuple[Placement, ...]) -> Placement:
    """Return the only placement supported by the minimal eager path."""
    if len(placements) != 1:
        raise ValueError(
            "FlexShard eager mode currently supports exactly one placement "
            f"per parameter, but got {len(placements)} placements."
        )
    return placements[0]


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


def _get_bucket_storage_debug_fqn(
    bucket_storage: ShardedBucketStorage,
) -> str | None:
    """Return a concise module/bucket FQN for profiler annotations."""
    owner_paths = sorted(
        {
            _strip_checkpoint_wrapped_module_path(".".join(fqn.split(".")[:-1]))
            for fqn in bucket_storage._param_infos
        }
    )
    if not owner_paths:
        return None
    common = _module_path_common_prefix(owner_paths)
    if common:
        return common
    top_level_paths = sorted(
        {
            _top_level_owner_path(bucket_storage._module, owner_path)
            for owner_path in owner_paths
        }
    )
    top_level_paths = [path for path in top_level_paths if path]
    if not top_level_paths:
        return None
    return ", ".join(top_level_paths)


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
    Collect parameters managed by this root-level flex_shard() call.
    """
    managed_params: list[tuple[str, nn.Parameter]] = []

    seen_params: dict[int, str] = {}

    # Use remove_duplicate=False so shared parameters are rejected instead of
    # leaving one alias unmanaged.
    for fqn, param in module.named_parameters(remove_duplicate=False):
        param_id = id(param)
        if param_id in seen_params:
            raise ValueError(
                "FlexShard eager mode does not support shared parameters; "
                f"{fqn!r} shares storage with {seen_params[param_id]!r}."
            )
        seen_params[param_id] = fqn
        managed_params.append((fqn, param))

    return managed_params


def _validate_flex_shard_mesh(mesh: DeviceMesh) -> None:
    """Validate mesh inputs for FlexShard eager mode."""
    if mesh.ndim != 1:
        raise ValueError(
            f"flex_shard requires a 1D DeviceMesh, but got {mesh.ndim}D mesh"
        )
    if mesh.device_type != "cuda":
        raise NotImplementedError(
            "FlexShard runtime requires a CUDA DeviceMesh. CPU bucket runtime "
            "is not supported; CPU offload will be added separately."
        )


def _get_device_from_mesh(mesh: DeviceMesh) -> torch.device:
    """Return the current rank's device for ``mesh``."""
    if mesh.device_type == "cpu":
        return torch.device("cpu")
    device_module = _get_device_handle(mesh.device_type)
    if device_module is None:
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
            "BucketSpec.placement_fn must return placements for exactly the "
            f"provided parameters; {', '.join(msg_parts)}."
        )

    from .placement_contract import Placement

    for fqn, placements in param_placements.items():
        placement = _get_single_placement(placements)
        if not isinstance(placement, Placement):
            raise TypeError(
                "BucketSpec.placement_fn must return Placement instances, but "
                f"{fqn!r} uses {type(placement).__name__}."
            )


def _validate_bucket_uniform_dtype_and_placement(
    bucket_assignments: BucketParamFQNsByIndex,
    param_placements: dict[str, tuple[Placement, ...]],
    buckets: list[Any],
    named_params: list[tuple[str, nn.Parameter]],
) -> None:
    """Validate minimal eager bucket constraints."""
    param_dict = dict(named_params)
    for bucket_idx, fqns in enumerate(bucket_assignments):
        if not fqns:
            continue
        reference_dtype = param_dict[fqns[0]].dtype
        reference_placements = param_placements[fqns[0]]
        for fqn in fqns:
            dtype = param_dict[fqn].dtype
            if dtype != reference_dtype:
                raise ValueError(
                    f"Bucket {bucket_idx} "
                    f"{buckets[bucket_idx].patterns} "
                    f"has mixed parameter dtypes: {fqns[0]!r} uses "
                    f"{reference_dtype} but {fqn!r} uses {dtype}. "
                    "All params in a FlexShard bucket storage must share the same "
                    "dtype."
                )
            placements = param_placements[fqn]
            if placements != reference_placements:
                raise ValueError(
                    f"Bucket {bucket_idx} "
                    f"{buckets[bucket_idx].patterns} "
                    f"has mixed placements: {fqns[0]!r} uses "
                    f"{reference_placements!r} but {fqn!r} uses "
                    f"{placements!r}. All params in a FlexShard bucket must "
                    "share the same placement tuple because bucket collectives "
                    "use one placement layout. Split parameters with different "
                    "placements into separate buckets."
                )
