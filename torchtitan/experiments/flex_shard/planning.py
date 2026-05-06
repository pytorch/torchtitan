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

from .module_wrapping import _check_not_already_flex_sharded
from .storage import _assign_params_to_buckets, BucketSpec
from .utils import (
    _get_device_from_mesh,
    _get_managed_named_params,
    _get_dp_shard_mesh,
    _validate_bucket_placements,
    _validate_eager_params,
    _validate_placements,
)

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .placements import Placement


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
