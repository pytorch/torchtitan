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

from .placements import FlatShard, Owned, Placement, RaggedShard, Shard

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


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
