# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full DTensor infrastructure for SPMD-style parallelization.

When ``parallelism.full_dtensor`` is enabled, all model parameters,
buffers, and inputs become DTensors on a multi-dimensional SPMD mesh.
FSDP uses ``DataParallelMeshDims`` to identify which mesh axes
are data-parallel.

TP and CP shardings are handled by ``Module.parallelize(spmd_mesh)`` using
config-based ``ShardingConfig``.
"""

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import DataParallelMeshDims
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import Placement

from torchtitan.distributed.parallel_dims import ParallelDims


def validate_config(
    parallel_dims: ParallelDims,
    model: nn.Module,
) -> None:
    """Validate that the current configuration is compatible with full DTensor.

    Walks ``model`` to discover the actual attention modules in use and
    raises ``NotImplementedError`` with a clear message if incompatible.
    """
    from torchtitan.models.common.attention import (
        ScaledDotProductAttention,
        VarlenAttention,
    )

    if parallel_dims.ep_enabled:
        raise NotImplementedError(
            "full_dtensor is not supported with Expert Parallel. "
            "Disable EP or disable full_dtensor."
        )

    if parallel_dims.cp_enabled:
        if any(
            isinstance(m, (ScaledDotProductAttention, VarlenAttention))
            for m in model.modules()
        ):
            raise NotImplementedError(
                "full_dtensor + CP is not supported with "
                "ScaledDotProductAttention or VarlenAttention. "
                "Use FlexAttention + CP or disable CP."
            )


def get_dp_mesh_axes(parallel_dims: ParallelDims) -> DataParallelMeshDims:
    """Build ``DataParallelMeshDims`` for dense (non-MoE) parameters.

    Uses ``dp_shard`` and ``cp`` as separate shard axes rather than
    the flattened ``fsdp`` axis from the non-full-dtensor path. The
    return type still spells ``Dims`` because that's the upstream class
    name.
    """
    shard_axes: list[str] = []
    if parallel_dims.dp_shard_enabled:
        shard_axes.append("dp_shard")
    if parallel_dims.cp_enabled:
        shard_axes.append("cp")

    if len(shard_axes) > 1:
        shard: str | tuple[str, ...] | None = tuple(shard_axes)
    elif shard_axes:
        shard = shard_axes[0]
    else:
        shard = None

    replicate = "dp_replicate" if parallel_dims.dp_replicate_enabled else None

    return DataParallelMeshDims(shard=shard, replicate=replicate)


def _dense_spmd_axes(parallel_dims: ParallelDims) -> list[str]:
    """Canonical dense SPMD axis list, intersected with enabled axes."""
    return [
        a
        for a in ("dp_replicate", "dp_shard", "cp", "tp")
        if parallel_dims.get_optional_mesh(a) is not None
    ]


def resolve_fsdp_mesh(
    model: nn.Module,
    parallel_dims: ParallelDims,
    full_dtensor: bool,
) -> tuple[DeviceMesh, DataParallelMeshDims | None]:
    """Select the FSDP mesh and optional DataParallelMeshDims.

    In full DTensor mode, returns the SPMD mesh and DataParallelMeshDims.
    In non-full DTensor mode, returns the conventional dp_mesh and None.
    """
    if full_dtensor:
        spmd_mesh = parallel_dims.get_mesh(_dense_spmd_axes(parallel_dims))
        dp_mesh_axes = get_dp_mesh_axes(parallel_dims)
        return spmd_mesh, dp_mesh_axes
    else:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        return parallel_dims.get_mesh(names), None


def parallelize_inputs(
    parallel_dims: ParallelDims,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
) -> tuple[DTensor, DTensor, dict[str, Any]]:
    """Convert inputs, labels, and extra kwargs to DTensors on the SPMD mesh.

    DP axes get Shard(0) (batch), CP gets Shard(1) (sequence), TP gets Replicate.
    Tensor values in extra_kwargs (e.g. positions) use the same placements.

    NOTE: This API assumes the inputs are already sharded; it only converts
    the class from ``torch.Tensor`` to ``DTensor`` via ``DTensor.from_local``.
    """
    mesh = parallel_dims.get_mesh(_dense_spmd_axes(parallel_dims))
    placements: list[Placement] = []
    if parallel_dims.dp_replicate_enabled:
        placements.append(Shard(0))
    if parallel_dims.dp_shard_enabled:
        placements.append(Shard(0))
    if parallel_dims.cp_enabled:
        placements.append(Shard(1))
    if parallel_dims.tp_enabled:
        placements.append(Replicate())

    assert mesh.ndim == len(placements)

    # Convert extra_kwargs tensors (e.g. positions) to DTensors. A new dict is
    # returned so the side effect is visible in the signature, not hidden.
    new_extra_kwargs: dict[str, Any] = {
        k: (
            DTensor.from_local(v, mesh, placements)
            if isinstance(v, torch.Tensor) and not isinstance(v, DTensor)
            else v
        )
        for k, v in extra_kwargs.items()
    }

    return (
        DTensor.from_local(inputs, mesh, placements),
        DTensor.from_local(labels, mesh, placements),
        new_extra_kwargs,
    )
