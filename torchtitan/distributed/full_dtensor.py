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

TP, CP, and EP shardings are handled by ``Module.parallelize(spmd_mesh)``
using config-based ``ShardingConfig``.
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


def _get_dp_mesh_axes(parallel_dims: ParallelDims) -> DataParallelMeshDims:
    """Build ``DataParallelMeshDims`` for dense (non-MoE) parameters.

    ``dp_shard`` is always included (force-kept-alive in the SPMD mesh
    even at size 1) so FSDP can pick the DP submesh out of the multi-axis
    SPMD mesh inside ``DeviceMesh._concatenate([dp_mesh, tp_mesh])``.
    """
    assert (
        parallel_dims.full_dtensor
    ), "_get_dp_mesh_axes is only meaningful under full_dtensor"
    shard_axes: list[str] = ["dp_shard"]
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


_DENSE_SPMD_AXES = ["dp_replicate", "dp_shard", "cp", "tp"]


def resolve_fsdp_mesh(
    model: nn.Module,
    parallel_dims: ParallelDims,
    full_dtensor: bool,
) -> tuple[DeviceMesh, DataParallelMeshDims | None]:
    """Select the dense FSDP mesh and optional DataParallelMeshDims.

    Under ``full_dtensor`` returns the full dense SPMD mesh and the DP
    axes from ``_get_dp_mesh_axes``. Otherwise returns the conventional
    1D ``dp_mesh`` and ``None``.
    """
    if full_dtensor:
        spmd_mesh = parallel_dims.get_enabled_mesh(_DENSE_SPMD_AXES)
        assert spmd_mesh is not None
        return spmd_mesh, _get_dp_mesh_axes(parallel_dims)
    dp_mesh = parallel_dims.get_enabled_mesh(["dp_replicate", "fsdp"])
    assert dp_mesh is not None
    return dp_mesh, None


def parallelize_inputs(
    parallel_dims: ParallelDims,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
) -> tuple[DTensor, DTensor, dict[str, Any]]:
    """Wrap ``inputs``, ``labels``, and tensor ``extra_kwargs`` as DTensors.

    Placements on the dense SPMD mesh: DP -> Shard(0), CP -> Shard(1),
    TP -> Replicate. Inputs are assumed already sharded; this only
    re-wraps via ``from_local``.
    """
    mesh = parallel_dims.get_enabled_mesh(_DENSE_SPMD_AXES)
    assert mesh is not None
    assert mesh.mesh_dim_names is not None
    input_shardings: dict[str, Placement] = {}
    if parallel_dims.dp_replicate_enabled:
        input_shardings["dp_replicate"] = Shard(0)
    if parallel_dims.dp_shard_enabled:
        input_shardings["dp_shard"] = Shard(0)
    if parallel_dims.cp_enabled:
        input_shardings["cp"] = Shard(1)
    placements: list[Placement] = [
        input_shardings.get(name, Replicate()) for name in mesh.mesh_dim_names
    ]

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
