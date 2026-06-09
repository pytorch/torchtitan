# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full DTensor infrastructure for multi-axis parallelization.

When ``parallelism.spmd_backend == "full_dtensor"`` is enabled, all model parameters,
buffers, and inputs become DTensors on a multi-dimensional dense mesh.
FSDP uses ``DataParallelMeshDims`` to identify which mesh axes
are data-parallel.

TP, CP, and EP shardings are handled by ``Module.parallelize(parallel_dims)``
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
    """Validate that the current configuration is compatible with multi-axis backends.

    Walks ``model`` to discover the actual attention modules in use and
    raises ``NotImplementedError`` with a clear message if incompatible.
    """
    from torchtitan.models.common.attention import (
        ScaledDotProductAttention,
        VarlenAttention,
    )

    if parallel_dims.cp_enabled:
        if any(
            isinstance(m, (ScaledDotProductAttention, VarlenAttention))
            for m in model.modules()
        ):
            raise NotImplementedError(
                f"{parallel_dims.spmd_backend} + CP is not supported with "
                "ScaledDotProductAttention or VarlenAttention. "
                "Use FlexAttention + CP or disable CP."
            )


def _get_dp_mesh_axes(parallel_dims: ParallelDims) -> DataParallelMeshDims:
    """Build ``DataParallelMeshDims`` for dense (non-MoE) parameters.

    ``dp_shard`` is always included (force-kept-alive in the dense storage mesh
    even at size 1) so FSDP can pick the DP submesh out of the multi-axis
    storage mesh inside ``DeviceMesh._concatenate([dp_mesh, tp_mesh])``.
    """
    assert parallel_dims.spmd_backend in (
        "full_dtensor",
        "spmd_types",
    ), "_get_dp_mesh_axes is only meaningful under full_dtensor or spmd_types"
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


_DENSE_STORAGE_AXES = ["dp_replicate", "dp_shard", "cp", "tp"]
_SPARSE_STORAGE_AXES = ["dp_replicate", "efsdp", "ep"]


def _get_sparse_dp_mesh_axes(parallel_dims: ParallelDims) -> DataParallelMeshDims:
    """Build ``DataParallelMeshDims`` for routed-expert (sparse) parameters.

    The FSDP axis is ``efsdp`` and ``dp_replicate`` is shared with the dense path.
    """
    shard_axis = "efsdp" if parallel_dims.ep_enabled else None
    replicate_axis = "dp_replicate" if parallel_dims.dp_replicate_enabled else None
    return DataParallelMeshDims(shard=shard_axis, replicate=replicate_axis)


def resolve_fsdp_mesh(
    parallel_dims: ParallelDims,
) -> tuple[DeviceMesh, DataParallelMeshDims]:
    """Select the dense storage mesh and DataParallelMeshDims."""
    assert parallel_dims.spmd_backend in (
        "full_dtensor",
        "spmd_types",
    ), "resolve_fsdp_mesh is only meaningful under full_dtensor or spmd_types"
    storage_mesh = parallel_dims.get_activated_mesh(_DENSE_STORAGE_AXES)
    assert storage_mesh is not None
    return storage_mesh, _get_dp_mesh_axes(parallel_dims)


def resolve_sparse_fsdp_mesh(
    parallel_dims: ParallelDims,
) -> tuple[DeviceMesh | None, DataParallelMeshDims | None]:
    """Sparse counterpart of ``resolve_fsdp_mesh`` for routed experts.

    Returns ``(None, None)`` when EP is disabled; otherwise the sparse
    storage mesh + sparse DP axes.
    """
    assert parallel_dims.spmd_backend in (
        "full_dtensor",
        "spmd_types",
    ), "resolve_sparse_fsdp_mesh is only meaningful under full_dtensor or spmd_types"
    if not parallel_dims.ep_enabled:
        return None, None
    sparse_mesh = parallel_dims.get_activated_mesh(_SPARSE_STORAGE_AXES)
    assert sparse_mesh is not None
    return sparse_mesh, _get_sparse_dp_mesh_axes(parallel_dims)


def parallelize_inputs(
    parallel_dims: ParallelDims,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
) -> tuple[DTensor, DTensor, dict[str, Any]]:
    """Wrap ``inputs``, ``labels``, and tensor ``extra_kwargs`` as DTensors.

    Placements on the dense storage mesh: DP -> Shard(0), CP -> Shard(1),
    TP -> Replicate. Inputs are assumed already sharded; this only
    re-wraps via ``from_local``.
    """
    mesh = parallel_dims.get_activated_mesh(_DENSE_STORAGE_AXES)
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
