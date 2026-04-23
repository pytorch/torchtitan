# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full DTensor infrastructure for SPMD-style parallelization.

When ``parallelism.full_dtensor`` is enabled, all model parameters,
buffers, and inputs become DTensors on a multi-dimensional SPMD mesh.
FSDP uses ``DataParallelMeshDims`` to identify which mesh dimensions
are data-parallel.

TP sharding is handled by ``Module.parallelize(spmd_mesh)`` using
config-based ``ShardingSpec``.
"""

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
        for m in model.modules():
            if isinstance(m, (ScaledDotProductAttention, VarlenAttention)):
                backend = (
                    "sdpa" if isinstance(m, ScaledDotProductAttention) else "varlen"
                )
                raise NotImplementedError(
                    f"full_dtensor + CP is not supported with {backend} attention. "
                    "After K/V all-gather on CP, Q and K/V have asymmetric sequence "
                    "lengths, which sdpa/varlen cannot handle with is_causal=True. "
                    "Use FlexAttention (e.g. --config llama3_debugmodel_flex_attn) "
                    "or disable CP."
                )


def get_dense_spmd_mesh(parallel_dims: ParallelDims) -> DeviceMesh:
    """Get the dense SPMD mesh for full DTensor parallelization.

    Uses the ``full_dtensor_dense`` global mesh which has separate ``dp_shard``
    and ``cp`` dimensions. Disabled dimensions (degree 1) are filtered out.

    The result is cached on the ``ParallelDims`` object so that all callers
    share the exact same ``DeviceMesh`` object — FSDP requires object identity.
    """
    if hasattr(parallel_dims, "_spmd_mesh"):
        return parallel_dims._spmd_mesh  # type: ignore[return-value]

    mesh_names = [
        n
        for n in ["dp_replicate", "dp_shard", "cp", "tp"]
        if parallel_dims.get_optional_mesh(n)
    ]
    assert mesh_names, "full_dtensor requires at least one mesh dimension"
    mesh = parallel_dims.get_mesh(mesh_names)
    parallel_dims._spmd_mesh = mesh  # type: ignore[attr-defined]
    return mesh


def get_dp_mesh_dims(parallel_dims: ParallelDims) -> DataParallelMeshDims:
    """Build DataParallelMeshDims for dense (non-MoE) parameters.

    Uses ``dp_shard`` and ``cp`` as separate shard dimensions rather than
    the flattened ``fsdp`` from the non-full-dtensor path.
    """
    shard_dims: list[str] = []
    if parallel_dims.dp_shard_enabled:
        shard_dims.append("dp_shard")
    if parallel_dims.cp_enabled:
        shard_dims.append("cp")

    if len(shard_dims) > 1:
        shard: str | tuple[str, ...] | None = tuple(shard_dims)
    elif shard_dims:
        shard = shard_dims[0]
    else:
        shard = None

    replicate: str | None = None
    if parallel_dims.dp_replicate_enabled:
        replicate = "dp_replicate"

    return DataParallelMeshDims(shard=shard, replicate=replicate)


def resolve_fsdp_mesh(
    model: nn.Module,
    parallel_dims: ParallelDims,
    full_dtensor: bool,
) -> tuple[DeviceMesh, DataParallelMeshDims | None]:
    """Select the FSDP mesh and optional DataParallelMeshDims.

    In full DTensor mode, returns the SPMD mesh and DataParallelMeshDims.
    In standard mode, returns the conventional dp_mesh and None.
    """
    if full_dtensor:
        spmd_mesh = get_dense_spmd_mesh(parallel_dims)
        _remove_sdpa_math_backend(model)
        dp_mesh_dims = get_dp_mesh_dims(parallel_dims)
        return spmd_mesh, dp_mesh_dims
    else:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        return parallel_dims.get_mesh(names), None


def _remove_sdpa_math_backend(model: nn.Module) -> None:
    """Remove MATH backend from SDPA modules.

    SDPA MATH backend decomposes into primitive ops that mix plain tensors
    with DTensors, causing errors. Flash and efficient backends have proper
    DTensor dispatch rules.
    """
    from torch.nn.attention import SDPBackend

    from torchtitan.models.common.attention import ScaledDotProductAttention
    from torchtitan.tools.logging import logger

    dropped = False
    for module in model.modules():
        if isinstance(module, ScaledDotProductAttention):
            if SDPBackend.MATH in module.sdpa_backends:
                module.sdpa_backends = [
                    b for b in module.sdpa_backends if b != SDPBackend.MATH
                ]
                dropped = True
    if dropped:
        logger.warning(
            "full_dtensor: dropped SDPBackend.MATH from SDPA modules. "
            "MATH decomposes to primitive ops that mix plain tensors with "
            "DTensors, which breaks dispatch. Using flash/efficient backends."
        )


def parallelize_inputs(
    parallel_dims: ParallelDims,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, torch.Tensor | None] | None = None,
) -> tuple[DTensor, DTensor]:
    """Convert inputs, labels, and extra kwargs to DTensors on the SPMD mesh.

    DP dims get Shard(0) (batch), CP gets Shard(1) (sequence), TP gets Replicate.
    Tensor values in extra_kwargs (e.g. positions) use the same placements.
    """
    mesh = get_dense_spmd_mesh(parallel_dims)
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

    # Convert extra_kwargs tensors (e.g. positions) to DTensors
    if extra_kwargs is not None:
        for key, value in extra_kwargs.items():
            if isinstance(value, torch.Tensor) and not isinstance(value, DTensor):
                extra_kwargs[key] = DTensor.from_local(value, mesh, placements)

    return DTensor.from_local(inputs, mesh, placements), DTensor.from_local(
        labels, mesh, placements
    )
