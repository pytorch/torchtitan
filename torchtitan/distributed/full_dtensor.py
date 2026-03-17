# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file contains utility functions to parallelize models with full dtensor.
# We will eventually replace the existing functions with these or merge them.

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import DataParallelMeshDims
from torch.distributed.tensor import distribute_module, DTensor
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.tools.logging import logger


def validate_config(parallel_dims: ParallelDims, model_config: Any) -> None:
    """Validate that the current configuration is compatible with full DTensor.

    Raises NotImplementedError with a clear message if incompatible.
    """
    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "full_dtensor is not supported with Pipeline Parallel. "
            "Disable PP or disable full_dtensor."
        )

    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "full_dtensor is not yet supported with Tensor Parallel. "
            "Use full_dtensor with FSDP/HSDP only (no TP)."
        )

    if parallel_dims.ep_enabled:
        raise NotImplementedError(
            "full_dtensor is not yet supported with Expert Parallel. "
            "Use full_dtensor with FSDP/HSDP only (no EP)."
        )

    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "full_dtensor is not supported with Context Parallel. "
            "Disable CP or disable full_dtensor."
        )

    layer = getattr(model_config, "layer", None)
    attn_config = getattr(layer, "attention", None) if layer else None
    attn_backend = getattr(attn_config, "attn_backend", "sdpa")
    if attn_backend in ("flex", "varlen"):
        raise NotImplementedError(
            f"full_dtensor is not supported with {attn_backend} attention. "
            "Flex/varlen attention does not support DTensor dispatch. "
            "Use sdpa attention or disable full_dtensor."
        )


def _get_spmd_mesh(parallel_dims: ParallelDims) -> DeviceMesh:
    """Get the SPMD mesh for full DTensor parallelization.

    When CP=1, "fsdp" == "dp_shard", so the dense_mesh is the correct SPMD mesh.
    CP>1 is not yet supported with full DTensor. validate_config() must be called
    before this to ensure CP is not enabled.
    """
    # TP is not yet supported with full DTensor (validate_config rejects it),
    # so the mesh only includes DP dimensions.
    mesh_names = [
        n for n in ["dp_replicate", "fsdp"] if parallel_dims.get_optional_mesh(n)
    ]
    return parallel_dims.get_mesh(mesh_names)


def get_dp_mesh_dims(parallel_dims: ParallelDims) -> DataParallelMeshDims:
    """Build DataParallelMeshDims from the active parallel dimensions."""
    shard: str | None = None
    replicate: str | None = None

    if parallel_dims.dp_shard_enabled:
        shard = "fsdp"
    if parallel_dims.dp_replicate_enabled:
        replicate = "dp_replicate"

    return DataParallelMeshDims(shard=shard, replicate=replicate)


def _remove_sdpa_math_backend(model: nn.Module) -> None:
    """Remove MATH backend from SDPAAttention modules.

    SDPA MATH backend decomposes into primitive ops that mix plain tensors
    with DTensors, causing errors. Flash and efficient backends have proper
    DTensor dispatch rules and work correctly.
    """
    from torch.nn.attention import SDPBackend

    from torchtitan.models.common.attention import ScaledDotProductAttentionWrapper

    for module in model.modules():
        if isinstance(module, ScaledDotProductAttentionWrapper):
            if SDPBackend.MATH in module.sdpa_backends:
                module.sdpa_backends = [
                    b for b in module.sdpa_backends if b != SDPBackend.MATH
                ]


def _find_tied_parameters(
    model: nn.Module,
) -> list[list[tuple[nn.Module, str]]]:
    """Find groups of tied (shared) parameters in the model.

    Returns a list of groups, where each group is a list of (module, param_name)
    tuples that share the same underlying nn.Parameter object.
    """
    param_to_locations: dict[int, list[tuple[nn.Module, str]]] = {}
    for module in model.modules():
        for name, param in module._parameters.items():
            if param is None:
                continue
            pid = id(param)
            if pid not in param_to_locations:
                param_to_locations[pid] = []
            param_to_locations[pid].append((module, name))

    # Only return groups with more than one location (actually tied)
    return [locs for locs in param_to_locations.values() if len(locs) > 1]


def _restore_tied_parameters(
    tied_groups: list[list[tuple[nn.Module, str]]],
) -> None:
    """Restore tied parameter identity after distribute_module.

    distribute_module creates separate DTensor nn.Parameters for each module,
    breaking weight tying. This function re-ties them by assigning the first
    group member's parameter to all others.
    """
    for group in tied_groups:
        # Use the first location's parameter as the canonical one
        canonical_module, canonical_name = group[0]
        canonical_param = canonical_module._parameters[canonical_name]
        for module, name in group[1:]:
            module._parameters[name] = canonical_param


def resolve_fsdp_mesh(
    model: nn.Module,
    parallel_dims: ParallelDims,
    full_dtensor: bool,
) -> tuple[DeviceMesh, DataParallelMeshDims | None]:
    """Select the FSDP mesh and optional DataParallelMeshDims.

    In full DTensor mode, distributes model parameters as DTensors on the SPMD
    mesh and returns DataParallelMeshDims telling FSDP which dims are data-parallel.
    In standard mode, returns the conventional dp_mesh and None.

    Returns (dp_mesh, dp_mesh_dims) to pass to the model's apply_fsdp().
    """
    if full_dtensor:
        spmd_mesh = distribute_model(model, parallel_dims)
        dp_mesh_dims = get_dp_mesh_dims(parallel_dims)
        return spmd_mesh, dp_mesh_dims
    else:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        return parallel_dims.get_mesh(names), None


def distribute_model(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> DeviceMesh:
    """Distribute model parameters as DTensors on the full SPMD mesh.

    All parameters become DTensors with Replicate() on every mesh dimension.
    This prepares the model for fully_shard() with dp_mesh_dims.

    Returns the SPMD mesh used for distribution.
    """
    _remove_sdpa_math_backend(model)

    # Record tied parameters before distribute_module breaks them
    tied_groups = _find_tied_parameters(model)

    spmd_mesh = _get_spmd_mesh(parallel_dims)
    distribute_module(model, spmd_mesh)

    # Restore tied parameter identity broken by distribute_module
    if tied_groups:
        _restore_tied_parameters(tied_groups)
        logger.info(
            f"Restored {len(tied_groups)} tied parameter group(s) "
            f"after distribute_module"
        )

    logger.info(
        f"Distributed model parameters as DTensors on SPMD mesh "
        f"with dims {spmd_mesh.mesh_dim_names}"
    )
    return spmd_mesh


def convert_buffers_to_dtensor(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> None:
    """Convert any non-DTensor buffers to Replicate DTensors on the SPMD mesh.

    This must be called after init_weights(), which may overwrite DTensor buffers
    (e.g. freqs_cis) with plain tensors.
    """
    spmd_mesh = _get_spmd_mesh(parallel_dims)
    replicate_placements = [Replicate()] * spmd_mesh.ndim

    for name, buf in model.named_buffers():
        if buf is not None and not isinstance(buf, DTensor):
            dtensor_buf = DTensor.from_local(
                buf, spmd_mesh, replicate_placements, run_check=False
            )
            # Walk to the parent module and set the buffer
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            # Preserve the original buffer's persistence flag
            persistent = parts[-1] in parent._buffers and parts[-1] not in (
                parent._non_persistent_buffers_set
                if hasattr(parent, "_non_persistent_buffers_set")
                else set()
            )
            parent.register_buffer(parts[-1], dtensor_buf, persistent=persistent)

    logger.info("Converted remaining plain tensor buffers to DTensors")


def parallelize_inputs(
    parallel_dims: ParallelDims, inputs: torch.Tensor, labels: torch.Tensor
) -> tuple[DTensor, DTensor]:
    """Convert inputs and labels to DTensors on the SPMD mesh."""
    mesh = _get_spmd_mesh(parallel_dims)
    # Each DP dimension shards inputs along batch (dim 0).
    # TP is not yet supported (validate_config rejects it).
    placements: list[Placement] = []
    if parallel_dims.dp_replicate_enabled:
        placements.append(Shard(0))
    if parallel_dims.dp_shard_enabled:
        placements.append(Shard(0))

    assert mesh.ndim == len(placements)
    return DTensor.from_local(inputs, mesh, placements), DTensor.from_local(
        labels, mesh, placements
    )
