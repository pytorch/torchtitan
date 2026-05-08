# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.distributed._functional_collectives as funcol
import torch.nn as nn

from torchtitan.experiments.flex_shard import (
    disable_active_parametrization,
    FlatShard,
    get_placements,
    is_flex_shard_param,
    Shard,
)
from torchtitan.models.llama3 import Llama3Model


_SHARD_INFO_ATTR = "_flex_shard_dcp_info"


def _get_local_data(param: nn.Parameter, info: dict) -> torch.Tensor:
    """Extract the DP-local plain tensor from a param that may be a DTensor."""
    data = param.data
    if info.get("is_dtensor"):
        from torch.distributed.tensor import DTensor

        if isinstance(data, DTensor):
            data = data.to_local()
    return data


def _reconstruct_full_tensor(
    local_data: torch.Tensor,
    info: dict,
) -> torch.Tensor:
    """All-gather on DP mesh, then full_tensor on TP/EP mesh if DTensor."""
    placement = info["placements"][0]
    mesh = info["mesh"]

    if isinstance(placement, Shard):
        dp_full = funcol.all_gather_tensor(local_data, placement.dim, mesh.get_group())
    elif isinstance(placement, FlatShard):
        dp_full = funcol.all_gather_tensor(local_data, 0, mesh.get_group())
        dp_full = dp_full.view(info["global_shape"])
    else:
        dp_full = local_data

    # For params that started as DTensors, also reconstruct across the
    # original TP/EP or full SPMD mesh.
    if info.get("is_dtensor"):
        from torch.distributed.tensor import DTensor

        dt = DTensor.from_local(
            dp_full,
            info["dtensor_mesh"],
            info["dtensor_placements"],
            run_check=False,
        )
        return dt.full_tensor()
    return dp_full


def _state_dict_post_hook(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict,
) -> dict[str, torch.Tensor]:
    """Replace sharded FlexShard params with all-gathered full tensors in state dict.

    This hook fires per-module during state_dict() traversal. For each FlexShard
    parameter, it all-gathers the shard to reconstruct the full tensor, making
    the state dict compatible with DCP save/load.

    Handles both plain tensor params (1D mesh) and DTensor params (multi-mesh
    FSDP + TP/EP composition).

    Uses sharding info stored on the module (not the tensor) by
    _register_dcp_hooks(), because to_empty() creates new tensors that lose
    per-tensor metadata.
    """
    shard_info = getattr(module, _SHARD_INFO_ATTR, {})
    for name, info in shard_info.items():
        key = prefix + name
        param = module._parameters.get(name)
        if param is not None and key in state_dict:
            local_data = _get_local_data(param, info)
            full = _reconstruct_full_tensor(local_data, info)
            state_dict[key] = full.detach().clone()
    return state_dict


def _shard_for_load(
    full_tensor: torch.Tensor,
    info: dict,
) -> torch.Tensor:
    """Shard a full tensor for loading into a FlexShard param.

    For DTensor params, first shards along TP/EP or full SPMD dims, then
    along the FlexShard DP dim.
    """
    data = full_tensor
    placement = info["placements"][0]
    mesh = info["mesh"]

    # For DTensor params, shard along TP/EP or full SPMD dims first. In full
    # SPMD mode, DP dims are Replicate() in dtensor_placements, so this only
    # applies the non-DP sharding before FlexShard chunks along DP below.
    if info.get("is_dtensor"):
        from torch.distributed.tensor import DTensor

        dt = DTensor.from_local(
            data,
            info["dtensor_mesh"],
            [torch.distributed.tensor.Replicate()] * info["dtensor_mesh"].ndim,
            run_check=False,
        )
        dt = dt.redistribute(info["dtensor_mesh"], info["dtensor_placements"])
        data = dt.to_local()

    # Shard along DP dim
    if isinstance(placement, Shard):
        rank = mesh.get_local_rank()
        chunks = data.chunk(mesh.size(), dim=placement.dim)
        return chunks[rank].contiguous()
    elif isinstance(placement, FlatShard):
        rank = mesh.get_local_rank()
        flat = data.reshape(-1)
        chunks = flat.chunk(mesh.size())
        return chunks[rank].contiguous()
    return data


def _load_state_dict_pre_hook(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    """Shard full tensors in state dict before load_state_dict assigns them.

    When loading from a checkpoint with full (unsharded) tensors into a
    FlexShard model with sharded parameters, this hook chunks the full tensor
    and replaces it with this rank's shard so the assignment succeeds.

    Handles both plain tensor params (1D mesh) and DTensor params (multi-mesh).
    """
    shard_info = getattr(module, _SHARD_INFO_ATTR, {})
    for name, info in shard_info.items():
        key = prefix + name
        param = module._parameters.get(name)
        if param is None or key not in state_dict:
            continue
        if state_dict[key].shape != param.shape:
            state_dict[key] = _shard_for_load(state_dict[key], info)


def _register_dcp_hooks(model: nn.Module) -> None:
    """Register state_dict hooks on all submodules that have FlexShard params.

    Captures sharding info (placements, mesh) from the tensors and stores it
    on the module, because to_empty() later creates new tensors that lose
    per-tensor metadata like _placements and _mesh.

    For DTensor params (from TP/EP), also stores the DTensor mesh and
    placements so the hooks can reconstruct the full tensor across all meshes.
    """
    for module in model.modules():
        shard_info: dict[str, dict] = {}
        for name, param in module._parameters.items():
            if param is not None and is_flex_shard_param(param):
                entry: dict = {
                    "placements": get_placements(param),
                    "mesh": param._mesh,
                    "global_shape": param._global_shape
                    if hasattr(param, "_global_shape")
                    else param.shape,
                    "is_dtensor": False,
                }
                if hasattr(param, "_spmd_mesh") and hasattr(param, "_spmd_placements"):
                    entry["is_dtensor"] = True
                    entry["dtensor_mesh"] = param._spmd_mesh
                    entry["dtensor_placements"] = tuple(param._spmd_placements)
                try:
                    from torch.distributed.tensor import DTensor

                    if isinstance(param, DTensor):
                        entry["is_dtensor"] = True
                        entry["dtensor_mesh"] = param._spec.mesh
                        entry["dtensor_placements"] = tuple(param._spec.placements)
                except ImportError:
                    pass
                shard_info[name] = entry
        if shard_info:
            setattr(module, _SHARD_INFO_ATTR, shard_info)
            module._register_state_dict_hook(_state_dict_post_hook)
            module._register_load_state_dict_pre_hook(
                _load_state_dict_pre_hook, with_module=True
            )


class FlexShardLlama3Model(Llama3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        with disable_active_parametrization():
            super().init_states(buffer_device=buffer_device)
