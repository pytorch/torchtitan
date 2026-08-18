# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TypeAlias

import spmd_types as spmd
import torch
import torch.nn as nn

from spmd_types.checker import typecheck
from spmd_types.runtime import has_local_type
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.experiments.graph_trainer.simple_fsdp import (
    _register_parametrization,
    is_active_parametrization,
    MixedPrecisionPolicy,
)
from torchtitan.protocols.module import Module


StorageTimePlacement: TypeAlias = tuple[
    DeviceMesh,
    str,
    spmd.PerMeshAxisSpmdType,
]


def _shard_param_for_fsdp_storage(
    param: nn.Parameter,
    fsdp_mesh: DeviceMesh,
    shard_axis_name: str,
    shard_dim: int,
) -> nn.Parameter:
    if not -param.ndim <= shard_dim < param.ndim:
        raise ValueError(
            f"Cannot shard parameter with {param.ndim} dimensions on tensor "
            f"dim {shard_dim}."
        )
    canonical_shard_dim = shard_dim % param.ndim
    dim_size = param.size(canonical_shard_dim)
    shard_degree = fsdp_mesh[shard_axis_name].size()
    if dim_size % shard_degree != 0:
        raise ValueError(
            f"Cannot evenly shard parameter shape {tuple(param.shape)} on tensor "
            f"dim {canonical_shard_dim} across {shard_degree} ranks."
        )

    # track GSPMD annotation on param, so weight init can read & feed correct shard
    fsdp_group = fsdp_mesh.get_group(shard_axis_name)
    fsdp_axis = spmd.MeshAxis.of(fsdp_group)
    with typecheck(local=False):
        param_for_storage = param.detach()
        spmd.mutate_type(
            param_for_storage,
            fsdp_axis,
            src=spmd.R,
            dst=spmd.I,
        )
        local_tensor = spmd.redistribute(
            param_for_storage,
            fsdp_group,
            src=spmd.I,
            dst=spmd.S(canonical_shard_dim),
        )

    local_param = nn.Parameter(local_tensor, requires_grad=param.requires_grad)
    spmd.assert_type_like(local_param, local_tensor)
    return local_param


def _get_non_dp_storage_time_placements(
    param: nn.Parameter,
    non_dp_mesh: DeviceMesh | None,
) -> tuple[StorageTimePlacement, ...]:
    """
    Return list of non-DP mesh axes requiring I->R convert in FWD,
    translating to P->I all-reduce in BWD.
    """
    if non_dp_mesh is None:
        return ()
    if not has_local_type(param):
        raise ValueError(
            "Parameters must have SPMD layouts before applying SimpleFSDP "
            "with a non_dp_mesh."
        )

    assert non_dp_mesh.mesh_dim_names is not None
    return tuple(
        (non_dp_mesh, axis_name, spmd.I)
        for axis, axis_name in enumerate(non_dp_mesh.mesh_dim_names)
        if non_dp_mesh.size(axis) > 1
        and spmd.get_axis_local_type(param, non_dp_mesh.get_group(axis_name)) is spmd.R
    )


class ReplicateComputation(Module):
    """Materialize parameter storage for replicated compute."""

    def __init__(
        self,
        storage_time_placements_by_param: dict[str, tuple[StorageTimePlacement, ...]],
        mp_policy: MixedPrecisionPolicy | None,
    ) -> None:
        super().__init__()
        self.storage_time_placements_by_param = storage_time_placements_by_param
        mp_policy = mp_policy or MixedPrecisionPolicy()
        self.param_dtype = mp_policy.param_dtype
        self.reduce_dtype = mp_policy.reduce_dtype

    def forward(self, param: torch.Tensor, param_name: str) -> torch.Tensor:
        if not is_active_parametrization():
            return param

        result = param
        for mesh, axis_name, storage_type in self.storage_time_placements_by_param[
            param_name
        ]:
            result = spmd.redistribute(
                result,
                mesh.get_group(axis_name),
                src=storage_type,
                dst=spmd.R,
                op_dtype=self.param_dtype,
                backward_options={"op_dtype": self.reduce_dtype},
            )
        return result


def data_parallel(
    model: nn.Module,
    fsdp_mesh: DeviceMesh,
    mode: str = "replicate",
    mp_policy: MixedPrecisionPolicy | None = None,
    shard_dim: int = 0,
    non_dp_mesh: DeviceMesh | None = None,
) -> nn.Module:
    """Apply local-tensor SPMD data parallelism to ``model``."""
    if fsdp_mesh.mesh_dim_names is None:
        raise ValueError("fsdp_mesh must have named axes.")
    fsdp_axis_names = fsdp_mesh.mesh_dim_names
    if non_dp_mesh is not None and non_dp_mesh.mesh_dim_names is None:
        raise ValueError("non_dp_mesh must have named axes.")

    # configure FSDP storage mesh axes, placements
    if mode == "replicate":
        storage_shard_axis_name = None
        fsdp_storage_types = (spmd.I,) * fsdp_mesh.ndim
    elif mode == "fully_shard":
        if fsdp_mesh.ndim != 1:
            raise ValueError("fully_shard requires a one-dimensional fsdp_mesh.")
        shard_axis_name = fsdp_axis_names[0]
        storage_shard_axis_name = shard_axis_name
        fsdp_storage_types = (spmd.S(shard_dim),)
    elif mode == "hybrid_shard":
        if fsdp_mesh.ndim != 2:
            raise ValueError("hybrid_shard requires a two-dimensional fsdp_mesh.")
        shard_axis_name = fsdp_axis_names[1]
        storage_shard_axis_name = shard_axis_name
        fsdp_storage_types = (spmd.I, spmd.S(shard_dim))
    else:
        raise ValueError(f"Unsupported mode {mode!r}.")

    fsdp_storage_time_placements = tuple(
        (fsdp_mesh, axis_name, storage_type)
        for axis_name, storage_type in zip(
            fsdp_axis_names,
            fsdp_storage_types,
            strict=True,
        )
    )

    modules = list(model.modules())
    for module in modules:
        params = dict(module.named_parameters(recurse=False))
        if "SimpleFSDP" in module.__class__.__name__:
            continue

        # for compute-time / pre-forward: besides FSDP axis unshard, collect params requiring
        # model-parallel axis I->R convert calls, so FSDP can handle BWD all-reduces.
        storage_time_placements_by_param = {}
        for param_name, param in params.items():
            if param is None:
                continue
            non_dp_storage_time_placements = _get_non_dp_storage_time_placements(
                param, non_dp_mesh
            )
            storage_time_placements_by_param[param_name] = (
                fsdp_storage_time_placements + non_dp_storage_time_placements
            )
            if storage_shard_axis_name is not None and param.numel() > 0:
                module.register_parameter(
                    param_name,
                    _shard_param_for_fsdp_storage(
                        param,
                        fsdp_mesh,
                        storage_shard_axis_name,
                        shard_dim,
                    ),
                )

        _register_parametrization(
            module,
            list(params),
            ReplicateComputation(
                storage_time_placements_by_param,
                mp_policy,
            ),
        )
    return model
