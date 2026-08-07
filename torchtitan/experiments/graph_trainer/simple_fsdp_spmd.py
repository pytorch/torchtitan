# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd
import torch
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer import simple_fsdp
from torchtitan.experiments.graph_trainer.simple_fsdp import MixedPrecisionPolicy
from torchtitan.protocols.module import Module


def build_legacy_dense_mesh(parallel_dims: ParallelDims) -> DeviceMesh:
    """Build the ("pp", "dp_replicate", "fsdp", "tp") mesh."""
    if parallel_dims._world_mesh is None:
        parallel_dims.build_mesh()
    assert parallel_dims._world_mesh is not None

    fsdp = parallel_dims.dp_shard * parallel_dims.cp
    backend_override = {
        name: "fake"
        for name, degree in (
            ("pp", parallel_dims.pp),
            ("dp_replicate", parallel_dims.dp_replicate),
            ("fsdp", fsdp),
            ("tp", parallel_dims.tp),
        )
        if not parallel_dims._mesh_exist(name, degree)
    }
    full_legacy_dense_mesh = parallel_dims._world_mesh._unflatten(
        0,
        (parallel_dims.pp, parallel_dims.dp_replicate, fsdp, parallel_dims.tp),
        ("pp", "dp_replicate", "fsdp", "tp"),
        backend_override=backend_override,
    )
    return full_legacy_dense_mesh["dp_replicate", "fsdp", "tp"]


class ReplicateComputation(Module):
    def __init__(
        self,
        storage_mesh: DeviceMesh,
        *,
        compute_mesh: DeviceMesh,
        shard_dim: int,
        mp_policy: MixedPrecisionPolicy | None,
    ) -> None:
        super().__init__()
        self.storage_mesh = storage_mesh
        self.compute_mesh = compute_mesh
        self.shard_dim = shard_dim
        mp_policy = mp_policy or MixedPrecisionPolicy()
        self.param_dtype: torch.dtype | None = mp_policy.param_dtype
        self.reduce_dtype: torch.dtype | None = mp_policy.reduce_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not simple_fsdp._active_parametrization:
            return x

        output = spmd.redistribute(
            x,
            self.storage_mesh.get_group("fsdp"),
            src=spmd.S(self.shard_dim),
            dst=spmd.R,
            op_dtype=self.param_dtype,
            backward_options={"op_dtype": self.reduce_dtype or x.dtype},
        )
        if spmd.is_type_checking():
            output = spmd.reinterpret_mesh(output, self.compute_mesh)
        return output


def data_parallel(
    model: torch.nn.Module,
    *,
    storage_mesh: DeviceMesh,
    compute_mesh: DeviceMesh,
    shard_dim: int,
    mp_policy: MixedPrecisionPolicy | None,
) -> torch.nn.Module:
    modules = list(model.modules())
    fsdp_axis = storage_mesh.get_group("fsdp")
    assert storage_mesh.mesh_dim_names is not None
    fsdp_degree = storage_mesh.size(storage_mesh.mesh_dim_names.index("fsdp"))

    for module in modules:
        params = dict(module.named_parameters(recurse=False))
        if "SimpleFSDP" in module.__class__.__name__:
            continue
        simple_fsdp._register_parametrization(
            module,
            list(params),
            ReplicateComputation(
                storage_mesh,
                compute_mesh=compute_mesh,
                shard_dim=shard_dim,
                mp_policy=mp_policy,
            ),
        )

    original_init_states = model.init_states  # pyrefly: ignore [missing-attribute]

    def init_states(*, buffer_device: torch.device | None = None) -> None:
        original_init_states(buffer_device=buffer_device)
        replacements: dict[int, torch.nn.Parameter] = {}
        with torch.no_grad():
            for module in modules:
                for name, param in list(module._parameters.items()):
                    if param is None or param.numel() == 0:
                        continue
                    replacement = replacements.get(id(param))
                    if replacement is not None:
                        module.register_parameter(name, replacement)
                        continue
                    if param.size(shard_dim) % fsdp_degree != 0:
                        raise ValueError(
                            f"Parameter {name} size {param.size(shard_dim)} on dim "
                            f"{shard_dim} is not divisible by FSDP degree "
                            f"{fsdp_degree}."
                        )

                    tensor = param
                    if spmd.has_local_type(tensor):
                        tensor = spmd.reinterpret_mesh(tensor, storage_mesh)
                    tensor = spmd.shard(
                        tensor,
                        fsdp_axis,
                        src=spmd.R,
                        dst=spmd.S(shard_dim),
                    )
                    replacement = torch.nn.Parameter(
                        tensor,
                        requires_grad=param.requires_grad,
                    )
                    module.register_parameter(name, replacement)
                    replacements[id(param)] = replacement
        delattr(model, "init_states")

    model.init_states = init_states  # pyrefly: ignore [missing-attribute]
    return model
