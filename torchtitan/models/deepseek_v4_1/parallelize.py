# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 parallelization: EP/FSDP/FullAC assembly (CP1-only).

V4.1 rejects CP>1 and PP>1 in its config, so this assembly wires only
what the supported matrix needs.  The decoder FSDP wrapper is the
generic helper from torchtitan.
"""

import torch
from torchtitan.config import (
    TORCH_DTYPE_MAP,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh, resolve_sparse_fsdp_mesh, validate_config
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder


def apply_activation_checkpointing(model, ac_config, dump_folder):
    """Apply the selected policy to the model."""
    policy = ac_config.build(dump_folder=dump_folder)
    policy.apply(model)


def parallelize_deepseek_v4_1(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Parallelize V4.1: sharding config, AC, compile, then the decoder FSDP wrap."""
    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        validate_config(parallel_dims, model)
        model.parallelize(parallel_dims)
    elif parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

    if ac_config is not None:
        apply_activation_checkpointing(model, ac_config, dump_folder)

    model_compile_enabled = compile_config.enable and "model" in compile_config.components
    if model_compile_enabled:
        for layer_id, transformer_block in model.layers.named_children():
            transformer_block.compile(backend=compile_config.backend, fullgraph=False, dynamic=False)

    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    else:
        dp_mesh_names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        dp_mesh_dims = None
        edp_mesh = None
        edp_mesh_dims = None
        if parallel_dims.ep_enabled:
            edp_mesh_names = ["dp_replicate", "efsdp"] if parallel_dims.dp_replicate_enabled else ["efsdp"]
            edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    ignored_params: set[torch.nn.Parameter] = set()
    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        ignored_params=ignored_params,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )
    return model
