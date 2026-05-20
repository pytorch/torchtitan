# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torch.distributed.fsdp import DataParallelMeshDims
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.deepseek_v3 import DeepSeekV3Model
from torchtitan.models.llama4.parallelize import apply_fsdp
from torchtitan.tools.logging import logger


# Adapted from llama4/infra/parallelize.py
def parallelize_deepseekv3(
    model: DeepSeekV3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    assert training.seq_len % parallel_dims.seq_len_divisor == 0, f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if parallelism.full_dtensor:
        raise NotImplementedError("full_dtensor is not supported yet.")

    model.parallelize(parallel_dims)
    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    if model_compile_enabled:
        apply_compile(model, compile_config)

    mesh_names: list[str] = []
    if parallel_dims.dp_replicate_enabled:
        mesh_names.append("dp_replicate")
    mesh_names.append("fsdp")
    if parallel_dims.tp_enabled:
        mesh_names.append("tp")
    dp_mesh = parallel_dims.get_mesh(mesh_names)
    dense_spmd_mesh = parallel_dims.get_activated_mesh(["dp", "cp", "tp"])
    dp_mesh_dims = (
        DataParallelMeshDims(
            shard="fsdp",
            replicate="dp_replicate" if parallel_dims.dp_replicate_enabled else None,
        )
        if dense_spmd_mesh is not None
        else None
    )

    edp_mesh = None
    sparse_spmd_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        sparse_spmd_mesh = parallel_dims.get_activated_mesh(
            ["dp_replicate", "efsdp", "ep"]
        )

    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        sparse_spmd_mesh=sparse_spmd_mesh,
        dp_mesh_dims=dp_mesh_dims,
    )

    logger.info("Applied fully_shard to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    return model
