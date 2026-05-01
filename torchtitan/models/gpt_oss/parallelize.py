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
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.full_dtensor import (
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
    validate_config,
)
from torchtitan.models.gpt_oss.model import GptOssModel
from torchtitan.models.llama4.parallelize import apply_fsdp
from torchtitan.tools.logging import logger

# Adapted from llama4/infra/parallelize.py
def parallelize_gptoss(
    model: GptOssModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if parallelism.full_dtensor:
        validate_config(parallel_dims, model)
        model.parallelize(parallel_dims)
    else:
        # CP: wrap inner attention forward BEFORE parallelize() so CP logic
        # runs inside the local_map boundary on local tensors.
        if parallel_dims.cp_enabled:
            apply_cp_to_forward(
                # pyrefly: ignore [missing-attribute]
                [block.attention.inner_attention for block in model.layers.values()],
                parallel_dims.get_mesh("cp"),
            )
        # ``model.parallelize`` walks every ``Module`` and applies its
        # ``sharding_config`` (dense + MoE).
        if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
            model.parallelize(parallel_dims)

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
        )

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, compile_config)

    if parallelism.full_dtensor:
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(
            model, parallel_dims, parallelism.full_dtensor
        )
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(
            parallel_dims, parallelism.full_dtensor
        )
    else:
        dp_mesh = parallel_dims.get_enabled_mesh(["dp_replicate", "fsdp"])
        assert dp_mesh is not None
        dp_mesh_dims = None
        edp_mesh = None
        edp_mesh_dims = None
        if parallel_dims.ep_enabled:
            edp_mesh = parallel_dims.get_enabled_mesh(["dp_replicate", "efsdp"])

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
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
    )

    logger.info("Applied fully_shard to the model")

    if parallel_dims.cp_enabled:
        logger.info("Applied Context Parallel to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    return model
