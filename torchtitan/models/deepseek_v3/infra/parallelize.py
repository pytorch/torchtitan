# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3.infra.parallelize import apply_ac, apply_fsdp
from torchtitan.models.deepseek_v3.model.moe import MoE
from torchtitan.tools.logging import logger


def apply_compile(model: nn.Module):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # Fail loudly if we exceed the recompile limit (default 8)
    torch._dynamo.config.fail_on_recompile_limit_hit = True

    for layer_id, transformer_block in model.layers.named_children():
        fullgraph=True
        if isinstance(transformer_block.moe, MoE):
            # Allow graph break for MoE layers
            fullgraph = False
        transformer_block = torch.compile(transformer_block, fullgraph=fullgraph)
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")


def parallelize_deepseekv3(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if job_config.training.compile:
        apply_compile(model)

    dp_mesh: DeviceMesh | None = None
    if (
        parallel_dims.dp_shard_enabled
    ):  # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard")
        else:
            dp_mesh_dim_names = ("dp_shard",)
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

    return model
