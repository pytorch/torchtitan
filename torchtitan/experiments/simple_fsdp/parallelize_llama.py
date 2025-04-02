# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torch.distributed import DeviceMesh

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3.parallelize_llama import apply_ac
from torchtitan.tools.logging import logger

from .simple_fsdp import data_parallel, MixedPrecisionPolicy


def parallelize_llama(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    # TODO(ruisizhang123): Add support for TP (on-going)
    # if parallel_dims.tp_enabled:
    #     if (
    #         job_config.parallelism.enable_async_tensor_parallel
    #         and not job_config.training.compile
    #     ):
    #         raise RuntimeError("Async TP requires --training.compile")

    #     enable_float8_linear = "float8" in job_config.model.converters
    #     float8_is_rowwise = job_config.float8.recipe_name in (
    #         "rowwise",
    #         "rowwise_with_gw_hp",
    #     )

    #     # For now, float8 all-gather with TP is only supported for tensorwise
    #     # float8 scaling recipes. For rowwise recipes, we use regular TP and
    #     # all-gather happens in high precision.
    #     enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

    #     apply_tp(
    #         model,
    #         world_mesh["tp"],
    #         loss_parallel=parallel_dims.loss_parallel_enabled,
    #         enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
    #         enable_async_tp=job_config.parallelism.enable_async_tensor_parallel,
    #     )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    # apply data parallel
    if (
        parallel_dims.dp_replicate_enabled
        or parallel_dims.dp_shard_enabled
        or parallel_dims.cp_enabled
    ):
        if parallel_dims.dp_replicate_enabled:
            if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
                dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
                dp_mode = "hybrid_shard"
            else:
                dp_mesh_dim_names = ("dp_replicate",)
                dp_mode = "replicate"
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
            dp_mode = "fully_shard"

        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        )

        model = data_parallel(
            model,
            world_mesh[tuple(dp_mesh_dim_names)],
            mode=dp_mode,
            ac_mode=job_config.activation_checkpoint.mode,
            mp_policy=mp_policy,
        )
        logger.info("Applied Data Parallel (dp mode=%s) to the model", dp_mode)

    if job_config.training.compile:
        torch._inductor.config.reorder_for_peak_memory = False
        model = torch.compile(model, fullgraph=True)

    return model
