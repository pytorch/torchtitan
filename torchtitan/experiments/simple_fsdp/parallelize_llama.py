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
from torchtitan.models.llama.parallelize_llama import apply_ac
from torchtitan.tools.logging import logger

from .simple_fsdp import apply_data_parallel, MixedPrecisionPolicy


def parallelize_llama(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply activation checkpointing, torch.compile, and simplefsdp to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    if parallel_dims.dp_shard_enabled or parallel_dims.dp_replicate_enabled:
        # TODO(ruisizhang123): Add support for hybrid sharding
        assert not (
            parallel_dims.dp_shard_enabled and parallel_dims.dp_replicate_enabled
        ), "Hybrid sharding is not supported"
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate",)
            fsdp_mode = "replicate"
        else:
            dp_mesh_dim_names = ("dp_shard",)
            fsdp_mode = "fully_shard"

        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        )

        model = apply_data_parallel(
            model,
            world_mesh[tuple(dp_mesh_dim_names)],
            mode=fsdp_mode,
            ac_mode=job_config.activation_checkpoint.mode,
            mp_policy=mp_policy,
        )
        logger.info("Applied SimpleFSDP (fsdp mode=%s) to the model", fsdp_mode)

    if job_config.training.compile:
        torch._inductor.config.reorder_for_peak_memory = False
        model = torch.compile(model, fullgraph=True)

    return model
