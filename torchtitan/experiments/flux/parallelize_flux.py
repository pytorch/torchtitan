# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


def parallelize_flux(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    if (
        parallel_dims.dp_shard_enabled or parallel_dims.dp_replicate_enabled
    ):  # apply FSDP or HSDP
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        apply_fsdp(
            model,
            world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            cpu_offload=job_config.training.enable_cpu_offload,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        cpu_offload (bool): Whether to offload model parameters to CPU. Defaults to False.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    linear_layers = [
        model.img_in,
        model.time_in,
        model.vector_in,
        model.txt_in,
    ]
    for layer in linear_layers:
        fully_shard(layer, **fsdp_config)

    for block in model.double_blocks:
        fully_shard(
            block,
            **fsdp_config,
        )

    for block in model.single_blocks:
        fully_shard(
            block,
            **fsdp_config,
        )
    # apply FSDP to last layer. Set reshard_after_forward=False for last layer to avoid gather right after reshard
    fully_shard(model.final_layer, **fsdp_config, reshard_after_forward=False)

    # Wrap all the rest of model
    fully_shard(model, **fsdp_config)


def apply_ac(model: nn.Module, ac_config):
    """Apply activation checkpointing to the model."""

    for layer_id, block in model.double_blocks.named_children():
        block = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
        model.double_blocks.register_module(layer_id, block)

    for layer_id, block in model.single_blocks.named_children():
        block = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
        model.single_blocks.register_module(layer_id, block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")


def parallelize_encoders(
    t5_model: nn.Module,
    clip_model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    if (
        parallel_dims.dp_shard_enabled or parallel_dims.dp_replicate_enabled
    ):  # apply FSDP or HSDP
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        )
        fsdp_config = {
            "mesh": world_mesh[tuple(dp_mesh_dim_names)],
            "mp_policy": mp_policy,
        }
        if job_config.training.enable_cpu_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy()
        # FSDP for encoder blocks
        for block in clip_model.hf_module.text_model.encoder.layers:
            fully_shard(block, **fsdp_config)
        fully_shard(clip_model, **fsdp_config)

        for block in t5_model.hf_module.encoder.block:
            fully_shard(block, **fsdp_config)
        fully_shard(t5_model.hf_module, **fsdp_config)

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied FSDP to the T5 and CLIP model")
        else:
            logger.info("Applied FSDP to the T5 and CLIP model")

    return t5_model, clip_model
