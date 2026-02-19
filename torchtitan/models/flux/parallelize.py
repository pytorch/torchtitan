# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.context_parallel import apply_cp_to_attention_module
from torchtitan.models.llama3.parallelize import disable_fsdp_gradient_division
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def parallelize_flux(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    if ac_config.mode != "none":
        apply_ac(model, ac_config)

    if parallel_dims.cp_enabled:
        apply_cp(model, parallel_dims.get_mesh("cp"))

    if parallel_dims.fsdp_enabled:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )

        dp_mesh = parallel_dims.get_mesh(names)
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            cpu_offload=training.enable_cpu_offload,
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
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    linear_layers = [
        model.img_in,
        model.time_in,
        model.vector_in,
        model.txt_in,
    ]
    for layer in linear_layers:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(layer, **fsdp_config)

    # pyrefly: ignore [not-iterable]
    for block in model.double_blocks:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(
            block,
            **fsdp_config,
        )

    # pyrefly: ignore [not-iterable]
    for block in model.single_blocks:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(
            block,
            **fsdp_config,
        )

    # apply FSDP to last layer. Set reshard_after_forward=False for last layer to avoid gather right after reshard
    # pyrefly: ignore [no-matching-overload]
    fully_shard(model.final_layer, **fsdp_config, reshard_after_forward=False)

    # Wrap all the rest of model
    fully_shard(model, **fsdp_config)

    # Disable FSDP's automatic gradient division for all FSDP modules
    disable_fsdp_gradient_division(model)


def apply_ac(model: nn.Module, ac_config):
    """Apply activation checkpointing to the model."""

    # pyrefly: ignore [missing-attribute]
    for layer_id, block in model.double_blocks.named_children():
        block = ptd_checkpoint_wrapper(block, preserve_rng_state=True)
        # pyrefly: ignore [missing-attribute]
        model.double_blocks.register_module(layer_id, block)

    # pyrefly: ignore [missing-attribute]
    for layer_id, block in model.single_blocks.named_children():
        block = ptd_checkpoint_wrapper(block, preserve_rng_state=True)
        # pyrefly: ignore [missing-attribute]
        model.single_blocks.register_module(layer_id, block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")


def apply_cp(model: nn.Module, cp_mesh: DeviceMesh) -> None:
    """
    Apply context parallelism to the Flux model.

    Args:
        model: The Flux model with double_blocks and single_blocks containing
            inner attention modules.
        cp_mesh: Device mesh for context parallel dimension

    Note:
        - Uses SDPA attention type
        - Applies to all inner_attention modules in double_blocks and single_blocks
    """
    # Collect all inner_attention modules from the Flux model
    attention_modules = []

    # pyrefly: ignore [not-iterable]
    for double_block in model.double_blocks:
        # pyrefly: ignore [missing-attribute]
        attention_modules.append(double_block.img_attn.inner_attention)
        # pyrefly: ignore [missing-attribute]
        attention_modules.append(double_block.txt_attn.inner_attention)
        # pyrefly: ignore [missing-attribute]
        attention_modules.append(double_block.inner_attention)

    # pyrefly: ignore [not-iterable]
    for single_block in model.single_blocks:
        # pyrefly: ignore [missing-attribute]
        attention_modules.append(single_block.inner_attention)

    # Apply CP using the shared implementation (always uses SDPA for Flux)
    apply_cp_to_attention_module(attention_modules, cp_mesh, "sdpa")

    logger.info("Applied Context Parallel to the Flux model")


def parallelize_encoders(
    t5_model: nn.Module,
    clip_model: nn.Module,
    parallel_dims: ParallelDims,
    *,
    training: TrainingConfig,
):
    if parallel_dims.dp_shard_enabled:  # apply FSDP or HSDP
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )

        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )
        dp_mesh = parallel_dims.get_mesh(names)
        fsdp_config: dict[str, Any] = {
            "mesh": dp_mesh,
            "mp_policy": mp_policy,
        }
        if training.enable_cpu_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy()

        # NOTE: only apply FSDP to the T5 encoder, not the CLIP text encoder.
        # CLIP Text encoder has low computation / communication ratio, so it's not necessary to apply FSDP to it.
        # pyrefly: ignore [missing-attribute]
        for block in t5_model.hf_module.encoder.block:
            fully_shard(block, **fsdp_config)
        # pyrefly: ignore [no-matching-overload]
        fully_shard(t5_model.hf_module, **fsdp_config)

        # Disable FSDP's automatic gradient division for all FSDP modules
        # pyrefly: ignore [bad-argument-type]
        disable_fsdp_gradient_division(t5_model.hf_module)

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the T5 encoder model")
        else:
            logger.info("Applied FSDP to the T5 encoder model")

    return t5_model, clip_model
