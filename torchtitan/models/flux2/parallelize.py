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


def parallelize_flux2(
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
    del model_converters, parallelism, compile_config, dump_folder

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
            logger.info("Applied HSDP to the FLUX.2 model")
        else:
            logger.info("Applied FSDP to the FLUX.2 model")

    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    linear_layers = [
        model.img_in,
        model.time_in,
        model.txt_in,
    ]
    if hasattr(model, "guidance_in"):
        linear_layers.append(model.guidance_in)

    for layer in linear_layers:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(layer, **fsdp_config)

    for module in [
        model.double_stream_modulation_img,
        model.double_stream_modulation_txt,
        model.single_stream_modulation,
        model.final_layer,
    ]:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(module, **fsdp_config)

    for block in model.double_blocks:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(block, **fsdp_config)

    for block in model.single_blocks:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(block, **fsdp_config)

    # Wrap all the rest of model
    fully_shard(model, **fsdp_config)

    # Disable FSDP's automatic gradient division for all FSDP modules
    disable_fsdp_gradient_division(model)


def apply_ac(model: nn.Module, ac_config: ActivationCheckpointConfig) -> None:
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

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the FLUX.2 model")


def apply_cp(model: nn.Module, cp_mesh: DeviceMesh) -> None:
    attention_modules = []

    for block in model.double_blocks:
        attention_modules.append(block.inner_attention)

    for block in model.single_blocks:
        attention_modules.append(block.inner_attention)

    apply_cp_to_attention_module(attention_modules, cp_mesh, "sdpa")

    logger.info("Applied Context Parallel to the FLUX.2 model")
