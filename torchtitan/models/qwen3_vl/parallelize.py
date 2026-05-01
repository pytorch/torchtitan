# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelization utilities for Qwen3-VL.

This module applies PT-D parallelisms and various training techniques
(activation checkpointing, compile, FSDP) to the Qwen3-VL model.
"""

import torch
import torch._inductor.config
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

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
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.models.llama4.parallelize import apply_fsdp
from torchtitan.tools.logging import logger


def _apply_fsdp_to_vision_encoder(
    vision_encoder: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    reshard_after_forward_policy: str = "default",
    pp_enabled: bool = False,
):
    """
    Apply FSDP to the vision encoder as a single unit.

    Wraps the entire vision encoder with one fully_shard call so all parameters
    are gathered in a single AllGather. The vision encoder's compute is small
    relative to the decoder, so per-layer sharding would launch many small
    AllGather kernels whose total overhead exceeds a single AllGather followed
    by computing all layers in one shot — even without overlap.

    Must be called before apply_fsdp on the decoder so the vision encoder is
    already sharded when the final fully_shard(model) is applied.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled=pp_enabled
    )

    fully_shard(
        vision_encoder,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )


def parallelize_qwen3_vl(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the Qwen3-VL model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    from torchtitan.distributed.full_dtensor import (
        resolve_fsdp_mesh,
        resolve_sparse_fsdp_mesh,
        validate_config,
    )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if parallelism.full_dtensor:
        validate_config(parallel_dims, model)
        # pyrefly: ignore [not-callable]
        model.parallelize(parallel_dims)
    else:
        if parallel_dims.cp_enabled:
            raise NotImplementedError(
                "Context Parallel is not yet supported for Qwen3-VL "
                "without full_dtensor."
            )
        # ``model.parallelize`` walks every ``Module`` and applies its
        # ``sharding_config`` (decoder dense + MoE + vision encoder).
        if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
            # pyrefly: ignore [not-callable]
            model.parallelize(parallel_dims)

    if parallel_dims.tp_enabled:
        if parallelism.enable_async_tensor_parallel and not model_compile_enabled:
            raise RuntimeError("Async TP requires torch.compile")
        if parallelism.enable_async_tensor_parallel:
            torch._inductor.config._micro_pipeline_tp = True

    # Apply activation checkpointing
    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )
        if model.vision_encoder is not None:
            apply_ac(
                # pyrefly: ignore [bad-argument-type]
                model.vision_encoder,
                ac_config,
                model_compile_enabled=model_compile_enabled,
                base_folder=dump_folder,
            )

    # Apply torch.compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, compile_config)
        if model.vision_encoder is not None:
            # pyrefly: ignore [bad-argument-type]
            apply_compile(model.vision_encoder, compile_config)

    # Resolve FSDP meshes (full_dtensor-aware).
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

    # FSDP the vision encoder as a single unit
    if model.vision_encoder is not None:
        _apply_fsdp_to_vision_encoder(
            # pyrefly: ignore [bad-argument-type]
            model.vision_encoder,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=parallel_dims.pp_enabled,
        )

    # FSDP the decoder with MoE-aware sharding
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

    logger.info("Applied fully_shard to the Qwen3-VL model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the Qwen3-VL model")

    return model
