# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelization strategy for LFM2.

This applies basic parallelism without tensor parallelism or pipeline parallelism:
- Activation Checkpointing (AC) - selective and full modes supported
- torch.compile
- Data Parallelism (FSDP/DDP)
"""

import torch
import torch.nn as nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.config.job_config import ActivationCheckpoint as ACConfig, Compile as CompileConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


def parallelize_lfm2(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply activation checkpointing, torch.compile, and data parallelism to LFM2.

    NOTE: Tensor parallelism and pipeline parallelism are not yet supported for LFM2.
    Activation checkpointing only supports selective mode.

    Args:
        model: LFM2 model instance
        parallel_dims: Parallel dimensions configuration
        job_config: Job configuration

    Returns:
        Parallelized model
    """

    # Raise error if TP or PP is enabled
    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "Tensor parallelism is not yet supported for LFM2. "
            "Please set tensor_parallel_degree = 1 in your config."
        )

    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "Pipeline parallelism is not yet supported for LFM2. "
            "Please set pipeline_parallel_degree = 1 in your config."
        )

    # Validate activation checkpointing mode
    if job_config.activation_checkpoint.mode not in ("none", "selective", "full"):
        raise ValueError(
            f"Invalid activation checkpointing mode: {job_config.activation_checkpoint.mode}. "
            "Valid modes: 'none', 'selective', 'full'"
        )

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    # Apply activation checkpointing
    if job_config.activation_checkpoint.mode in ("selective", "full"):
        apply_lfm2_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            base_folder=job_config.job.dump_folder,
        )
        if job_config.activation_checkpoint.mode == "selective":
            logger.info("Applied Selective Activation Checkpointing to LFM2 attention blocks")
        else:
            logger.info("Applied Full Activation Checkpointing to all LFM2 layers")

    # Apply torch.compile
    if model_compile_enabled:
        apply_compile(model, job_config.compile)

    # Apply data parallelism (FSDP or DDP)
    if parallel_dims.fsdp_enabled:
        # dp_mesh is the mesh for FSDP/HSDP
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)
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
            logger.info("Applied HSDP to LFM2 model")
        else:
            logger.info("Applied FSDP to LFM2 model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to LFM2 model")
    elif parallel_dims.dp_replicate_enabled:
        dp_replicate_mesh = parallel_dims.get_mesh("dp_replicate")
        if parallel_dims.world_size != dp_replicate_mesh.size():
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            dp_replicate_mesh,
            enable_compile=model_compile_enabled,
        )

    return model


class ConvBlockWrapper(nn.Module):
    """Wrapper for LFM2ConvBlock that accepts and ignores attention-specific arguments.

    This is needed because when activation checkpointing is enabled, the checkpoint wrapper
    passes all arguments that attention blocks receive to conv blocks as well. Conv blocks
    only need the hidden_states tensor and should ignore all attention-specific arguments.
    """
    def __init__(self, conv_block):
        super().__init__()
        self.conv_block = conv_block

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=None,
        use_cache=None,
        **kwargs
    ):
        # Conv blocks only need hidden_states, ignore all attention-specific arguments
        return (self.conv_block(hidden_states),)


def apply_lfm2_ac(
    model: nn.Module,
    ac_config: ACConfig,
    *,
    model_compile_enabled: bool = False,
    base_folder: str = "",
) -> None:
    """Apply activation checkpointing to LFM2 layers.

    Supports two modes:
    - "selective": Checkpoint only attention blocks (which have O(seq_len²) memory).
                   Conv blocks are skipped since they have much lower O(seq_len) memory.
    - "full": Checkpoint ALL layers (both attention and conv blocks) for maximum
              memory savings at the cost of more recomputation.

    Args:
        model: LFM2 model instance
        ac_config: Activation checkpointing configuration (mode: "selective" or "full")
        model_compile_enabled: Whether torch.compile is enabled
        base_folder: Base folder for saving checkpointing artifacts
    """
    from lfm2.main import LFM2Block, LFM2ConvBlock
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper,
    )

    # Count blocks for logging
    attn_block_count = 0
    conv_block_count = 0
    checkpointed_count = 0

    for layer_id, layer in model.layers.named_children():
        should_checkpoint = False
        is_conv_block = False

        if isinstance(layer, LFM2Block):
            # Attention block
            attn_block_count += 1
            should_checkpoint = True  # Always checkpoint attention in both modes
        elif isinstance(layer, LFM2ConvBlock):
            # Conv block
            conv_block_count += 1
            is_conv_block = True
            should_checkpoint = (ac_config.mode == "full")  # Only checkpoint conv in full mode

        if should_checkpoint:
            # For conv blocks, wrap them first so they can accept attention_mask
            layer_to_checkpoint = ConvBlockWrapper(layer) if is_conv_block else layer

            checkpointed_layer = ptd_checkpoint_wrapper(
                layer_to_checkpoint,
                preserve_rng_state=ac_config.preserve_rng_state,
                determinism_check=ac_config.determinism_check,
                early_stop=ac_config.early_stop,
                debug=ac_config.debug,
            )
            model.layers.register_module(layer_id, checkpointed_layer)
            checkpointed_count += 1

    if ac_config.mode == "selective":
        logger.info(
            f"Applied selective AC to {checkpointed_count} attention blocks "
            f"({conv_block_count} conv blocks left untouched for efficiency)"
        )
    else:  # full mode
        logger.info(
            f"Applied full AC to all {checkpointed_count} layers "
            f"({attn_block_count} attention + {conv_block_count} conv blocks)"
        )


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    """Apply torch.compile to the LFM2 model."""
    # Extract only torch.compile parameters (backend)
    compile_kwargs = {"backend": compile_config.backend}
    for layer in model.model.model.layers:
        layer.forward = torch.compile(layer.forward, **compile_kwargs)
    logger.info("Applied torch.compile to LFM2 model")


def apply_fsdp(
    model: nn.Module,
    dp_mesh,
    param_dtype,
    reduce_dtype,
    pp_enabled: bool = False,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str | None = None,
):
    """Apply FSDP to the LFM2 model.

    Args:
        model: The model to apply FSDP to
        dp_mesh: Device mesh for data parallelism
        param_dtype: Data type for parameters
        reduce_dtype: Data type for gradient reduction
        pp_enabled: Whether pipeline parallelism is enabled
        cpu_offload: Whether to enable CPU offloading
        reshard_after_forward_policy: Reshard after forward policy
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    if reshard_after_forward_policy:
        if reshard_after_forward_policy == "true":
            fsdp_config["reshard_after_forward"] = True
        elif reshard_after_forward_policy == "false":
            fsdp_config["reshard_after_forward"] = False
        elif reshard_after_forward_policy == "num_layers":
            fsdp_config["reshard_after_forward"] = int(
                model.model_args.num_conv_blocks + model.model_args.num_attention_blocks
            )

    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # Apply FSDP to each layer in the model
    for layer in model.model.model.layers:
        fully_shard(layer, **fsdp_config)

    # Apply FSDP to the root model
    fully_shard(model, **fsdp_config)


def apply_ddp(
    model: nn.Module,
    dp_mesh,
    enable_compile: bool,
):
    """Apply DDP to the LFM2 model.

    Note: torch.compile should be applied before calling this function.
    The layers will already be compiled when DDP wraps the model.

    Args:
        model: The model to apply DDP to
        dp_mesh: Device mesh for data parallelism
        enable_compile: Whether compilation is enabled (for logging)
    """
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)
    logger.info(f"Applied DDP to LFM2 model{' (with compiled layers)' if enable_compile else ''}")
