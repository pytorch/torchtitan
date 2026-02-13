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
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac

from torchtitan.models.llama3.infra.parallelize import (
    _op_sac_save_list,
    apply_ddp,
)
from torchtitan.models.llama4.infra.parallelize import (
    apply_compile,
    apply_fsdp,
    apply_moe_ep_tp,
)
from torchtitan.tools.logging import logger


def parallelize_qwen3_vl(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the Qwen3-VL model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    assert isinstance(model.visual, nn.Module), "Model must have a vision encoder"

    # Validate sequence length divisibility
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # Check attention type compatibility
    attn_type = getattr(model.model_args, "attn_type", "sdpa")
    if job_config.parallelism.context_parallel_degree > 1 and attn_type != "sdpa":
        raise NotImplementedError("CP support is only supported for SDPA.")

    # TP is not yet supported for VLM training
    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "Tensor Parallelism for Qwen3-VL training is still in progress."
        )

    # Apply MoE expert parallelism to decoder layers
    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
        )

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    # Apply activation checkpointing
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_op_sac_save_list,
        )
        apply_ac(model.visual, job_config.activation_checkpoint)

    # Apply torch.compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, job_config.compile, parallel_dims.ep_enabled)
    if job_config.compile.enable:
        _apply_compile_to_visual(model.visual, job_config.compile)

    # Apply FSDP or HSDP
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

        # FSDP the vision encoder components individually for memory efficiency
        _apply_fsdp_to_visual(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
        )

        # FSDP the decoder with MoE-aware sharding (reuses llama4 apply_fsdp)
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the Qwen3-VL model")
        else:
            logger.info("Applied FSDP to the Qwen3-VL model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the Qwen3-VL model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the Qwen3-VL model")

    elif parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh("dp_replicate")
        if dp_mesh is not None and dp_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=job_config.compile.enable,
        )

    return model


def _apply_fsdp_to_visual(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply FSDP to the vision encoder components individually.

    This must be called before the llama4 apply_fsdp so that vision encoder
    components are individually sharded before the final fully_shard(model).
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    if not hasattr(model, "visual") or model.visual is None:
        return

    # Shard patch embedding
    if hasattr(model.visual, "patch_embed"):
        fully_shard(
            model.visual.patch_embed,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Shard each vision transformer layer
    for layer_id, transformer_block in model.visual.layers.items():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Shard merger if present
    if hasattr(model.visual, "merger"):
        fully_shard(
            model.visual.merger,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Shard projector if present
    if hasattr(model, "projector") and model.projector is not None:
        fully_shard(
            model.projector,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )


def _apply_compile_to_visual(visual: nn.Module, compile_config):
    """Apply torch.compile to vision encoder transformer blocks."""
    for layer_id, transformer_block in visual.layers.named_children():
        transformer_block = torch.compile(
            transformer_block,
            backend=compile_config.backend,
            fullgraph=True,
        )
        visual.layers.register_module(layer_id, transformer_block)
    logger.info("Compiling each visual TransformerBlock with torch.compile")
