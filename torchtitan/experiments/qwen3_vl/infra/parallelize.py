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
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac

from torchtitan.models.llama3.infra.parallelize import apply_ddp
from torchtitan.models.llama4.infra.parallelize import (
    apply_compile,
    apply_fsdp,
    apply_moe_ep_tp,
)
from torchtitan.models.qwen3.infra.parallelize import _op_sac_save_list
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
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Context Parallel is not yet supported for Qwen3-VL."
        )

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    if parallel_dims.tp_enabled:
        if (
            job_config.parallelism.enable_async_tensor_parallel
            and not model_compile_enabled
        ):
            raise RuntimeError("Async TP requires torch.compile")

        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        tp_mesh = parallel_dims.get_mesh("tp")

        # Apply TP to vision encoder
        _apply_tp_to_visual(model.visual, tp_mesh)

        # Apply TP to decoder without SequenceParallel.
        # VLM needs full-sequence access between decoder blocks for vision
        # scatter and DeepStack, so hidden states stay Replicate.
        _apply_decoder_tp(
            model,
            tp_mesh,
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_async_tp=job_config.parallelism.enable_async_tensor_parallel,
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


def _apply_decoder_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
):
    """Apply tensor parallelism to the decoder without SequenceParallel.

    Unlike Qwen3's apply_non_moe_tp which uses SequenceParallel (hidden states
    are Shard(1) between blocks), this keeps hidden states as Replicate. This is
    necessary for VLM because vision scatter and DeepStack operate on the full
    sequence with boolean masks that aren't DTensor-aware.

    The trade-off is slightly higher activation memory (full sequence on each
    rank instead of 1/TP), but it avoids costly all-gather/re-shard at every
    vision scatter and DeepStack layer.
    """
    # Parallelize embedding, norm, and output — no SequenceParallel
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
            ),
            "output": ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    if enable_float8_tensorwise_tp:
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
        )

        rowwise_parallel, colwise_parallel = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
        )
    else:
        rowwise_parallel, colwise_parallel = (
            RowwiseParallel,
            ColwiseParallel,
        )

    # Apply TP to every transformer block's linear layers.
    # Norms run redundantly on Replicate data (cheap).
    for transformer_block in model.layers.values():
        layer_plan = {
            # Wrap attention inputs so rope_cache becomes a Replicate DTensor,
            # needed because wq/wk/wv outputs are DTensors and apply_rotary_emb
            # multiplies them with cos/sin from rope_cache.
            "attention": PrepareModuleInput(
                input_layouts=(Replicate(), Replicate(), None, None),
                desired_input_layouts=(Replicate(), Replicate(), None, None),
            ),
            "attention.wq": colwise_parallel(use_local_output=False),
            "attention.wk": colwise_parallel(use_local_output=False),
            "attention.wv": colwise_parallel(use_local_output=False),
            "attention.q_norm": SequenceParallel(sequence_dim=2),
            "attention.k_norm": SequenceParallel(sequence_dim=2),
            "attention.wo": rowwise_parallel(output_layouts=Replicate()),
        }

        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_parallel(output_layouts=Replicate()),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the decoder (no SequenceParallel)"
    )


def _apply_tp_to_visual(
    visual: nn.Module,
    tp_mesh: DeviceMesh,
):
    """Apply tensor parallelism to the vision encoder.

    Uses TP without SequenceParallel: data between blocks stays Replicate
    (all ranks hold full hidden_states). This is simpler because norms and
    position embeddings don't need DTensor conversion, and vision encoder
    sequence lengths are short enough that redundant norm computation is cheap.
    Memory savings come from sharding the linear layer weights.
    """
    # TP plan for each vision transformer block
    layer_plan = {
        "attn.qkv": ColwiseParallel(),
        "attn.proj": RowwiseParallel(),
        "mlp.linear_fc1": ColwiseParallel(),
        "mlp.linear_fc2": RowwiseParallel(),
    }

    for transformer_block in visual.layers.values():
        parallelize_module(transformer_block, tp_mesh, layer_plan)

    # TP plan for patch mergers (main + deepstack)
    merger_plan = {
        "linear_fc1": ColwiseParallel(),
        "linear_fc2": RowwiseParallel(),
    }

    parallelize_module(visual.merger, tp_mesh, merger_plan)
    for merger in visual.deepstack_merger_list:
        parallelize_module(merger, tp_mesh, merger_plan)

    logger.info("Applied Tensor Parallelism to the vision encoder")


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
