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
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.components.quantization.float8 import find_float8_linear_config
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.tensor_parallel import NoParallel
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.models.llama3.parallelize import apply_replicate
from torchtitan.models.llama4.parallelize import (
    apply_compile,
    apply_fsdp,
    apply_moe_ep_tp,
)
from torchtitan.models.qwen3.parallelize import _op_sac_save_list
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def _replicate_module_params(module: nn.Module, mesh: DeviceMesh):
    """Convert a module's direct (non-recursive) parameters to Replicate DTensors."""
    for name, param in module.named_parameters(recurse=False):
        replicated = nn.Parameter(
            distribute_tensor(param.data, mesh, [Replicate()]),
            requires_grad=param.requires_grad,
        )
        setattr(module, name, replicated)


def _apply_tp_to_decoder(
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
    top_level_plan = {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        "norm": NoParallel(),
        "output": ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=not loss_parallel,
        ),
    }
    parallelize_module(model, tp_mesh, top_level_plan)

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
    # NoParallel on norms sets their params as Replicate DTensors on tp_mesh
    # (for consistent (fsdp, tp) mesh after FSDP) and inserts I/O hooks that
    # convert local tensor ↔ DTensor at the norm boundary, keeping the block's
    # data path in local-tensor space as RowwiseParallel(use_local_output=True)
    # expects.
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": NoParallel(),
            "ffn_norm": NoParallel(),
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
    # Use NoParallel on patch_embed so its params become Replicate DTensors
    # on tp_mesh (ensuring a consistent (fsdp, tp) mesh after FSDP), while
    # its I/O hooks convert plain pixel_values to DTensor on entry and back
    # to local tensor on exit — avoiding mixed tensor/DTensor errors with
    # pos_embeds (which are computed as plain tensors).
    parallelize_module(visual, tp_mesh, {"patch_embed": NoParallel()})

    # pos_embed.weight is accessed directly (not through forward), so we
    # just need its weight to be a DTensor on tp_mesh for mesh consistency.
    # The vision encoder's compute_position_embeddings() already calls
    # .to_local() on it, so the downstream pos_embeds stays a plain tensor.
    _replicate_module_params(visual.pos_embed, tp_mesh)

    # TP plan for each vision transformer block.
    # NoParallel on norms sets their params as Replicate DTensors on tp_mesh
    # (for consistent (fsdp, tp) mesh after FSDP) and inserts I/O hooks that
    # convert local tensor → DTensor on entry and DTensor → local tensor on
    # exit. This keeps the block's data path in local-tensor space (as
    # ColwiseParallel/RowwiseParallel with use_local_output=True expect).
    layer_plan = {
        "norm1": NoParallel(),
        "norm2": NoParallel(),
        "attn.qkv": ColwiseParallel(),
        "attn.proj": RowwiseParallel(),
        "mlp.linear_fc1": ColwiseParallel(),
        "mlp.linear_fc2": RowwiseParallel(),
    }

    for transformer_block in visual.layers.values():
        parallelize_module(transformer_block, tp_mesh, layer_plan)

    # TP plan for patch mergers (main + deepstack).
    merger_plan = {
        "norm": NoParallel(),
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
            reshard_after_forward = True
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

    # Shard deepstack mergers
    if hasattr(model.visual, "deepstack_merger_list"):
        for merger in model.visual.deepstack_merger_list:
            fully_shard(
                merger,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    # Shard pos_embed and any other remaining visual parameters by wrapping
    # the visual module itself. Sub-modules (patch_embed, layers, merger,
    # deepstack_merger_list) are already individually wrapped above.
    # This top-level wrap captures pos_embed.weight and rotary_pos_emb.
    fully_shard(
        model.visual,
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


def parallelize_qwen3_vl(
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
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the Qwen3-VL model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    assert isinstance(model.visual, nn.Module), "model.visual must be an nn.Module"

    # Validate sequence length divisibility
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # Check attention type compatibility
    if parallel_dims.cp_enabled:
        raise NotImplementedError("Context Parallel is not yet supported for Qwen3-VL.")

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if parallel_dims.tp_enabled:
        if parallelism.enable_async_tensor_parallel and not model_compile_enabled:
            raise RuntimeError("Async TP requires torch.compile")

        float8_config = find_float8_linear_config(model_converters.converters)
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = float8_config is not None and float8_config.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        tp_mesh = parallel_dims.get_mesh("tp")

        # Apply TP to vision encoder
        if model.visual is not None:
            _apply_tp_to_visual(model.visual, tp_mesh)

        # Apply TP to decoder without SequenceParallel.
        # VLM needs full-sequence access between decoder blocks for vision
        # scatter and DeepStack, so hidden states stay Replicate.
        _apply_tp_to_decoder(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_async_tp=parallelism.enable_async_tensor_parallel,
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
    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_op_sac_save_list,
        )
        if model.visual is not None:
            apply_ac(model.visual, ac_config)

    # Apply torch.compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, compile_config, parallel_dims.ep_enabled)
    if compile_config.enable:
        if model.visual is not None:
            _apply_compile_to_visual(model.visual, compile_config)

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
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        )

        # FSDP the decoder with MoE-aware sharding (reuses llama4 apply_fsdp)
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=False,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the Qwen3-VL model")
        else:
            logger.info("Applied FSDP to the Qwen3-VL model")

        if training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the Qwen3-VL model")

    elif parallel_dims.dp_replicate_enabled:
        apply_replicate(
            model,
            parallel_dims.get_mesh("dp_replicate"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

    return model
