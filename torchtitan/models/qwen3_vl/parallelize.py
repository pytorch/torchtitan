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
from torch.distributed.tensor import distribute_tensor, Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.expert_parallel import (
    ReordererSequenceParallel,
    TensorParallel,
)
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.tensor_parallel import (
    ColwiseParallelWithGradPlacement,
    NoParallel,
)
from torchtitan.models.llama3.parallelize import apply_replicate
from torchtitan.models.llama4.parallelize import (
    apply_compile,
    apply_fsdp,
    apply_moe_ep_tp,
)
from torchtitan.models.qwen3.parallelize import _op_sac_save_list
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def _apply_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    ep_mesh: DeviceMesh | None,
    etp_mesh: DeviceMesh | None,
):
    """Apply TP to MoE layers for VLM (no SequenceParallel).

    Unlike llama4's apply_moe_ep_tp which assumes Shard(1) hidden states
    (from SequenceParallel), this uses Replicate input/output layouts because
    Qwen3-VL keeps hidden states as Replicate for vision scatter and DeepStack.

    The key difference: MoE output uses Partial → Replicate (all-reduce)
    instead of Partial → Shard(1) (reduce-scatter).
    """
    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue

        moe_layer_plan = {
            "moe": PrepareModuleInputOutput(
                input_layouts=(Replicate(),),
                desired_input_layouts=(Replicate(),),
                use_local_input=False,
                output_layouts=(Partial(),),
                desired_output_layouts=(Replicate(),),
            ),
            "moe.router.gate": NoParallel(
                local_output_grad_placements=(Partial(),),
            ),
        }

        if ep_mesh is not None and etp_mesh is None:
            moe_layer_plan.update({"moe.reorderer": ReordererSequenceParallel()})

        if transformer_block.moe.shared_experts is not None:
            moe_layer_plan.update(
                {
                    "moe.shared_experts.w1": ColwiseParallelWithGradPlacement(
                        local_input_grad_placements=(Partial(),)
                    ),
                    "moe.shared_experts.w2": RowwiseParallel(
                        output_layouts=Partial(),
                    ),
                    "moe.shared_experts.w3": ColwiseParallelWithGradPlacement(
                        local_input_grad_placements=(Partial(),)
                    ),
                }
            )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=moe_layer_plan,
        )

        # Apply TensorParallel to experts when no EP (TP-only MoE)
        if ep_mesh is None:
            parallelize_module(
                module=transformer_block.moe.experts,
                device_mesh=tp_mesh,
                parallelize_plan=TensorParallel(),
            )


def _apply_tp_to_decoder(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
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
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=False),
            "attention.q_norm": SequenceParallel(sequence_dim=2),
            "attention.k_norm": SequenceParallel(sequence_dim=2),
            "attention.wo": RowwiseParallel(output_layouts=Replicate()),
        }

        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward.w1": ColwiseParallel(),
                    "feed_forward.w2": RowwiseParallel(output_layouts=Replicate()),
                    "feed_forward.w3": ColwiseParallel(),
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
        f"Applied {'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the decoder (no SequenceParallel)"
    )


def _apply_tp_to_vision_encoder(
    vision_encoder: nn.Module,
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
    parallelize_module(vision_encoder, tp_mesh, {"patch_embed": NoParallel()})

    # pos_embed is an nn.Parameter (not a submodule) used with direct indexing
    # and bilinear interpolation. We convert it to a Replicate DTensor on
    # tp_mesh for mesh consistency. compute_position_embeddings() calls
    # .to_local() on it before indexing.
    vision_encoder.pos_embed = nn.Parameter(
        distribute_tensor(vision_encoder.pos_embed.data, tp_mesh, [Replicate()]),
        requires_grad=vision_encoder.pos_embed.requires_grad,
    )

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

    for transformer_block in vision_encoder.layers.values():
        parallelize_module(transformer_block, tp_mesh, layer_plan)

    # TP plan for patch mergers (main + deepstack).
    merger_plan = {
        "norm": NoParallel(),
        "linear_fc1": ColwiseParallel(),
        "linear_fc2": RowwiseParallel(),
    }

    parallelize_module(vision_encoder.merger, tp_mesh, merger_plan)
    for merger in vision_encoder.deepstack_merger_list:
        parallelize_module(merger, tp_mesh, merger_plan)

    logger.info("Applied Tensor Parallelism to the vision encoder")


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

    This wraps the entire vision encoder with one fully_shard call so that all
    parameters are gathered in a single AllGather. Since FSDP2 lacks forward
    prefetching, layer-wise sharding would serialize AllGather and compute
    per layer. A single wrap avoids this overhead.

    This must be called before the llama4 apply_fsdp so that the vision encoder
    is sharded before the final fully_shard(model).
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


def _apply_compile_to_vision_encoder(vision_encoder: nn.Module, compile_config):
    """Apply torch.compile to vision encoder transformer blocks."""
    for layer_id, transformer_block in vision_encoder.layers.named_children():
        transformer_block = torch.compile(
            transformer_block,
            backend=compile_config.backend,
            fullgraph=True,
        )
        vision_encoder.layers.register_module(layer_id, transformer_block)
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

        tp_mesh = parallel_dims.get_mesh("tp")

        # Apply TP to vision encoder
        if model.vision_encoder is not None:
            _apply_tp_to_vision_encoder(model.vision_encoder, tp_mesh)

        # Apply TP to decoder without SequenceParallel.
        # VLM needs full-sequence access between decoder blocks for vision
        # scatter and DeepStack, so hidden states stay Replicate.
        _apply_tp_to_decoder(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_async_tp=parallelism.enable_async_tensor_parallel,
        )

    # Apply MoE parallelism to decoder layers.
    # TP plan is handled separately via _apply_moe_tp because Qwen3-VL keeps
    # hidden states as Replicate (not Shard(1)), requiring Partial → Replicate
    # (all-reduce) at the MoE output instead of Partial → Shard(1) (reduce-scatter).
    if parallel_dims.tp_enabled:
        _apply_moe_tp(
            model,
            tp_mesh=parallel_dims.get_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
        )

    # Apply EP to experts (handles all-to-all token dispatch/combine).
    # Pass tp_mesh=None so apply_moe_ep_tp only applies the EP experts plan
    # without overriding the TP plan we applied above.
    if parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=None,
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
        if model.vision_encoder is not None:
            apply_ac(model.vision_encoder, ac_config)

    # Apply torch.compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, compile_config, parallel_dims.ep_enabled)
    if compile_config.enable:
        if model.vision_encoder is not None:
            _apply_compile_to_vision_encoder(model.vision_encoder, compile_config)

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
        if model.vision_encoder is not None:
            _apply_fsdp_to_vision_encoder(
                model.vision_encoder,
                dp_mesh,
                param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
                reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
                reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
                pp_enabled=parallel_dims.pp_enabled,
            )

        # FSDP the decoder with MoE-aware sharding (reuses llama4 apply_fsdp)
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
