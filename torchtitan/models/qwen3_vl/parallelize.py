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
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.components.quantization import find_pad_multiple

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
from torchtitan.distributed.tensor_parallel import NoParallel
from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.models.llama4.parallelize import apply_fsdp, apply_moe_ep_tp
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def _apply_non_moe_tp_to_decoder(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_async_tp: bool,
):
    """Apply tensor parallelism to the decoder with full DTensor.

    Hidden states flow as DTensor(Replicate) between layers. The embedding
    output is plain tensor (needed for vision scatter before the decoder loop),
    but each transformer block wraps its input as DTensor via PrepareModuleInput.
    DeepStack boolean indexing handles DTensor via to_local in the 
    """
    top_level_plan = {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
            use_local_output=True,  # plain tensor needed for _scatter_vision_embeds
        ),
        "norm": NoParallel(),
        "output": ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=not loss_parallel,
        ),
    }
    # pyrefly: ignore [bad-argument-type]
    parallelize_module(model, tp_mesh, top_level_plan)

    # Detect whether fused QKV is used by checking the first layer
    # pyrefly: ignore [not-callable]
    first_block = next(iter(model.layers.values()))
    use_fused_qkv = isinstance(
        first_block.attention.qkv_linear,  # pyrefly: ignore [missing-attribute]
        FusedQKVLinear,
    )

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        if use_fused_qkv:
            qkv_plan = {
                "attention.qkv_linear.wqkv": ColwiseParallel(use_local_output=False),
            }
        else:
            qkv_plan = {
                "attention.qkv_linear.wq": ColwiseParallel(use_local_output=False),
                "attention.qkv_linear.wk": ColwiseParallel(use_local_output=False),
                "attention.qkv_linear.wv": ColwiseParallel(use_local_output=False),
            }

        # Wrap hidden_states as DTensor(Replicate) at block entry, and
        # convert back to plain tensor at block exit. This keeps DTensor
        # inside each block (for TP/EP) while returning plain tensors
        # between blocks for DeepStack boolean indexing.
        parallelize_module(
            transformer_block,
            tp_mesh,
            PrepareModuleInputOutput(
                input_layouts=(Replicate(), Replicate(), None, None),
                desired_input_layouts=(Replicate(), Replicate(), None, None),
                output_layouts=(Replicate(),),
                desired_output_layouts=(Replicate(),),
                use_local_output=True,
            ),
        )

        layer_plan = {
            "attention_norm": NoParallel(),
            "ffn_norm": NoParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Replicate(), Replicate(), None, None),
                desired_input_layouts=(Replicate(), Replicate(), None, None),
            ),
            **qkv_plan,
            "attention.q_norm": SequenceParallel(
                sequence_dim=2, use_local_output=False
            ),
            "attention.k_norm": SequenceParallel(
                sequence_dim=2, use_local_output=False
            ),
            "attention.wo": RowwiseParallel(
                output_layouts=Replicate(),
                use_local_output=False,
            ),
        }

        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward.w1": ColwiseParallel(use_local_output=False),
                    "feed_forward.w2": RowwiseParallel(
                        output_layouts=Replicate(),
                        use_local_output=False,
                    ),
                    "feed_forward.w3": ColwiseParallel(use_local_output=False),
                }
            )

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
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

    Hidden states flow as DTensor(Replicate) throughout — all ranks hold the
    full hidden_states. Only the linear layers (qkv, proj, fc1, fc2) are
    sharded via ColwiseParallel/RowwiseParallel to save memory. Norms operate
    on Replicate DTensors directly.

    The learned position embedding uses local_map to unwrap DTensor for
    bilinear interpolation. TODO: Add a DTensor sharding prop rule for
    F.interpolate (at least Replicate → Replicate) upstream in PyTorch,
    then remove the local_map workaround in vision_encoder.py.
    """
    # NoParallel on patch_embed distributes its params as Replicate DTensors
    # on tp_mesh for FSDP mesh consistency. Its input hook wraps plain
    # pixel_values as DTensor(Replicate), and the output stays as DTensor
    # (Replicate) to flow through the rest of the vision encoder.
    parallelize_module(vision_encoder, tp_mesh, {"patch_embed": NoParallel()})

    # pos_embed is an nn.Parameter (not a submodule), so it can't be targeted
    # by parallelize_module's plan dict. We distribute it as Replicate DTensor
    # on tp_mesh for FSDP mesh consistency. The vision encoder's
    # compute_position_embeddings uses local_map to unwrap it for interpolation.
    vision_encoder.pos_embed = nn.Parameter(
        # pyrefly: ignore [bad-argument-type]
        distribute_tensor(vision_encoder.pos_embed.data, tp_mesh, [Replicate()]),
        requires_grad=vision_encoder.pos_embed.requires_grad,
    )

    # TP plan for each vision transformer block.
    # hidden_states flows through as DTensor (Replicate) so residual adds work.
    # NoParallel on norms sets their params as Replicate DTensors on tp_mesh
    # (for consistent (fsdp, tp) mesh after FSDP).
    # RowwiseParallel uses use_local_output=False to return DTensor (Replicate)
    # so residual connections (hidden_states + attn/mlp output) stay in DTensor
    # space. ColwiseParallel uses use_local_output=True (default) to return
    # local shards for the internal attention/MLP computation.
    layer_plan = {
        "norm1": NoParallel(),
        "norm2": NoParallel(),
        "attn.qkv": ColwiseParallel(),  # needs plain tensor for reshape after qkv
        "attn.proj": RowwiseParallel(use_local_output=False),
        "mlp.linear_fc1": ColwiseParallel(use_local_output=False),
        "mlp.linear_fc2": RowwiseParallel(use_local_output=False),
    }

    # pyrefly: ignore [not-callable]
    for transformer_block in vision_encoder.layers.values():
        # pyrefly: ignore [bad-argument-type]
        parallelize_module(transformer_block, tp_mesh, layer_plan)

    # TP plan for patch mergers (main + deepstack).
    # RowwiseParallel uses default use_local_output=True so mergers return
    # plain tensors — their outputs are used in vision scatter and DeepStack
    # which operate on plain tensors with boolean masks.
    merger_plan = {
        "norm": NoParallel(),
        "linear_fc1": ColwiseParallel(use_local_output=False),
        "linear_fc2": RowwiseParallel(),
    }

    # pyrefly: ignore [bad-argument-type]
    parallelize_module(vision_encoder.merger, tp_mesh, merger_plan)
    # pyrefly: ignore [not-iterable]
    for merger in vision_encoder.deepstack_merger_list:
        # pyrefly: ignore [bad-argument-type]
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
            # pyrefly: ignore [bad-argument-type]
            _apply_tp_to_vision_encoder(model.vision_encoder, tp_mesh)

        # Apply TP to decoder without SequenceParallel.
        # VLM needs full-sequence access between decoder blocks for vision
        # scatter and DeepStack, so hidden states stay as replicated plain tensors.
        _apply_non_moe_tp_to_decoder(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_async_tp=parallelism.enable_async_tensor_parallel,
        )

    # Apply MoE parallelism to decoder layers.
    # enable_sp=False because Qwen3-VL keeps hidden states as plain tensors
    # (not Shard(1) DTensors) for vision scatter and DeepStack.
    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            enable_sp=False,
            pad_multiple=find_pad_multiple(model_converters.converters),
        )


    # Apply activation checkpointing
    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )
        if model.vision_encoder is not None:
            # pyrefly: ignore [bad-argument-type]
            apply_ac(model.vision_encoder, ac_config)

    # Apply torch.compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, compile_config)
    if compile_config.enable:
        if model.vision_encoder is not None:
            # pyrefly: ignore [bad-argument-type]
            apply_compile(model.vision_encoder, compile_config)

    # Apply FSDP / HSDP unconditionally (fully_shard handles dp_shard=1)
    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    # FSDP the vision encoder as a single unit (see _apply_fsdp_to_vision_encoder)
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
    )

    logger.info("Applied fully_shard to the Qwen3-VL model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the Qwen3-VL model")

    return model
