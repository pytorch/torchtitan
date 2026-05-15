# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelization utilities for Qwen3.5.

This module applies PT-D parallelisms and various training techniques
(activation checkpointing, compile, FSDP) to the Qwen3.5 model.
"""

import torch.nn as nn

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
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama4.parallelize import apply_fsdp, apply_moe_ep_tp
from torchtitan.models.qwen3_5.sharding import (
    set_deltanet_sub_module_sharding,
    set_vision_encoder_sub_module_sharding,
)
from torchtitan.tools.logging import logger


def parallelize_qwen3_5(
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
    parallelism to the Qwen3.5 model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    # Context Parallel: wrap inner attention forward BEFORE TP so CP logic
    # runs inside the local_map boundary on local tensors.
    # Applies to full attention layers only — GatedDeltaNet is recurrent
    # and allgathers the full sequence via cp=Replicate() in sharding.
    if parallel_dims.cp_enabled:
        cp_mesh = parallel_dims.get_mesh("cp")
        full_attn_inner_modules = [
            block.attn.inner_attention  # pyrefly: ignore [missing-attribute]
            for block in model.layers.values()  # pyrefly: ignore [not-callable]
            if block.layer_type == "full_attn"  # pyrefly: ignore [missing-attribute]
        ]
        if full_attn_inner_modules:
            apply_cp_to_forward(full_attn_inner_modules, cp_mesh)

    if parallel_dims.tp_enabled:
        if parallelism.enable_async_tensor_parallel and not model_compile_enabled:
            raise RuntimeError("Async TP requires torch.compile")

        tp_mesh = parallel_dims.get_mesh("tp")

        # For sub-modules built inline, set _sharding_config on built modules.
        # pyrefly: ignore [not-callable]
        for block in model.layers.values():
            # pyrefly: ignore [missing-attribute]
            if block.layer_type != "full_attn":
                # pyrefly: ignore [missing-attribute]
                set_deltanet_sub_module_sharding(block.attn)
        if model.vision_encoder is not None:
            set_vision_encoder_sub_module_sharding(model.vision_encoder)
        # pyrefly: ignore [not-callable]
        model.parallelize(tp_mesh)
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            enable_sp=parallel_dims.tp_enabled,
        )

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

    if model_compile_enabled:
        apply_compile(model, compile_config)
        if model.vision_encoder is not None:
            # pyrefly: ignore [bad-argument-type]
            apply_compile(model.vision_encoder, compile_config)

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

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

    logger.info("Applied fully_shard to the Qwen3.5 model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the Qwen3.5 model")

    return model


def pipeline_qwen3_5(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    model_config,
    **kwargs,
):
    """PP wrapper that assigns vision_encoder to the first pipeline stage.

    Delegates to ``pipeline_llm`` after injecting ``vision_encoder`` into
    the first stage's FQN list (the auto-generated LLM split doesn't know
    about vision encoder modules).
    """
    import dataclasses

    from torchtitan.distributed.pipeline_parallel import (
        _generate_llm_fqn_per_model_part,
        _get_pipeline_metadata,
        pipeline_llm,
    )

    if parallelism.module_fqns_per_model_part is None:
        (
            num_virtual_stages,
            num_layers,
            input_weight,
            output_weight,
        ) = _get_pipeline_metadata(parallel_dims, parallelism, model_config)
        fqn_per_part = _generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )
        # Vision encoder lives on the first stage alongside tok_embeddings
        if hasattr(model, "vision_encoder") and model.vision_encoder is not None:
            fqn_per_part[0].insert(0, "vision_encoder")
        parallelism = dataclasses.replace(
            parallelism, module_fqns_per_model_part=fqn_per_part
        )

    return pipeline_llm(
        model,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        model_config=model_config,
        **kwargs,
    )
