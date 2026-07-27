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

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.fsdp import MixedPrecisionPolicy

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp


def _apply_fsdp_to_vision_encoder(
    vision_encoder: nn.Module,
    dp_mesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    reshard_after_forward_policy: str = "default",
    pp_enabled: bool = False,
):
    """FSDP the vision encoder as a single unit.

    One AllGather for all vision params is more efficient than per-layer
    sharding — the vision encoder is small relative to the decoder.
    Must be called before apply_fsdp on the decoder.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled=pp_enabled
    )
    fully_shard(
        vision_encoder,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )


def parallelize_qwen3_5(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    skip_dp: bool = False,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the Qwen3.5 model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    if parallelism.spmd_backend == "full_dtensor":
        raise NotImplementedError("full_dtensor is not supported yet.")

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Context Parallel is not yet supported for Qwen3.5. "
            "GatedDeltaNet (75% of layers) requires full-sequence allgather, "
            "and multimodal CP needs vision scatter before CP sharding."
        )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        if parallelism.enable_async_tensor_parallel and not model_compile_enabled:
            raise RuntimeError("Async TP requires torch.compile")

        # pyrefly: ignore [not-callable]
        model.parallelize(parallel_dims)

    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    if ac_config is not None:
        ac_policy = ac_config.build(dump_folder=dump_folder)
        ac_policy.apply(model)
        if model.vision_encoder is not None:
            ac_policy.apply(model.vision_encoder)

    if model_compile_enabled:
        apply_compile(model, compile_config)
        if model.vision_encoder is not None:
            # pyrefly: ignore [bad-argument-type]
            apply_compile(model.vision_encoder, compile_config)

    # Skip FSDP for inference (vLLM): FSDP forward hooks are incompatible with
    # torch.inference_mode(). TP/EP sharding above is kept.
    if skip_dp:
        return model

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    if model.vision_encoder is not None:
        _apply_fsdp_to_vision_encoder(
            model.vision_encoder,  # pyrefly: ignore [bad-argument-type]
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=parallel_dims.pp_enabled,
        )

    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp_to_decoder(
        model,  # pyrefly: ignore [bad-argument-type]
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
    )

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
        # Vision encoder lives on the first stage alongside tok_embeddings. This
        # adds load to stage 0 that the auto split doesn't model (input_weight
        # only accounts for tok_embeddings); for a heavy vision encoder, bump
        # parallelism.pipeline_parallel_first_stage_less_layers to rebalance.
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
