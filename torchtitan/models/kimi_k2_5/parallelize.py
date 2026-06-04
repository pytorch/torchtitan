# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization utilities for Kimi K2.5 (MoonViT3d vision encoder + the
DeepSeekV3-based decoder).

Tensor/expert parallelism is applied uniformly by ``model.parallelize(parallel_dims)``,
which reads the ``ShardingConfig`` that ``set_kimi_k2_5_sharding_config`` places
on every decoder *and* vision-encoder sub-config (config-based sharding via the
Module protocol — no manual ``parallelize_module`` plans).

FSDP is then applied in two parts: the vision encoder as a single unit (its
compute is small relative to the decoder, so one all-gather beats many per-layer
ones), then the decoder with MoE-aware per-layer wrapping.

SequenceParallel is intentionally not used: the decoder keeps hidden states
``Replicate`` so the multimodal forward can scatter vision features across the
full sequence. Context Parallel and ``full_dtensor`` are not supported.
"""

import dataclasses

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.fsdp import MixedPrecisionPolicy

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
from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
    pipeline_llm,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama4.parallelize import apply_fsdp
from torchtitan.tools.logging import logger


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
    sharding — the vision encoder is small relative to the decoder. Must be
    called before ``apply_fsdp`` on the decoder so the encoder is already
    sharded when the final ``fully_shard(model)`` runs.
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


def parallelize_kimi_k2_5(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply TP/EP, activation checkpointing, ``torch.compile``, and FSDP.

    Order: config-based TP/EP (``model.parallelize``) -> async-TP ->
    activation checkpointing -> compile -> FSDP (vision encoder as a single
    unit, then the MoE-aware decoder).

    NOTE: the passed-in model should preferably be on meta device; otherwise it
    must fit in GPU or CPU memory.
    """
    if parallelism.full_dtensor:
        raise NotImplementedError("full_dtensor is not supported yet.")

    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Context Parallel is not yet supported for Kimi K2.5: vision scatter "
            "needs the full sequence before CP would shard it."
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        if parallelism.enable_async_tensor_parallel and not model_compile_enabled:
            raise RuntimeError("Async TP requires torch.compile")
        # Config-based TP/EP for both the decoder and the vision encoder.
        model.parallelize(parallel_dims)

    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )
        if model.vision_encoder is not None:
            apply_ac(
                model.vision_encoder,
                ac_config,
                model_compile_enabled=model_compile_enabled,
                base_folder=dump_folder,
            )

    if model_compile_enabled:
        apply_compile(model, compile_config)
        if model.vision_encoder is not None:
            apply_compile(model.vision_encoder, compile_config)

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    # FSDP the vision encoder as a single unit, before the decoder's apply_fsdp.
    #
    # Skipped under PP: a standalone fully_shard root whose forward does not run
    # on a given microbatch (the text-only path, or a stage that holds the
    # vision encoder but receives no pixels) leaves its parameters sharded
    # (never all-gathered), which trips FSDP's per-microbatch post-backward
    # callback. Under PP we instead let the root ``fully_shard(model)`` inside
    # ``apply_fsdp`` absorb the vision-encoder params into the always-executed
    # root group.
    if model.vision_encoder is not None and not parallel_dims.pp_enabled:
        _apply_fsdp_to_vision_encoder(
            model.vision_encoder,
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

    logger.info("Applied fully_shard to the Kimi K2.5 model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the Kimi K2.5 model")

    return model


def pipeline_kimi_k2_5(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    model_config,
    **kwargs,
):
    """Pipeline-parallel wrapper that puts the vision encoder on the first stage.

    The auto-generated LLM stage split (``_generate_llm_fqn_per_model_part``)
    only knows about the decoder modules (``tok_embeddings``, ``layers.*``,
    ``norm``, ``lm_head``). We inject ``vision_encoder`` into the first stage's
    FQN list so it is co-located with ``tok_embeddings`` (vision features are
    scattered into the embedding sequence before the decoder layers), then
    delegate to ``pipeline_llm``.

    On stages other than the first, ``tok_embeddings`` and ``vision_encoder``
    are pruned to ``None``; ``KimiK25Model.forward`` already guards on
    ``self.tok_embeddings is not None`` and only touches the vision encoder on
    the embedding stage, so the multimodal logic is skipped there.
    """
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
        if getattr(model, "vision_encoder", None) is not None:
            # Co-locate the vision encoder with tok_embeddings on stage 0.
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
