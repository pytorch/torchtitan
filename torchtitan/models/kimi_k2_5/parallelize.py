# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization for Kimi K2.5 (MoonViT3d vision encoder + DeepSeekV3 decoder).

TP/SP/EP is applied by ``model.parallelize(parallel_dims)`` from the
``ShardingConfig`` that ``set_kimi_k2_5_sharding_config`` sets on every
sub-config (see ``sharding.py``). FSDP is then applied in two parts: the vision
encoder as a single unit (its compute is small, so one all-gather beats many
per-layer ones), then the decoder with MoE-aware per-layer wrapping. Context
Parallel and ``full_dtensor`` are not supported.
"""

import torch.nn as nn

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
    apply_fsdp_to_vision_encoder,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp


def parallelize_kimi_k2_5(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Apply TP/EP, activation checkpointing, ``torch.compile``, and FSDP.

    Order: config-based TP/EP (``model.parallelize``) -> async-TP ->
    activation checkpointing -> compile -> FSDP (vision encoder as a single
    unit, then the MoE-aware decoder).

    NOTE: the passed-in model should preferably be on meta device; otherwise it
    must fit in GPU or CPU memory.
    """
    if parallelism.spmd_backend == "full_dtensor":
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
            apply_compile(model.vision_encoder, compile_config)

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    # FSDP the vision encoder as a single unit, before the decoder's FSDP.
    #
    # Skipped under PP: as its own fully_shard root, the encoder's forward may
    # not run on a given microbatch (a text-only batch, or a stage that holds the
    # encoder but gets no pixels), so its params are never all-gathered -- which
    # trips FSDP's per-microbatch post-backward callback. Under PP we leave it
    # unwrapped so its params fall into the root ``fully_shard(model)`` in
    # ``apply_fsdp_to_decoder``.
    if model.vision_encoder is not None and not parallel_dims.pp_enabled:
        apply_fsdp_to_vision_encoder(
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

    apply_fsdp_to_decoder(
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

    return model
