# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch.nn as nn

from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    apply_fsdp_to_vision_encoder,
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)

from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
    pipeline_with_first_stage_modules,
)
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from torchtitan.models.kimi_k3.layout import (
    infer_block_layout_tables_from_stages,
    layer_to_stage_from_split,
)
from torchtitan.models.kimi_k3.pipeline_stage import (
    AttnResPipelineStage,
    PPRankLocalCache,
)
from torchtitan.tools.logging import logger
from .model import KimiK3Model


def parallelize_kimi_k3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    skip_dp: bool = False,
) -> nn.Module:
    """Apply FSDP2 to the Kimi K3 decoder and vision encoder."""

    unsupported_parallelisms = [
        name
        for name, enabled in (
            ("tensor parallel", parallel_dims.tp_enabled),
            ("context parallel", parallel_dims.cp_enabled),
        )
        if enabled
    ]
    if unsupported_parallelisms:
        raise NotImplementedError(
            "Kimi K3 currently supports FSDP2 data parallelism "
            f"only; disable {', '.join(unsupported_parallelisms)}."
        )
    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError("Kimi K3 does not support model compilation yet.")

    assert isinstance(model, KimiK3Model)
    if parallelism.spmd_backend == "spmd_types":
        # Seed replicated layouts for parameters outside the explicit expert
        # declarations. Vision buffers declare their DP layouts separately.
        annotate_replicated_parameters(model, parallel_dims)

    if parallelism.spmd_backend == "spmd_types" or parallel_dims.ep_enabled:
        # model_registry's moe_comm_backend picks the dispatcher: standard
        # (default), deepep and minimal_async_ep run on this model; hybridep
        # needs GB200-class hardware.
        model.parallelize(parallel_dims)

    if ac_config is not None:
        ac_policy = ac_config.build(dump_folder=dump_folder)
        ac_policy.apply(model)
        if model.vision_encoder is not None:
            ac_policy.apply(model.vision_encoder)

    # Skip FSDP wrapper for inference. FSDP's forward hooks
    # are incompatible with torch.inference_mode() used by vLLM.
    # AC and compile are disabled via config (mode="none", enable=False).
    if skip_dp:
        return model

    if parallelism.spmd_backend == "spmd_types":
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    else:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        dp_mesh_dims = None
        edp_mesh = None
        edp_mesh_dims = None
        if parallel_dims.ep_enabled:
            edp_mesh_names = (
                ["dp_replicate", "efsdp"]
                if parallel_dims.dp_replicate_enabled
                else ["efsdp"]
            )
            edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    vision_encoder = model.vision_encoder
    if vision_encoder is not None:
        # TODO: An image batch on one DP rank and a text-only batch on another
        # execute different FSDP collectives, deadlock, and hit a 90-second
        # timeout. A general solution is needed.
        apply_fsdp_to_vision_encoder(
            vision_encoder,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            dp_mesh_dims=dp_mesh_dims,
        )

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
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model


_KIMI_ATTN_RES_LAST_STAGE_FQNS = ("output_res_proj", "output_res_norm")
# The vision tower rides with the embedding; core prunes it on the stages
# that do not hold it, and a text-only model does not have it at all.
_KIMI_K3_FIRST_STAGE_FQNS = ("vision_encoder",)


def _kimi_k3_last_stage_modules(model: nn.Module) -> tuple[str, ...]:
    """Modules the split pins next to the head: the AttnRes aggregation."""
    return tuple(n for n in _KIMI_ATTN_RES_LAST_STAGE_FQNS if hasattr(model, n))


def _module_fqns_per_model_part(model: nn.Module, kwargs: dict) -> list[list[str]]:
    """The split core applies, recomputed here to read the layer map off it.

    The same inputs give the same split on every rank, so this replaces a
    collective: the config's split when there is one, otherwise core's
    generated one with this model's pinned modules, in core's own order --
    the aggregation modules through ``last_stage_modules`` and the tower
    prepended to the first stage, the way
    ``pipeline_with_first_stage_modules`` does it.
    """
    parallelism = kwargs["parallelism"]
    if parallelism.module_fqns_per_model_part is not None:
        return parallelism.module_fqns_per_model_part
    (
        num_virtual_stages,
        num_layers,
        input_weight,
        output_weight,
    ) = _get_pipeline_metadata(
        kwargs["parallel_dims"], parallelism, kwargs["model_config"]
    )
    fqns = _generate_llm_fqn_per_model_part(
        num_virtual_stages,
        num_layers,
        input_weight,
        output_weight,
        last_stage_modules=_kimi_k3_last_stage_modules(model),
    )
    fqns[0][:0] = [
        fqn
        for fqn in _KIMI_K3_FIRST_STAGE_FQNS
        if getattr(model, fqn, None) is not None
    ]
    return fqns


def _schedule_stages(schedule: _PipelineSchedule) -> list[AttnResPipelineStage]:
    """The stages a schedule holds on this rank."""
    if isinstance(schedule, PipelineScheduleSingle):
        stages = [schedule._stage]
    elif isinstance(schedule, PipelineScheduleMulti):
        stages = list(schedule._stages)
    else:
        raise RuntimeError(
            f"Unexpected pipeline schedule class {type(schedule).__name__}."
        )
    assert all(isinstance(s, AttnResPipelineStage) for s in stages)
    return cast(list[AttnResPipelineStage], stages)


def pipeline_kimi_k3(model: nn.Module, *, attn_res_cache: bool = True, **kwargs):
    """``pipelining_fn`` for Kimi K3.

    Builds the schedule on :class:`AttnResPipelineStage` over core's split with
    this model's pinned modules, then gives every stage the routing tables
    computed from that split: the layer-to-stage map is read off the split
    and the stage-to-rank map is the schedule's own.

    ``attn_res_cache`` is a property of the transport, not of the model: with
    it, a hop carries only the blocks the receiving rank has not seen and the
    rank's store serves its later stages; without it, every hop carries the
    whole stack. A recipe turns it off with
    ``functools.partial(pipeline_kimi_k3, attn_res_cache=False)`` as the
    ``pipelining_fn``. The two transports sum the block gradients in a
    different order, so they are not bitwise against each other. Every rank
    must resolve it identically: a rank routing differently from its peers
    hangs the first hop with nothing pointing at the cause.
    """
    # Core's split places the embedding, the layers and the head. On top of
    # it the vision tower goes to the stage that holds the embedding (vision
    # features are spliced into the embeddings; nothing vision-side crosses a
    # stage boundary), which is core's first-stage entry, and the AttnRes
    # aggregation modules to the stage that holds the head, since the final
    # block attention runs there.
    (
        pp_schedule,
        model_parts,
        has_first_stage,
        has_last_stage,
    ) = pipeline_with_first_stage_modules(
        model,
        stage_class=AttnResPipelineStage,
        first_stage_module_fqns=_KIMI_K3_FIRST_STAGE_FQNS,
        last_stage_modules=_kimi_k3_last_stage_modules(model),
        **kwargs,
    )

    stages = _schedule_stages(pp_schedule)
    model_config = kwargs["model_config"]
    layer_cfgs = model_config.layers
    n_layers = len(layer_cfgs)
    layers_per_block = layer_cfgs[0].attn_res_block_size
    num_blocks = -(-n_layers // layers_per_block)
    # The split is whatever core applied, uneven stages included: the
    # config's FQNs, or the generated ones from the same metadata and the same
    # pinned modules, so every rank reads the same layer-to-stage map off it
    # with no collective; the schedule owns stage-to-rank.
    layer_to_stage = layer_to_stage_from_split(
        _module_fqns_per_model_part(model, kwargs)
    )
    layout = infer_block_layout_tables_from_stages(
        stages,
        stage_to_rank=dict(stages[0].stage_index_to_group_rank),
        num_blocks=num_blocks,
        n_layers=n_layers,
        layers_per_block=layers_per_block,
        layer_to_stage=layer_to_stage,
        cache=attn_res_cache,
    )
    store = PPRankLocalCache()
    for stage in stages:
        stage.set_routing(layout, store)
    logger.info(
        "Kimi K3 pipeline: %d stage(s) on this rank %s, block transport %s",
        len(stages),
        [s.stage_index for s in stages],
        "delta with rank store" if attn_res_cache else "whole stack every hop",
    )
    return pp_schedule, model_parts, has_first_stage, has_last_stage
