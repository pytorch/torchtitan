# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)

from torch.distributed.pipelining.stage import _PipelineStageBase, PipelineStage

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
    llm_split_with_pinned_modules,
    pipeline_llm,
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


def _as_attn_res_stage(stage: _PipelineStageBase) -> AttnResPipelineStage:
    """``stage`` rebuilt as an :class:`AttnResPipelineStage` from its own fields.

    Core builds plain ``PipelineStage``s and K3 swaps each for its subclass
    here, rather than threading a stage class through core's pipelining. The
    rebuilt stage wraps the same, already parallelized module, so the model
    parts core returned are still the objects the stages run. Building a stage
    only reads its process group (the per-direction P2P groups, when enabled,
    are cached per group), so nothing collective runs a second time.
    """
    assert isinstance(stage, PipelineStage)
    rebuilt = AttnResPipelineStage(
        stage.submod,
        stage.stage_index,
        stage.num_stages,
        stage.device,
        group=stage.group,
        dw_builder=stage.dw_builder,
        get_mesh=stage._mesh_cache._get_mesh_cb,
    )
    # The schedule wrote its stage-to-rank map onto the stage it was handed.
    rebuilt.stage_index_to_group_rank = stage.stage_index_to_group_rank
    return rebuilt


def _swap_in_attn_res_stages(
    schedule: _PipelineSchedule,
) -> list[AttnResPipelineStage]:
    """Replace the stages a schedule holds on this rank with AttnRes stages.

    A schedule keeps its stages in ``_stage`` (one per rank) or ``_stages``
    (several per rank) and nowhere else, so replacing those is the whole swap.
    """
    if isinstance(schedule, PipelineScheduleSingle):
        rebuilt = _as_attn_res_stage(schedule._stage)
        schedule._stage = rebuilt
        return [rebuilt]
    if isinstance(schedule, PipelineScheduleMulti):
        rebuilt_stages = [_as_attn_res_stage(s) for s in schedule._stages]
        held: list[_PipelineStageBase] = list(rebuilt_stages)
        schedule._stages = held
        return rebuilt_stages
    raise RuntimeError(f"Unexpected pipeline schedule class {type(schedule).__name__}.")


def pipeline_kimi_k3(model: nn.Module, *, attn_res_cache: bool = True, **kwargs):
    """``pipelining_fn`` for Kimi K3.

    Builds the schedule with core's pipelining over core's split with this
    model's pinned modules, rebuilds each stage it holds on this rank as an
    :class:`AttnResPipelineStage` from the constructed stage's own fields (a
    small local swap instead of an intrusive change to core's stage
    construction; whether a stage subclass is the right abstraction for the
    attention residual is still open), then gives every stage the routing tables
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
    # stage boundary) and the AttnRes aggregation modules to the stage that
    # holds the head, since the final block attention runs there. Core builds
    # that split; this function keeps the one object and both hands it over
    # and reads the layer map off it, so no rank recomputes it and none of
    # them can disagree.
    parallelism = kwargs.pop("parallelism")
    module_fqns_per_model_part = parallelism.module_fqns_per_model_part
    if module_fqns_per_model_part is None:
        module_fqns_per_model_part, parallelism = llm_split_with_pinned_modules(
            model,
            parallel_dims=kwargs["parallel_dims"],
            parallelism=parallelism,
            model_config=kwargs["model_config"],
            first_stage_module_fqns=_KIMI_K3_FIRST_STAGE_FQNS,
            last_stage_module_fqns=_kimi_k3_last_stage_modules(model),
        )

    pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_llm(
        model,
        parallelism=parallelism,
        **kwargs,
    )

    stages = _swap_in_attn_res_stages(pp_schedule)
    model_config = kwargs["model_config"]
    layer_cfgs = model_config.layers
    n_layers = len(layer_cfgs)
    layers_per_block = layer_cfgs[0].attn_res_block_size
    num_blocks = -(-n_layers // layers_per_block)
    # The split is whatever core applied, uneven stages included: the
    # config's FQNs, or the generated ones from the same metadata and the same
    # pinned modules, so every rank reads the same layer-to-stage map off it
    # with no collective; the schedule owns stage-to-rank.
    layer_to_stage = layer_to_stage_from_split(module_fqns_per_model_part)
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
