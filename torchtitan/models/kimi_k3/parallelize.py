# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
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
)

from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    get_schedule_class,
    pipeline_llm,
)
from torchtitan.models.kimi_k3.layout import (
    gather_layer_to_stage,
    infer_block_layout_tables_from_stages,
)
from torchtitan.models.kimi_k3.pipeline_stage import AttnResPipelineStage, RankStore
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
    if parallelism.spmd_backend != "partial_dtensor":
        raise NotImplementedError(
            "Kimi K3 FSDP2 currently supports the partial_dtensor SPMD backend "
            "only; the config registry pins it."
        )
    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError("Kimi K3 does not support model compilation yet.")

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
    # The routed experts shard on their own data-parallel mesh, which excludes
    # the expert axis; the same shape deepseek_v3 resolves.
    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh = parallel_dims.get_optional_mesh(
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )

    assert isinstance(model, KimiK3Model)
    if parallel_dims.ep_enabled:
        # model_registry's moe_comm_backend picks the dispatcher: standard
        # (default), deepep and minimal_async_ep run on this model; hybridep
        # needs GB200-class hardware.
        model.parallelize(parallel_dims)

    if ac_config is not None:
        ac_policy = ac_config.build(dump_folder=dump_folder)
        ac_policy.apply(model)
        if model.vision_encoder is not None:
            ac_policy.apply(model.vision_encoder)

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
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model


_KIMI_ATTN_RES_LAST_STAGE_FQNS = ("output_res_proj", "output_res_norm")


def kimi_k3_module_fqns_per_model_part(
    model: nn.Module,
    *,
    model_config,
    parallelism,
    pp: int,
) -> list[list[str]] | None:
    """The pipeline split of a Kimi K3 model, built from its config.

    Core's layer distribution (``_generate_llm_fqn_per_model_part``) places the
    embedding, the layers and the head; on top of it this model needs the
    AttnRes aggregation modules (``output_res_proj``, ``output_res_norm``) on
    the stage that holds ``lm_head``, since the final block attention runs
    there, and the vision tower on the stage that holds the embedding, since
    vision features are spliced into the embeddings and nothing vision-side
    crosses a stage boundary. Returns None when the split does not apply (no
    pipeline parallelism, or a config without layers); the caller keeps
    whatever split the user configured.
    """
    if pp <= 1 or model_config is None:
        return None
    layers = getattr(model_config, "layers", None)
    if layers is None:
        return None
    num_layers = len(layers)
    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage
    # The stage count core's _get_pipeline_metadata derives from the same fields.
    if layers_per_stage is not None:
        num_virtual_stages = math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )
    else:
        schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
        stages_per_rank = 1 if issubclass(schedule_class, PipelineScheduleSingle) else 2
        num_virtual_stages = pp * stages_per_rank
    fqns = _generate_llm_fqn_per_model_part(
        num_virtual_stages, num_layers, input_weight, output_weight
    )
    # Core spells the head ``output``; this model calls it ``lm_head``. Any
    # FQN matching no child makes core set that child to None on every stage.
    fqns = [["lm_head" if n == "output" else n for n in stage] for stage in fqns]
    tail = [n for n in _KIMI_ATTN_RES_LAST_STAGE_FQNS if hasattr(model, n)]
    fqns[-1].extend(tail)
    if getattr(model, "vision_encoder", None) is not None:
        embed_stage = next(
            (stage for stage in fqns if "tok_embeddings" in stage), fqns[0]
        )
        embed_stage.append("vision_encoder")
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

    Splits the model with this model's names, builds the schedule on
    :class:`AttnResPipelineStage`, then gives every stage the routing tables
    computed from the split the trainer applied: the layer-to-stage map is one
    all-gather over the pipeline group, and the stage-to-rank map is the
    schedule's own.

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
    import dataclasses

    parallelism = kwargs["parallelism"]
    if parallelism.module_fqns_per_model_part is None:
        fqns = kimi_k3_module_fqns_per_model_part(
            model,
            model_config=kwargs.get("model_config"),
            parallelism=parallelism,
            pp=kwargs["parallel_dims"].pp,
        )
        if fqns is not None:
            kwargs["parallelism"] = dataclasses.replace(
                parallelism, module_fqns_per_model_part=fqns
            )
    pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_llm(
        model, stage_class=AttnResPipelineStage, **kwargs
    )

    stages = _schedule_stages(pp_schedule)
    model_config = kwargs["model_config"]
    layer_cfgs = model_config.layers
    n_layers = len(layer_cfgs)
    layers_per_block = layer_cfgs[0].attn_res_block_size
    num_blocks = -(-n_layers // layers_per_block)
    # The split is whatever the trainer applied, uneven stages included: a
    # rank sees only its own stages, so the layer-to-stage map is one
    # all-gather over the pipeline group; the schedule owns stage-to-rank.
    layer_to_stage = gather_layer_to_stage(stages, stages[0].group)
    layout = infer_block_layout_tables_from_stages(
        stages,
        stage_to_rank=dict(stages[0].stage_index_to_group_rank),
        num_blocks=num_blocks,
        n_layers=n_layers,
        layers_per_block=layers_per_block,
        layer_to_stage=layer_to_stage,
        cache=attn_res_cache,
    )
    store = RankStore()
    for stage in stages:
        stage.set_routing(layout, store)
    logger.info(
        "Kimi K3 pipeline: %d stage(s) on this rank %s, block transport %s",
        len(stages),
        [s.stage_index for s in stages],
        "delta with rank store" if attn_res_cache else "whole stack every hop",
    )
    return pp_schedule, model_parts, has_first_stage, has_last_stage
