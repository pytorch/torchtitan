# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline parallelism for Kimi K3.

The split puts the block attention residual's final aggregation with the head
and the vision tower with the embedding; the stages are
:class:`AttnResPipelineStage`, whose hops carry the block stack's delta (see
``pipeline_stage.py``) once the routing tables are built from the split the
trainer actually applied.
"""

import math
from typing import cast

import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
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
