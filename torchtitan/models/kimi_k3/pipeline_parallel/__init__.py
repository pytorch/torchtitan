# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline parallelism for Kimi K3: core's split with the tower and the aggregation
pinned to its ends, AttnRes stages, and the block routing tables."""

import logging

from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.pipelining.stage import _PipelineStageBase, PipelineStage

from torchtitan.distributed.pipeline_parallel import (
    pipeline_with_first_last_stage_modules,
)
from torchtitan.protocols.model import BaseModel

from .layout import infer_block_layout_tables, layer_to_stage_from_split
from .stage import AttnResPipelineStage, PPRankLocalCache

__all__ = ["pipeline_kimi_k3"]

logger = logging.getLogger(__name__)


def _as_attn_res_stage(stage: _PipelineStageBase) -> AttnResPipelineStage:
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


def _require_loop_style(
    schedule: _PipelineSchedule, stage_to_rank: dict[int, int], pp: int
) -> None:
    """The rank store keeps a block for the rank's later stages, so the cached transport
    needs the loop-style assignment, stage s on rank s % pp; any other one is refused."""
    for stage in sorted(stage_to_rank):
        rank = stage_to_rank[stage]
        if rank != stage % pp:
            raise ValueError(
                f"{type(schedule).__name__} is unsupported with attn_res_cache: stage "
                f"{stage} sits on rank {rank}, not on rank {stage % pp} of the "
                "loop-style assignment the rank store assumes. Use a looped schedule "
                "such as Interleaved1F1B, or turn attn_res_cache off."
            )


def pipeline_kimi_k3(model: BaseModel, *, attn_res_cache: bool = True, **kwargs):
    """pipelining_fn for Kimi K3; with attn_res_cache a hop carries only the blocks the
    receiving rank lacks, without it the whole stack, and every rank must agree."""
    (
        pp_schedule,
        model_parts,
        has_first_stage,
        has_last_stage,
        split,
    ) = pipeline_with_first_last_stage_modules(
        model,
        first_stage_module_fqns=model.pipeline_first_stage_module_fqns,
        last_stage_module_fqns=model.pipeline_last_stage_module_fqns,
        return_split=True,
        **kwargs,
    )

    stages = _swap_in_attn_res_stages(pp_schedule)
    stage_to_rank = dict(stages[0].stage_index_to_group_rank)
    if attn_res_cache:
        _require_loop_style(pp_schedule, stage_to_rank, kwargs["parallel_dims"].pp)
    model_config = kwargs["model_config"]
    layer_cfgs = model_config.layers
    n_layers = len(layer_cfgs)
    layers_per_block = layer_cfgs[0].attn_res_block_size
    layer_to_stage = layer_to_stage_from_split(split)
    layout = infer_block_layout_tables(
        stage_to_rank=stage_to_rank,
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
