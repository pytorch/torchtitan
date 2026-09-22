# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline parallelism for Kimi K3: core's split with the tower and the aggregation
pinned to its ends, AttnRes stages, and the block routing tables."""

import copy
import logging

from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.pipelining.stage import _PipelineStageBase, PipelineStage

from torchtitan.distributed.pipeline_parallel import (
    get_module_fqns_per_model_part,
    pipeline_llm,
)
from torchtitan.protocols.model import BaseModel

from .cache import PPRankLocalCache
from .layout import infer_block_layout_tables, layer_to_stage_from_split
from .stage import AttnResPipelineStage
from .vision_dep import (
    install_vision_dep,
    VisionDepPipelineStage,
    vit_dep_split,
    VIT_DEP_STAGE_FQNS,
)

__all__ = ["pipeline_kimi_k3"]

logger = logging.getLogger(__name__)


def _as_attn_res_stage(
    stage: _PipelineStageBase, stage_class: type[AttnResPipelineStage]
) -> AttnResPipelineStage:
    assert isinstance(stage, PipelineStage)
    rebuilt = stage_class(
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
    stage_class: type[AttnResPipelineStage] = AttnResPipelineStage,
) -> list[AttnResPipelineStage]:
    if isinstance(schedule, PipelineScheduleSingle):
        rebuilt = _as_attn_res_stage(schedule._stage, stage_class)
        schedule._stage = rebuilt
        return [rebuilt]
    if isinstance(schedule, PipelineScheduleMulti):
        rebuilt_stages = [_as_attn_res_stage(s, stage_class) for s in schedule._stages]
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
    receiving rank lacks, without it the whole stack, and every rank must agree.

    The model config's vision_dep gives the vision tower and the embedding the first
    stage alone, with the encodes run ahead of the forward that reads them or placed
    in the idle intervals of the schedule's own action order.
    """
    dep = kwargs["model_config"].vision_dep
    _check_vision_dep(dep)
    if dep.enabled:
        kwargs["parallelism"] = _with_vit_dep_split(model, kwargs)
    parallelism = kwargs["parallelism"]
    split = parallelism.pipeline_parallel_module_fqns_per_model_part
    if split is None:
        split = get_module_fqns_per_model_part(
            model,
            first_stage_module_fqns=model.pipeline_first_stage_module_fqns,
            last_stage_module_fqns=model.pipeline_last_stage_module_fqns,
            parallel_dims=kwargs["parallel_dims"],
            parallelism=parallelism,
            model_config=kwargs["model_config"],
        )
        derived = copy.copy(parallelism)
        derived.pipeline_parallel_module_fqns_per_model_part = split
        kwargs["parallelism"] = derived
    pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_llm(
        model, **kwargs
    )

    stages = _swap_in_attn_res_stages(
        pp_schedule, VisionDepPipelineStage if dep.enabled else AttnResPipelineStage
    )
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
    if dep.prefetch or dep.bubble:
        install_vision_dep(
            pp_schedule,
            stages,  # pyrefly: ignore[bad-argument-type]
            rank=kwargs["parallel_dims"].get_mesh("pp").get_local_rank(),
            pp_size=kwargs["parallel_dims"].pp,
            prefetch=dep.prefetch,
            bubble=dep.bubble,
            cost_ratio=dep.bubble_cost_ratio,
            max_pending=dep.bubble_max_pending,
        )
    logger.info(
        "Kimi K3 pipeline: %d stage(s) on this rank %s, block transport %s",
        len(stages),
        [s.stage_index for s in stages],
        "delta with rank store" if attn_res_cache else "whole stack every hop",
    )
    return pp_schedule, model_parts, has_first_stage, has_last_stage


def _check_vision_dep(dep) -> None:
    if dep.prefetch and dep.bubble:
        raise ValueError(
            "vision_dep.prefetch and vision_dep.bubble are alternatives; set one."
        )
    if (dep.prefetch or dep.bubble) and not dep.enabled:
        raise ValueError(
            "vision_dep.prefetch and vision_dep.bubble move the tower's encodes "
            "around the pipeline's actions, which needs the tower on a stage of "
            "its own: set vision_dep.enabled."
        )
    if dep.prefetch < 0:
        raise ValueError(
            f"vision_dep.prefetch must not be negative, got {dep.prefetch}."
        )


def _with_vit_dep_split(model: BaseModel, kwargs: dict):
    """The parallelism config with the vit_dep split, unless one is already spelled out."""
    parallelism = kwargs["parallelism"]
    configured = parallelism.pipeline_parallel_module_fqns_per_model_part
    if configured is None:
        derived = copy.copy(parallelism)
        derived.pipeline_parallel_module_fqns_per_model_part = vit_dep_split(
            model,
            parallel_dims=kwargs["parallel_dims"],
            parallelism=parallelism,
            model_config=kwargs["model_config"],
        )
        return derived
    if set(configured[0]) != set(VIT_DEP_STAGE_FQNS):
        raise ValueError(
            "vit_dep puts the vision tower and the embedding on the first stage "
            f"alone; the configured split starts with {configured[0]}."
        )
    return parallelism
