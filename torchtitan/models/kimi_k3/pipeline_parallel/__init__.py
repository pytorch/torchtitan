# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline parallelism for Kimi K3: core's split with the tower and the aggregation
pinned to its ends, AttnRes stages, and the block routing tables."""

import copy
import logging

import torch
import torch.distributed as dist
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.pipelining.stage import _PipelineStageBase, PipelineStage

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.pipeline_parallel import (
    get_module_fqns_per_model_part,
    pipeline_llm,
)
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from torchtitan.protocols.model import BaseModel

from .cache import PPRankLocalCache
from .layout import infer_block_layout_tables, layer_to_stage_from_split
from .stage import AttnResPipelineStage
from .vision_dep import install_vision_dep, VisionDepPipelineStage

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


def _vision_replica(
    model: BaseModel, *, parallelism_context, training, ac_config, dump_folder, device
):
    tower = getattr(model, "vision_encoder", None)
    if tower is None:
        raise ValueError("vision_dep needs a model with a vision encoder.")
    if training.enable_cpu_offload:
        raise ValueError("vision_dep does not support training.enable_cpu_offload.")
    replica = copy.deepcopy(tower)
    with parallelism_context.activate_spmd():
        annotate_replicated_parameters(replica, parallelism_context)
        replica._parallelize(parallelism_context)
        if ac_config is not None:
            ac_config.build(dump_folder=dump_folder).apply(replica)
    replica.to_empty(device=device)
    # The first step overwrites these weights; drawing them must not move the model's seed.
    forked = [device] if torch.device(device).type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=forked):
        replica.init_states()
    with torch.no_grad():
        dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
        for param in replica.parameters():
            param.data = param.data.to(dtype)
    return replica.train()


def _pp_groups(parallelism_context) -> list[list[int]]:
    pp_mesh = parallelism_context.get_mesh("pp")
    if dist.get_backend(pp_mesh.get_group()) == "fake":
        raise ValueError("vision_dep needs a real pipeline process group.")
    pp = parallelism_context.pp
    stride = dist.get_world_size() // pp
    groups = [[base + k * stride for k in range(pp)] for base in range(stride)]
    if pp_mesh.mesh.tolist() not in groups:
        raise RuntimeError(
            f"The pipeline group {pp_mesh.mesh.tolist()} is not a pp slice of the "
            "world mesh."
        )
    return groups


def pipeline_kimi_k3(model: BaseModel, *, attn_res_cache: bool = True, **kwargs):
    """pipelining_fn for Kimi K3; with attn_res_cache a hop carries only the blocks the
    receiving rank lacks, without it the whole stack, and every rank must agree."""
    model_config = kwargs["model_config"]
    dep = model_config.vision_dep
    replica = None
    if dep.enabled:
        replica = _vision_replica(
            model,
            parallelism_context=kwargs["parallelism_context"],
            training=kwargs["training"],
            ac_config=kwargs["ac_config"],
            dump_folder=kwargs["dump_folder"],
            device=kwargs["device"],
        )
    parallelism = kwargs["parallelism"]
    split = parallelism.pipeline_parallel_module_fqns_per_model_part
    if split is None:
        split = get_module_fqns_per_model_part(
            model,
            first_stage_module_fqns=model.pipeline_first_stage_module_fqns,
            last_stage_module_fqns=model.pipeline_last_stage_module_fqns,
            parallelism_context=kwargs["parallelism_context"],
            parallelism=parallelism,
            model_config=model_config,
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
        _require_loop_style(
            pp_schedule, stage_to_rank, kwargs["parallelism_context"].pp
        )
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
    if replica is not None:
        parallelism_context = kwargs["parallelism_context"]
        dp_mesh = parallelism_context.get_optional_mesh("dp")
        tp_mesh = parallelism_context.get_optional_mesh("tp")
        pp_schedule = install_vision_dep(
            pp_schedule,
            stages,
            replica=replica,
            pp_groups=_pp_groups(parallelism_context),
            dp_group=None if dp_mesh is None else dp_mesh.get_group(),
            tp_group=None if tp_mesh is None else tp_mesh.get_group(),
            hidden_dim=model_config.dim,
            compute_dtype=TORCH_DTYPE_MAP[kwargs["training"].mixed_precision_param],
            bubble=dep.bubble,
            cost_ratio=dep.bubble_cost_ratio,
        )
    logger.info(
        "Kimi K3 pipeline: %d stage(s) on this rank %s, block transport %s%s",
        len(stages),
        [s.stage_index for s in stages],
        "delta with rank store" if attn_res_cache else "whole stack every hop",
        ", vision encodes decoupled" if replica is not None else "",
    )
    return pp_schedule, model_parts, has_first_stage, has_last_stage
