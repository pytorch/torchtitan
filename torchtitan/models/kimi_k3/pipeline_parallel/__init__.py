# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline parallelism for Kimi K3: core's split with the tower and the aggregation
pinned to its ends, AttnRes stages, and the block routing tables.

Tensor suffixes: T is tokens, N is AttnRes blocks, and D is hidden size.
"""

import copy
import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.pipelining.stage import _PipelineStageBase, PipelineStage

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.distributed.pipeline_parallel import (
    build_decoder_stage_io,
    DecoderStageIO,
    get_module_fqns_per_model_part,
    pipeline_llm,
)
from torchtitan.models.base.model import BaseModel

from .cache import PPRankLocalCache
from .layout import (
    BlockLayoutTables,
    infer_block_layout_tables,
    layer_to_stage_from_split,
)
from .stage import AttnResPipelineStage

__all__ = ["pipeline_kimi_k3"]

logger = logging.getLogger(__name__)


def _as_attn_res_stage(
    stage: _PipelineStageBase, stage_metadata: dict[str, Any]
) -> AttnResPipelineStage:
    assert isinstance(stage, PipelineStage)
    rebuilt = AttnResPipelineStage(
        stage.submod,
        stage.stage_index,
        stage.num_stages,
        stage.device,
        group=stage.group,
        dw_builder=stage.dw_builder,
        get_mesh=stage._mesh_cache._get_mesh_cb,
        **stage_metadata,
    )
    # The schedule wrote its stage-to-rank map onto the stage it was handed.
    rebuilt.stage_index_to_group_rank = stage.stage_index_to_group_rank
    return rebuilt


def _schedule_stages(schedule: _PipelineSchedule) -> list[_PipelineStageBase]:
    if isinstance(schedule, PipelineScheduleSingle):
        return [schedule._stage]
    if isinstance(schedule, PipelineScheduleMulti):
        return list(schedule._stages)
    raise RuntimeError(f"Unexpected pipeline schedule class {type(schedule).__name__}.")


def _swap_in_attn_res_stages(
    schedule: _PipelineSchedule,
    stage_metadata: Mapping[int, dict[str, Any]],
) -> list[AttnResPipelineStage]:
    rebuilt = [
        _as_attn_res_stage(stage, stage_metadata[stage.stage_index])
        for stage in _schedule_stages(schedule)
    ]
    if isinstance(schedule, PipelineScheduleSingle):
        (schedule._stage,) = rebuilt
    elif isinstance(schedule, PipelineScheduleMulti):
        held: list[_PipelineStageBase] = list(rebuilt)
        schedule._stages = held
    return rebuilt


def _build_attn_res_stage_metadata(
    stage_io: DecoderStageIO, layout: BlockLayoutTables
) -> dict[int, dict[str, Any]]:
    """Static metadata for every stage: core's decoder boundaries plus the block stack.

    The block stack shares the hidden state's token sharding (CP-local, and split
    over TP under sequence parallelism), so each ``delta_TND`` reuses the token
    count of core's ``hidden_TD`` example.
    """
    hidden_TD = stage_io.hidden
    num_tokens, dim = hidden_TD.shape

    def carriers(num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
        delta_TND = torch.empty(
            num_tokens, num_blocks, dim, dtype=hidden_TD.dtype, device="meta"
        )
        return hidden_TD, delta_TND.requires_grad_()

    last = layout.num_stages - 1
    metadata: dict[int, dict[str, Any]] = {}
    for stage_index in range(layout.num_stages):
        inputs = (
            (stage_io.decoder_input,)
            if stage_index == 0
            else carriers(len(layout.delta_to_send(stage_index - 1)))
        )
        outputs = (
            (stage_io.decoder_output,)
            if stage_index == last
            else carriers(len(layout.delta_to_send(stage_index)))
        )
        metadata[stage_index] = {
            "input_args": inputs,
            "output_args": outputs,
            # A tuple containing None is explicit no-gradient metadata. Bare None
            # means unknown metadata and would re-enable dynamic inference.
            "input_grads": (
                (None,) if stage_index == 0 else tuple(t.detach() for t in inputs)
            ),
            "output_grads": (
                (None,) if stage_index == last else tuple(t.detach() for t in outputs)
            ),
        }
    return metadata


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
    parallelism = kwargs["parallelism"]
    split = parallelism.pipeline_parallel_module_fqns_per_model_part
    if split is None:
        split = get_module_fqns_per_model_part(
            model,
            first_stage_module_fqns=model.pipeline_first_stage_module_fqns,
            last_stage_module_fqns=model.pipeline_last_stage_module_fqns,
            parallelism_context=kwargs["parallelism_context"],
            parallelism=parallelism,
            model_config=kwargs["model_config"],
        )
        derived = copy.copy(parallelism)
        derived.pipeline_parallel_module_fqns_per_model_part = split
        kwargs["parallelism"] = derived
    pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_llm(
        model, **kwargs
    )

    stage_to_rank = dict(_schedule_stages(pp_schedule)[0].stage_index_to_group_rank)
    if attn_res_cache:
        _require_loop_style(
            pp_schedule, stage_to_rank, kwargs["parallelism_context"].pp
        )
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

    # Core infers stage metadata on real process groups by running one serial
    # forward and backward through every stage before step 1. At full K3 scale
    # that pass hit the NCCL timeout, so K3 always declares its boundaries.
    parallelism_context = kwargs["parallelism_context"]
    if (
        parallelism_context.tp_enabled
        and kwargs["parallelism"].enable_sequence_parallel
    ):
        if not any(fqn.startswith("layers.") for fqn in split[0]):
            # The multimodal embedding output is TP-replicated; layer 0 shards it.
            raise ValueError(
                "Kimi K3 pipeline parallelism with sequence parallelism needs a "
                "decoder layer on the first stage, so every boundary carries "
                "sequence-sharded tokens."
            )
    stage_io = build_decoder_stage_io(
        parallelism_context=parallelism_context,
        parallelism=kwargs["parallelism"],
        training=kwargs["training"],
        precision=kwargs["precision"],
        model_config=model_config,
        lm_head_in_loss=isinstance(kwargs["loss_fn"], ChunkedLossWrapper),
    )
    stages = _swap_in_attn_res_stages(
        pp_schedule, _build_attn_res_stage_metadata(stage_io, layout)
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
