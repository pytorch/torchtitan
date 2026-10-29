# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline parallelism for Kimi K3: core's split with the tower and the aggregation
pinned to its ends, AttnRes stages, and the block routing tables.

Tensor suffixes: T is tokens, N is AttnRes blocks, D is hidden size, and V is
vocabulary size.
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
from torch.distributed.pipelining.stage import _PipelineStageBase

from torchtitan.components.loss import ChunkedLossWrapper, LossFunction
from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed.pipeline_parallel import (
    get_module_fqns_per_model_part,
    pipeline_llm,
    TorchTitanPipelineStage,
)
from torchtitan.models.base.model import BaseModel
from torchtitan.models.common.decoder import Decoder

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
    stage: _PipelineStageBase,
    stage_metadata: dict[str, Any] | None = None,
) -> AttnResPipelineStage:
    assert isinstance(stage, TorchTitanPipelineStage)
    rebuilt = AttnResPipelineStage(
        stage.submod,
        stage.stage_index,
        stage.num_stages,
        stage.device,
        group=stage.group,
        dw_builder=stage.dw_builder,
        get_mesh=stage._mesh_cache._get_mesh_cb,
        empty_cache_before_first_backward=stage.empty_cache_before_first_backward,
        **(stage_metadata or {}),
    )
    # The schedule wrote its stage-to-rank map onto the stage it was handed.
    rebuilt.stage_index_to_group_rank = stage.stage_index_to_group_rank
    return rebuilt


def _swap_in_attn_res_stages(
    schedule: _PipelineSchedule,
    stage_metadata_by_index: Mapping[int, dict[str, Any]] | None = None,
) -> list[AttnResPipelineStage]:
    def rebuild(stage: _PipelineStageBase) -> AttnResPipelineStage:
        metadata = None
        if stage_metadata_by_index is not None:
            assert stage.stage_index in stage_metadata_by_index
            metadata = stage_metadata_by_index[stage.stage_index]
        return _as_attn_res_stage(stage, metadata)

    if isinstance(schedule, PipelineScheduleSingle):
        rebuilt = rebuild(schedule._stage)
        schedule._stage = rebuilt
        return [rebuilt]
    if isinstance(schedule, PipelineScheduleMulti):
        rebuilt_stages = [rebuild(s) for s in schedule._stages]
        held: list[_PipelineStageBase] = list(rebuilt_stages)
        schedule._stages = held
        return rebuilt_stages
    raise RuntimeError(f"Unexpected pipeline schedule class {type(schedule).__name__}.")


def _schedule_stages(schedule: _PipelineSchedule) -> list[_PipelineStageBase]:
    if isinstance(schedule, PipelineScheduleSingle):
        return [schedule._stage]
    if isinstance(schedule, PipelineScheduleMulti):
        return list(schedule._stages)
    raise RuntimeError(f"Unexpected pipeline schedule class {type(schedule).__name__}.")


def _build_attn_res_stage_metadata(
    *,
    training: TrainingConfig,
    model_config: Decoder.Config,
    loss_fn: LossFunction,
    layout: BlockLayoutTables,
    context_parallel_degree: int = 1,
) -> dict[int, dict[str, Any]]:
    """Describe every K3 PP boundary without executing the full pipeline.

    Dynamic inference would run a serial forward and backward through all PP
    stages before step 1. That setup pass is prohibitively slow for full K3.
    """
    num_tokens, remainder = divmod(
        training.num_tokens_per_microbatch_per_dp_rank, context_parallel_degree
    )
    if remainder:
        raise ValueError(
            "Static Kimi K3 pipeline metadata requires the microbatch token "
            f"count to be divisible by CP ({context_parallel_degree})."
        )
    hidden_dim = model_config.dim
    dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]

    def hidden() -> torch.Tensor:
        hidden_TD = torch.empty(num_tokens, hidden_dim, dtype=dtype, device="meta")
        return hidden_TD.requires_grad_()

    def delta(num_blocks: int) -> torch.Tensor:
        delta_TND = torch.empty(
            num_tokens, num_blocks, hidden_dim, dtype=dtype, device="meta"
        )
        return delta_TND.requires_grad_()

    metadata_by_index: dict[int, dict[str, Any]] = {}
    for stage_index in range(layout.num_stages):
        is_first = stage_index == 0
        is_last = stage_index == layout.num_stages - 1

        if is_first:
            tokens_T = torch.empty(num_tokens, dtype=torch.int64, device="meta")
            input_args = (tokens_T,)
            input_grads = (None,)
        else:
            hidden_input_TD = hidden()
            delta_input_TND = delta(len(layout.delta_to_send(stage_index - 1)))
            input_args = (hidden_input_TD, delta_input_TND)
            input_grads = (
                hidden_input_TD.detach(),
                delta_input_TND.detach(),
            )

        if is_last:
            if isinstance(loss_fn, ChunkedLossWrapper):
                output_args = (hidden(),)
            else:
                logits_TV = torch.empty(
                    num_tokens,
                    model_config.vocab_size,
                    dtype=dtype,
                    device="meta",
                ).requires_grad_()
                output_args = (logits_TV,)
            output_grads = (None,)
        else:
            hidden_output_TD = hidden()
            delta_output_TND = delta(len(layout.delta_to_send(stage_index)))
            output_args = (hidden_output_TD, delta_output_TND)
            output_grads = (
                hidden_output_TD.detach(),
                delta_output_TND.detach(),
            )

        metadata_by_index[stage_index] = {
            "input_args": input_args,
            "output_args": output_args,
            "input_grads": input_grads,
            "output_grads": output_grads,
        }
    return metadata_by_index


def _require_loop_style(
    schedule: _PipelineSchedule, stage_to_rank: dict[int, int], pp: int
) -> None:
    """The rank store keeps a block for the rank's later stages, so the cached transport
    needs the loop-style assignment, stage s on rank s % pp; any other one is refused.
    """
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

    original_stages = _schedule_stages(pp_schedule)
    stage_to_rank = dict(original_stages[0].stage_index_to_group_rank)
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

    stage_metadata_by_index = None
    parallelism_context = kwargs["parallelism_context"]
    if parallelism_context.tp == 1:
        stage_metadata_by_index = _build_attn_res_stage_metadata(
            training=kwargs["training"],
            model_config=model_config,
            loss_fn=kwargs["loss_fn"],
            layout=layout,
            context_parallel_degree=parallelism_context.cp,
        )
    # SPMD CP boundaries carry plain tensors with CP-local token counts.
    # TODO: Describe TP-sharded K3 boundaries to avoid dynamic inference there.
    stages = _swap_in_attn_res_stages(pp_schedule, stage_metadata_by_index)
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
