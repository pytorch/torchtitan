# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GraphTrainer stage-graph provider and resolved execution plan.

Graph construction for SPMD without and with gradient accumulation lives in
``spmd_graph_builder``; PP stage graph construction lives in
``graph_pp.pp_graph_builder``. Shared helpers and the flat calling convention
live in ``graph_builder_utils``.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import Any, cast, Literal, TYPE_CHECKING

import torch
from torch.distributed.pipelining import PipelineStageInfo
from torch.distributed.pipelining.schedules import (
    _PipelineContext,
    _PipelineScheduleRuntime,
)

from torchtitan.distributed import ParallelismContext
from torchtitan.experiments.graph_trainer.common_utils import (
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.graph_builder_utils import (
    GraphTrainerConfigView,
)
from torchtitan.experiments.graph_trainer.graph_pp import stage_builder
from torchtitan.experiments.graph_trainer.graph_pp.pp_graph_builder import (
    _build_graph_pp_overlap_graphs,
    _build_stage_graphs,
    _compile_stage_graphs,
)
from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    _GraphComputationType,
    FORWARD_BACKWARD,
    FORWARD_BACKWARD_FIRST_WITH_UNSHARD,
    FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD,
    FORWARD_BACKWARD_NOGRADACCUM,
    FULL_FORWARD_BACKWARD,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import (
    GraphPipelineStage,
    OverlapStageGraphs,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    normalize_graph_pp_microbatch_inputs,
)
from torchtitan.experiments.graph_trainer.spmd_gradient_accumulation_graph_builder import (
    _build_gradient_accumulation_fwd_bwd_graphs,
)
from torchtitan.experiments.graph_trainer.spmd_graph_builder import (
    _build_fwd_bwd_graphs,
)


if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
    from torchtitan.models.common.dist_moe.runtime import _DistMoeForwardContext


UnshardPlacement = Literal["first_microbatch", "schedule", "every_microbatch"]


ReduceGradPlacement = Literal["last_microbatch", "schedule", "every_microbatch"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class GraphExecutionPlan:
    """Resolved FSDP placement, gradient accumulation, and action variants.

    ``unshard`` and ``reduce_grad`` are ``None`` without FSDP. With FSDP:

    - SPMD without gradient accumulation keeps both boundaries in its single
      joint graph.
    - SPMD with gradient accumulation either keeps both in every microbatch
      or moves them into the first and last microbatches.
    - PP places both boundaries in the schedule.

    ``reuse_unsharded_parameters`` and the computation-type helpers describe
    the SPMD schedules only.
    """

    pp_enabled: bool
    num_microbatches: int
    unshard: UnshardPlacement | None
    reduce_grad: ReduceGradPlacement | None
    fuse_wgrad_accumulation: bool

    @property
    def has_gradient_accumulation(self) -> bool:
        return self.num_microbatches > 1

    @property
    def extract_fsdp_param_unshard(self) -> bool:
        return self.unshard == "schedule"

    @property
    def extract_fsdp_grad_reduction(self) -> bool:
        return self.reduce_grad == "schedule"

    @property
    def unshard_in_first_microbatch(self) -> bool:
        return self.unshard == "first_microbatch"

    @property
    def reduce_grad_in_last_microbatch(self) -> bool:
        return self.reduce_grad == "last_microbatch"

    @property
    def split_fsdp_param_unshard(self) -> bool:
        return self.extract_fsdp_param_unshard or self.unshard_in_first_microbatch

    @property
    def split_fsdp_grad_reduction(self) -> bool:
        return self.extract_fsdp_grad_reduction or self.reduce_grad_in_last_microbatch

    @property
    def reuse_unsharded_parameters(self) -> bool:
        """Whether SPMD with gradient accumulation keeps parameters unsharded
        across all microbatches."""
        return not self.pp_enabled and self.split_fsdp_param_unshard

    @property
    def repeated_computation_type(self) -> _GraphComputationType:
        return (
            FORWARD_BACKWARD
            if self.unshard_in_first_microbatch
            else FULL_FORWARD_BACKWARD
        )

    def computation_type_for_microbatch(self, index: int) -> _GraphComputationType:
        if index == 0 and self.unshard_in_first_microbatch:
            return FORWARD_BACKWARD_FIRST_WITH_UNSHARD
        if index == 0 and self.has_gradient_accumulation:
            return FORWARD_BACKWARD_NOGRADACCUM
        if index == self.num_microbatches - 1 and self.reduce_grad_in_last_microbatch:
            return FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD
        return self.repeated_computation_type


def _trace_kwargs_from_context(ctx: _PipelineContext) -> dict[str, Any]:
    if ctx.kwarg_mbs is None:
        return {}
    return ctx.kwarg_mbs[0]


def _resolve_dist_moe_activation_slot(
    forward_context: _DistMoeForwardContext | None,
    *,
    stage_index: int,
    microbatch_index: int = 0,
) -> torch.Tensor | None:
    """Resolve the representative Dist-MoE slot used to trace one stage."""
    if forward_context is None:
        return None
    return forward_context.resolve_activation_slot(
        PipelineStageInfo(
            stage_index=stage_index,
            microbatch_index=microbatch_index,
        )
    )


@dataclasses.dataclass(slots=True)
class GraphTrainerStageGraphProvider:
    """Build bound GraphPP stage graphs with GraphTrainer tracing and passes.

    Args:
        loss_fn: Loss function used to trace last-stage loss and backward.
        config: Full Trainer configuration for SPMD, or its compile,
            parallelism, and model fields for PP.
        plan: Resolved FSDP placement and gradient accumulation choices.
    """

    loss_fn: Callable
    config: "GraphTrainer.Config | GraphTrainerConfigView"
    plan: GraphExecutionPlan
    parallelism_context: ParallelismContext | None = None
    _warned_cuda_graph: bool = False
    # Calling convention:
    # key = (forward_stage_index, backward_stage_index); the graph is reused
    # across microbatches for that stage pair.
    _overlap_graphs: dict[tuple[int, int], OverlapStageGraphs] | None = None

    def _warn_if_cuda_graph_pass_requested(self) -> None:
        if self._warned_cuda_graph:
            return
        if not self.config.compile.enable_passes:
            return
        if "cuda_graph_pass" in self.config.compile.disable_passes:
            return
        warnings.warn(
            "GraphPP compiles extracted stage graphs with use_cuda_graph=False "
            "even though cuda_graph_pass is enabled. CUDA graph capture needs "
            "a separate GraphPP runtime integration. Pass "
            "Add 'cuda_graph_pass' to compile.disable_passes to silence this warning.",
            stacklevel=3,
        )
        self._warned_cuda_graph = True

    def prepare_graphs(
        self,
        schedule: _PipelineScheduleRuntime,
        ctx: _PipelineContext,
        *,
        loss_kwargs: dict[str, Any],
        dist_moe_forward_context: _DistMoeForwardContext | None = None,
    ) -> dict[tuple[int, int], OverlapStageGraphs]:
        """Build one graph per stage with its optional Dist-MoE slot input."""
        graph_stages = [cast(GraphPipelineStage, stage) for stage in schedule._stages]
        maybe_register_blockmask_pytree_node()
        trace_ctx = ctx
        if ctx.arg_mbs is not None or ctx.kwarg_mbs is not None:
            # Upstream PP creates a fresh _PipelineContext for each action, but
            # all contexts share the same arg_mbs/kwarg_mbs lists. Mutate lists
            # that upstream provided so later actions see graphable BlockMask objects.
            # Use distinct objects for tracing so make_fx does not consume the
            # same closure tensor objects that runtime replay will receive.
            num_microbatches = (
                len(ctx.arg_mbs) if ctx.arg_mbs is not None else len(ctx.kwarg_mbs)
            )
            arg_mbs = (
                ctx.arg_mbs
                if ctx.arg_mbs is not None
                else [() for _ in range(num_microbatches)]
            )
            kwarg_mbs = (
                ctx.kwarg_mbs
                if ctx.kwarg_mbs is not None
                else [{} for _ in range(num_microbatches)]
            )
            trace_arg_mbs, trace_kwarg_mbs = normalize_graph_pp_microbatch_inputs(
                arg_mbs,
                kwarg_mbs,
            )
            runtime_arg_mbs, runtime_kwarg_mbs = normalize_graph_pp_microbatch_inputs(
                arg_mbs,
                kwarg_mbs,
            )
            if ctx.arg_mbs is not None:
                ctx.arg_mbs[:] = runtime_arg_mbs
            if ctx.kwarg_mbs is not None:
                ctx.kwarg_mbs[:] = runtime_kwarg_mbs
            trace_ctx = _PipelineContext(
                schedule,
                trace_arg_mbs,
                trace_kwarg_mbs,
                ctx.target_mbs,
                ctx.losses,
            )
        if not self.plan.pp_enabled:
            if len(graph_stages) != 1 or self.parallelism_context is None:
                raise ValueError(
                    "Joint forward/backward requires one stage and parallel dims"
                )
            stage = graph_stages[0]
            if stage.graphs is None:
                trace_inputs = (
                    stage,
                    stage_builder._trace_args_for_stage(stage, trace_ctx),
                    _trace_kwargs_from_context(trace_ctx),
                    stage_builder._trace_target_from_context(stage, trace_ctx),
                    loss_kwargs,
                )
                trainer_config = cast("GraphTrainer.Config", self.config)
                if self.plan.has_gradient_accumulation:
                    _build_gradient_accumulation_fwd_bwd_graphs(
                        *trace_inputs,
                        loss_fn=self.loss_fn,
                        trainer_config=trainer_config,
                        parallelism_context=self.parallelism_context,
                        plan=self.plan,
                    )
                else:
                    _build_fwd_bwd_graphs(
                        *trace_inputs,
                        loss_fn=self.loss_fn,
                        trainer_config=trainer_config,
                        parallelism_context=self.parallelism_context,
                    )
            return {}

        for stage in graph_stages:
            if stage.graphs is not None:
                continue
            _build_stage_graphs(
                stage,
                stage_builder._trace_args_for_stage(stage, trace_ctx),
                _trace_kwargs_from_context(trace_ctx),
                stage_builder._trace_target_from_context(stage, trace_ctx),
                loss_kwargs,
                loss_fn=self.loss_fn,
                config=self.config,
                parallelism_context=self.parallelism_context,
                compile_graphs=False,
                extract_fsdp_param_unshard=self.plan.extract_fsdp_param_unshard,
                extract_fsdp_grad_reduction=self.plan.extract_fsdp_grad_reduction,
                activation_slot_id_1=_resolve_dist_moe_activation_slot(
                    dist_moe_forward_context,
                    stage_index=stage.stage_index,
                ),
            )

        required_overlap_pairs = stage_builder._required_multiplex_pairs(schedule)
        if not required_overlap_pairs:
            self._overlap_graphs = {}
            overlap_graphs: dict[tuple[int, int], OverlapStageGraphs] = {}
        elif self._overlap_graphs is None:
            self._overlap_graphs = _build_graph_pp_overlap_graphs(
                schedule,
                compile_config=self.config.compile,
            )
            overlap_graphs = dict(self._overlap_graphs)
        else:
            missing_pairs = required_overlap_pairs - set(self._overlap_graphs)
            if missing_pairs:
                raise ValueError(
                    "GraphPP cached overlap graphs do not cover current "
                    f"schedule pairs: missing {sorted(missing_pairs)}."
                )
            overlap_graphs = {
                pair: self._overlap_graphs[pair] for pair in required_overlap_pairs
            }

        for stage in graph_stages:
            _compile_stage_graphs(stage, compile_config=self.config.compile)
        return overlap_graphs
