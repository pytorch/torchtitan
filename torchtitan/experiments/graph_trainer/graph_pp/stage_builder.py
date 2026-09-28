# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Pipeline-parallel graph construction helpers for GraphTrainer."""

import dataclasses
from collections.abc import Callable
from typing import Any, cast, Protocol

import torch
import torch.fx as fx
from torch.distributed.pipelining.schedules import (
    _PipelineContext,
    _PipelineScheduleRuntime,
    OVERLAP_F_B,
)

from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.graph_pp.graph_multiplex import (
    multiplex_fw_bw_graph,
)
from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    BACKWARD,
    BACKWARD_WITH_REDUCE_GRAD,
)
from torchtitan.experiments.graph_trainer.graph_pp.split_di_dw import (
    GraphPPDiDwSplit,
    split_di_dw_graph,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import (
    GraphPipelineStage,
    OverlapStageGraphs,
    SplitStageGraphs,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    overlap_fw_bw_sub_actions,
)


class _StageGraphModules(Protocol):
    fw: fx.GraphModule
    full_bw: fx.GraphModule


class _StageGraphMeta(Protocol):
    num_param_grad_values: int
    num_input_grad_values: int


class _GraphTrainerStageGraphs(SplitStageGraphs, Protocol):
    modules: _StageGraphModules
    meta: _StageGraphMeta
    compiled: bool

    def _forward_args(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
        *,
        unsharded_param_values: list[Any],
        flat_buffer_values: list[Any],
        runtime_validate: bool = False,
    ) -> list[Any]: ...

    def _backward_args(
        self,
        stage_output: tuple[Any, ...],
        saved_values_for_backward: tuple[Any, ...],
        output_grads_from_next: tuple[Any, ...],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]: ...

    def _grad_accumulator_args(self) -> list[torch.Tensor]: ...

    def _split_forward_outputs(
        self,
        fw_outputs: tuple[Any, ...],
    ) -> tuple[Any, tuple[Any, ...]]: ...

    def _split_full_backward_outputs(
        self,
        bw_outputs: tuple[Any, ...],
    ) -> tuple[list[Any], list[Any]]: ...


@dataclasses.dataclass(slots=True)
class GraphTrainerOverlapGraphs(OverlapStageGraphs):
    """GraphTrainer-backed executor for one multiplexed ``OVERLAP_F_B`` pair."""

    fw_graphs: _GraphTrainerStageGraphs
    bw_graphs: _GraphTrainerStageGraphs
    multiplexed_graph: fx.GraphModule
    execute_graph_module: Callable[[fx.GraphModule, list[Any]], tuple[Any, ...]]

    def forward_backward(
        self,
        *,
        backward_stage_output: tuple[Any, ...],
        backward_saved_values_for_backward: tuple[Any, ...],
        output_grads_from_next: tuple[Any, ...],
        forward_args: tuple[Any, ...],
        forward_kwargs: dict[str, Any],
        forward_target: Any,
        forward_loss_kwargs: dict[str, Any],
        forward_unsharded_param_values: list[Any],
        forward_flat_buffer_values: list[Any],
        runtime_validate: bool = False,
    ) -> tuple[list[Any], list[Any], Any, tuple[Any, ...]]:
        """Run one multiplexed backward/forward graph.

        Calling convention:
            ``(*backward_inputs, *backward_grad_accumulators, *forward_inputs)``
            ``-> (*param_grads, *input_grads, *user_outputs,``
            ``    *saved_for_backward, *side_effect_outputs)``
        """

        bw_args = [
            *self.bw_graphs._backward_args(
                backward_stage_output,
                backward_saved_values_for_backward,
                output_grads_from_next,
                runtime_validate=runtime_validate,
            ),
            *self.bw_graphs._grad_accumulator_args(),
        ]
        fw_args = self.fw_graphs._forward_args(
            forward_args,
            forward_kwargs,
            forward_target,
            forward_loss_kwargs,
            unsharded_param_values=forward_unsharded_param_values,
            flat_buffer_values=forward_flat_buffer_values,
            runtime_validate=runtime_validate,
        )
        multiplex_args: list[Any] = []
        multiplex_args.extend(bw_args)
        multiplex_args.extend(fw_args)
        multiplexed_outputs = self.execute_graph_module(
            self.multiplexed_graph,
            multiplex_args,
        )
        num_bw_outputs = (
            self.bw_graphs.meta.num_param_grad_values
            + self.bw_graphs.meta.num_input_grad_values
        )
        input_grads, param_grads = self.bw_graphs._split_full_backward_outputs(
            multiplexed_outputs[:num_bw_outputs]
        )
        output, saved_values_for_backward = self.fw_graphs._split_forward_outputs(
            multiplexed_outputs[num_bw_outputs:]
        )
        return input_grads, param_grads, output, saved_values_for_backward


def _split_stage_backward_graph(
    bw_module: fx.GraphModule,
    *,
    num_param_grads: int,
    num_input_grads: int,
) -> GraphPPDiDwSplit | None:
    """Split and validate the optional PP backward-input/weight graphs."""
    didw_split = split_di_dw_graph(
        bw_module,
        num_param_grads=num_param_grads,
    )
    if didw_split is not None and didw_split.num_input_grads != num_input_grads:
        raise ValueError(
            "GraphPP dI/dW split changed the raw input-gradient count: "
            f"expected {num_input_grads}, got {didw_split.num_input_grads}"
        )
    return didw_split


def _required_multiplex_pairs(
    schedule: _PipelineScheduleRuntime,
) -> set[tuple[int, int]]:
    try:
        pipeline_order = schedule.pipeline_order_with_comms
    except AttributeError as exc:
        raise ValueError(
            "GraphPP overlap graph construction requires a runtime PP schedule "
            "with pipeline_order_with_comms."
        ) from exc
    required_pairs: set[tuple[int, int]] = set()
    for action in pipeline_order.get(schedule.rank, []):
        if action.computation_type != OVERLAP_F_B:
            continue
        fw_action, bw_action = overlap_fw_bw_sub_actions(
            action,
            backward_computation_types=(
                BACKWARD,
                BACKWARD_WITH_REDUCE_GRAD,
            ),
        )
        required_pairs.add((fw_action.stage_index, bw_action.stage_index))
    return required_pairs


def _build_graph_pp_overlap_graphs(
    schedule: _PipelineScheduleRuntime,
    *,
    compile_config: GraphTrainerCompileConfig,
    annotate_graph: Callable[..., None],
    compile_graph_module: Callable[..., fx.GraphModule],
    execute_graph_module: Callable[[fx.GraphModule, list[Any]], tuple[Any, ...]],
) -> dict[tuple[int, int], OverlapStageGraphs]:
    """Build multiplexed graphs required by ``OVERLAP_F_B`` schedule actions."""

    stage_index_to_stage = {
        stage.stage_index: cast(GraphPipelineStage, stage) for stage in schedule._stages
    }
    overlap_graphs: dict[tuple[int, int], OverlapStageGraphs] = {}
    for fw_stage_idx, bw_stage_idx in _required_multiplex_pairs(schedule):
        pair = (fw_stage_idx, bw_stage_idx)
        fw_stage = stage_index_to_stage[fw_stage_idx]
        bw_stage = stage_index_to_stage[bw_stage_idx]
        if fw_stage.graphs is None or bw_stage.graphs is None:
            raise ValueError(
                "GraphPP overlap graph construction requires both stage "
                f"graphs first: forward_stage={fw_stage_idx}, "
                f"backward_stage={bw_stage_idx}."
            )
        fw_graphs = cast(_GraphTrainerStageGraphs, fw_stage.graphs)
        bw_graphs = cast(_GraphTrainerStageGraphs, bw_stage.graphs)
        if fw_graphs.compiled or bw_graphs.compiled:
            raise ValueError(
                "GraphPP overlap graphs must be built before stage graphs are compiled."
            )
        multiplexed_graph = multiplex_fw_bw_graph(
            fw_graphs.modules.fw,
            bw_graphs.modules.full_bw,
        )
        annotate_graph(
            multiplexed_graph,
            stage_index=fw_stage_idx,
            callable_name="multiplex",
            action_name="OVERLAP_F_B",
        )
        compiled_graph = compile_graph_module(
            multiplexed_graph,
            compile_config=compile_config,
            graph_name=f"stage_{fw_stage_idx}_fw_stage_{bw_stage_idx}_bw_multiplex",
        )
        overlap_graphs[pair] = GraphTrainerOverlapGraphs(
            fw_graphs=fw_graphs,
            bw_graphs=bw_graphs,
            multiplexed_graph=compiled_graph,
            execute_graph_module=execute_graph_module,
        )
    return overlap_graphs


def _flat_output_grads_from_stage_metadata(
    stage: GraphPipelineStage,
) -> tuple[Any, ...]:
    """Return flat non-last output grads supplied by upstream PP metadata."""

    stage_meta = stage._stage_meta
    output_grad_metas = stage_meta.output_grads
    if output_grad_metas is None:
        raise ValueError(
            "GraphPP requires upstream PP backward metadata before tracing a "
            f"non-last stage. Missing output_grads for stage {stage.stage_index}."
        )
    output_metas = stage_meta.outputs
    if output_metas is not None and len(output_grad_metas) != len(output_metas):
        raise ValueError(
            "GraphPP output grad metadata does not match stage output structure: "
            f"{len(output_grad_metas)} metadata entries for "
            f"{len(output_metas)} output leaves"
        )
    return tuple(
        None if meta is None else stage._to_tensor(meta) for meta in output_grad_metas
    )


def _example_args_from_stage_metadata(stage: GraphPipelineStage) -> tuple[Any, ...]:
    if stage._stage_meta.inputs is None:
        raise ValueError(
            "GraphPP stage metadata was not initialized before graph construction: "
            f"stage {stage.stage_index}"
        )
    return tuple(stage._to_tensor(meta) for meta in stage._stage_meta.inputs)


def _trace_args_for_stage(
    stage: GraphPipelineStage,
    ctx: _PipelineContext,
) -> tuple[Any, ...]:
    """Return representative positional inputs for a stage trace.

    Match upstream PP metadata/runtime semantics: first-stage positional inputs
    are true user microbatch args, while non-first positional inputs are
    received activations represented by PP stage metadata. ``minimal_fx_tracer``
    fakeifies tensor inputs before executing the traced function.
    """
    if stage.is_first:
        if ctx.arg_mbs is None:
            return ()
        return tuple(ctx.arg_mbs[0])
    return _example_args_from_stage_metadata(stage)


def _trace_target_from_context(
    stage: GraphPipelineStage,
    ctx: _PipelineContext,
) -> Any:
    if not stage.is_last or ctx.target_mbs is None:
        return None
    return ctx.target_mbs[0]
