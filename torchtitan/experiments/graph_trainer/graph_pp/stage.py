# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Stage containers and graph execution contracts for ``GraphRuntime``."""

import dataclasses
from collections.abc import Callable
from typing import Any, Protocol

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    _PipelineContext,
    _PipelineScheduleRuntime,
)
from torch.distributed.pipelining.stage import PipelineStage


class StageGraphs(Protocol):
    """Common contract shared by all stage graph executors.

    Calling convention:
        Parameter-gradient lists remain in flat graph-output order until
        ``param_grads_for_accumulation`` converts them to live parameter order.
    """

    @property
    def accumulates_gradients_in_graph(self) -> bool:
        """Return whether backward writes into graph-owned accumulators."""

    def zero_grad_(self) -> list[Any]:
        """Zero graph-owned gradient accumulators.

        Calling convention:
            ``zero_grad_() -> flat_gradient_accumulators``
        """

    def unshard_params(
        self,
        flat_param_values: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        """Materialize parameter values consumed by a graph action."""

    def reduce_grads(
        self,
        unsharded_param_grads: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        """Reduce raw parameter gradients after graph execution."""

    def param_grads_for_accumulation(
        self,
        param_grads: list[Any],
    ) -> list[Any]:
        """Convert graph gradients to live parameter order.

        Calling convention:
            ``param_grads_for_accumulation(flat_graph_gradients)``
            ``-> live_parameter_gradients``
        """


class SplitStageGraphs(StageGraphs, Protocol):
    """Bound split-graph execution contract for one model stage.

    Implementations own their graph modules, metadata, and low-level graph
    executor. ``GraphRuntime`` passes explicit runtime values and stores the
    returned values in ``GraphPipelineStage.state`` and upstream PP caches.
    """

    @property
    def supports_backward_input_weight_split(self) -> bool:
        """Return whether backward input and weight graphs are separate.

        Returns:
            bool: ``True`` when ``backward_input`` and ``backward_weight``
            should run as separate schedule actions; ``False`` when
            one full-backward graph handles both.
        """

    def unshard_params(
        self,
        flat_param_values: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        """Materialize parameter values consumed by forward graphs.

        Args:
            flat_param_values (list[Any]): Flat parameter values from the
                stage module.
            runtime_validate (bool): Whether to run repeated per-microbatch
                validation before executing the graph.

        Returns:
            list[Any]: Flat unsharded parameter values expected by later
            forward calls.
        """

    def forward(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
        *,
        unsharded_param_values: list[Any],
        flat_buffer_values: list[Any],
        runtime_validate: bool = False,
    ) -> tuple[Any, tuple[Any, ...]]:
        """Run the stage forward graph.

        Args:
            args (tuple[Any, ...]): Runtime positional forward inputs for this
                microbatch.
            kwargs (dict[str, Any]): Runtime keyword forward inputs for this
                microbatch.
            target (Any): Target value for last-stage loss graphs, or ``None``
                for non-last stages.
            loss_kwargs (dict[str, Any]): Extra loss keyword arguments for
                last-stage graphs.
            unsharded_param_values (list[Any]): Flat parameter values returned by
                ``unshard_params``.
            flat_buffer_values (list[Any]): Flat buffer values from the stage
                module.
            runtime_validate (bool): Whether to run repeated per-microbatch
                validation before executing the graph.

        Returns:
            tuple[Any, tuple[Any, ...]]: ``(stage_output,
            saved_values_for_backward)``.
        """

    def full_backward(
        self,
        stage_output: tuple[Any, ...],
        saved_values_for_backward: tuple[Any, ...],
        output_grads_from_next: tuple[Any, ...],
        *,
        runtime_validate: bool = False,
    ) -> tuple[list[Any], list[Any]]:
        """Run the full backward graph for one microbatch.

        Args:
            stage_output (tuple[Any, ...]): Forward user output tuple for this
                microbatch.
            saved_values_for_backward (tuple[Any, ...]): Values returned by
                the forward graph for the backward graph.
            output_grads_from_next (tuple[Any, ...]): Output gradients received
                from the next pipeline stage. Last stages pass an empty tuple.
            runtime_validate (bool): Whether to run repeated per-microbatch
                validation before executing the graph.

        Returns:
            tuple[list[Any], list[Any]]: ``(input_grads_to_prev,
            unsharded_param_grads)``.
        """

    def backward_input(
        self,
        stage_output: tuple[Any, ...],
        saved_values_for_backward: tuple[Any, ...],
        output_grads_from_next: tuple[Any, ...],
        *,
        runtime_validate: bool = False,
    ) -> tuple[list[Any], tuple[Any, ...]]:
        """Run the input-gradient half of split backward.

        Args:
            stage_output (tuple[Any, ...]): Forward user output tuple for this
                microbatch.
            saved_values_for_backward (tuple[Any, ...]): Values returned by
                the forward graph for backward.
            output_grads_from_next (tuple[Any, ...]): Output gradients received
                from the next pipeline stage.
            runtime_validate (bool): Whether to run repeated per-microbatch
                validation before executing the graph.

        Returns:
            tuple[list[Any], tuple[Any, ...]]: ``(input_grads_to_prev,
            saved_values_for_backward_weight)``.
        """

    def backward_weight(
        self,
        saved_values_for_backward_weight: tuple[Any, ...],
    ) -> list[Any]:
        """Run the weight-gradient half of split backward.

        Args:
            saved_values_for_backward_weight (tuple[Any, ...]): Live values returned by
                ``backward_input`` for the weight-gradient graph.

        Returns:
            list[Any]: Flat unsharded parameter-gradient values.
        """

    def reduce_grads(
        self,
        unsharded_param_grads: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        """Reduce unsharded parameter gradients.

        Args:
            unsharded_param_grads (list[Any]): Flat accumulated unsharded
                parameter gradients.
            runtime_validate (bool): Whether to run repeated per-microbatch
                validation before executing the graph.

        Returns:
            list[Any]: Flat sharded parameter-gradient values.
        """


class JointStageGraphs(StageGraphs, Protocol):
    """Bound joint forward/loss/backward graph for a PP=1 stage.

    Calling convention:
        ``(args, kwargs, target, loss_kwargs)``
        ``-> (loss, parameter_gradients)``
    """

    def forward_backward(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
        *,
        unsharded_param_values: list[Any],
        flat_buffer_values: list[Any],
        runtime_validate: bool = False,
    ) -> tuple[Any, list[Any]]:
        """Run one joint graph and return its loss and parameter gradients."""


class OverlapStageGraphs(Protocol):
    """Bound graph execution contract for one ``OVERLAP_F_B`` stage pair."""

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
        """Run one multiplexed forward/backward graph pair.

        Args:
            backward_stage_output (tuple[Any, ...]): Backward stage forward
                output tuple.
            backward_saved_values_for_backward (tuple[Any, ...]): Values saved
                by the backward stage's earlier forward action.
            output_grads_from_next (tuple[Any, ...]): Output gradients received
                by the backward stage.
            forward_args (tuple[Any, ...]): Runtime positional forward inputs
                for the forward stage.
            forward_kwargs (dict[str, Any]): Runtime keyword forward inputs for
                the forward stage.
            forward_target (Any): Target value for a last-stage forward graph,
                or ``None``.
            forward_loss_kwargs (dict[str, Any]): Extra loss keyword arguments
                for a last-stage forward graph.
            forward_unsharded_param_values (list[Any]): Flat unsharded params
                for the forward stage.
            forward_flat_buffer_values (list[Any]): Flat buffers for the
                forward stage.
            runtime_validate (bool): Whether to run repeated per-microbatch
                validation before executing the multiplexed graph.

        Returns:
            tuple[list[Any], list[Any], Any, tuple[Any, ...]]:
            ``(input_grads_to_prev, unsharded_param_grads, forward_output,
            forward_saved_values_for_backward)``.
        """


class StageGraphsProvider(Protocol):
    """Build or attach stage graphs before graph runtime execution."""

    def prepare_graphs(
        self,
        schedule: _PipelineScheduleRuntime,
        ctx: _PipelineContext,
        *,
        loss_kwargs: dict[str, Any],
    ) -> dict[tuple[int, int], OverlapStageGraphs]:
        """Prepare every local stage for graph runtime execution.

        Args:
            schedule (_PipelineScheduleRuntime): Runtime pipeline schedule
                being executed by the graph runtime.
            ctx (_PipelineContext): Pipeline schedule context for the current
                step.
            loss_kwargs (dict[str, Any]): Extra loss keyword arguments from the
                graph runtime.

        Returns:
            dict[tuple[int, int], OverlapStageGraphs]: Mapping from
            ``(forward_stage_index, backward_stage_index)`` to the multiplexed
            overlap graph executor for that stage pair.
        """


@dataclasses.dataclass(slots=True)
class GraphPPStageRuntimeState:
    """Mutable per-step runtime state for a ``GraphPipelineStage``.

    Attributes:
        flat_param_values (list[Any]): Flat parameter values from the stage
            module.
        flat_buffer_values (list[Any]): Flat buffer values from the stage
            module.
        unsharded_param_values (list[Any]): Flat unsharded params consumed by
            forward graphs.
        unsharded_param_grads (list[Any]): Flat unsharded gradient accumulator
            slots.
        sharded_param_grads (list[Any]): Flat reduced gradients after
            ``reduce_grads``.
        trainable_params (list[torch.Tensor]): Stage parameters that receive
            accumulated gradients.
    """

    flat_param_values: list[Any] = dataclasses.field(default_factory=list)
    flat_buffer_values: list[Any] = dataclasses.field(default_factory=list)
    unsharded_param_values: list[Any] = dataclasses.field(default_factory=list)
    unsharded_param_grads: list[Any] = dataclasses.field(default_factory=list)
    sharded_param_grads: list[Any] = dataclasses.field(default_factory=list)
    trainable_params: list[torch.Tensor] = dataclasses.field(default_factory=list)

    def clear(self) -> None:
        """Clear all per-step runtime values."""
        self.flat_param_values = []
        self.flat_buffer_values = []
        self.unsharded_param_values = []
        self.unsharded_param_grads = []
        self.sharded_param_grads = []
        self.trainable_params = []


class GraphPipelineStage(PipelineStage):
    """Pipeline stage whose compute actions are backed by explicit graphs.

    ``GraphPipelineStage`` intentionally owns only PP runtime state and the
    bound graph executor consumed by the runtime. Graph construction policy is
    provided separately through a ``StageGraphsProvider`` so callers can
    supply non-GraphTrainer graph implementations without coupling the stage to
    GraphTrainer tracing or compilation.

    Args:
        submodule (nn.Module): Stage-local model chunk.
        stage_index (int): Global pipeline stage index.
        num_stages (int): Total number of virtual pipeline stages.
        device (torch.device): Device used by the upstream PP stage.
        input_args (Any): Optional static input metadata accepted by
            ``PipelineStage``.
        output_args (Any): Optional static output metadata accepted by
            ``PipelineStage``.
        group (torch.distributed.ProcessGroup | None): Pipeline process group.
        get_mesh (Callable | None): Optional DTensor mesh lookup callback.
    """

    def __init__(
        self,
        submodule: nn.Module,
        *,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        input_args: Any = None,
        output_args: Any = None,
        group: torch.distributed.ProcessGroup | None = None,
        get_mesh: Callable | None = None,
    ) -> None:
        super().__init__(
            submodule,
            stage_index,
            num_stages,
            device,
            input_args=input_args,
            output_args=output_args,
            group=group,
            get_mesh=get_mesh,
        )
        self.graphs: SplitStageGraphs | JointStageGraphs | None = None
        self.state = GraphPPStageRuntimeState()
        self.saved_values_for_backward_weight_cache: dict[int, tuple[Any, ...]] = {}
        self._graph_pp_grads_scaled = False

    def set_graphs(
        self,
        graphs: SplitStageGraphs | JointStageGraphs,
    ) -> None:
        """Attach a bound graph executor to this stage.

        Args:
            graphs: Split or joint stage graph executor.
        """

        self.graphs = graphs

    def scale_grads(self, grad_scale_factor: int) -> None:
        """Scale accumulated graph gradients with upstream PP semantics.

        Args:
            grad_scale_factor (int): Divisor applied in-place to tensor
                gradients.
        """

        grads = (
            self.state.sharded_param_grads
            if self.state.sharded_param_grads
            else self.state.unsharded_param_grads
        )
        if grad_scale_factor == 1:
            return
        for grad in grads:
            if isinstance(grad, torch.Tensor):
                grad.div_(grad_scale_factor)
