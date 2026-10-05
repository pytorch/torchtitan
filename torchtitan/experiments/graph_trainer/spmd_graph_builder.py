# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""SPMD joint forward/backward graph construction and executor.

Traces the joint forward/loss/backward graph shared by both SPMD paths and
builds the executor for SPMD without gradient accumulation (one joint graph).
SPMD with gradient accumulation builds on this module in
``spmd_gradient_accumulation_graph_builder``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.distributed import ParallelismContext
from torchtitan.experiments.graph_trainer.common_utils import (
    compute_annotated_loss,
    compute_parameter_gradients,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.graph_builder_utils import (
    _graphtrainer_cudagraphs_enabled,
    _remove_cuda_graph_pass,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import (
    GraphPipelineStage,
    JointStageGraphs,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    compile_time_passes,
    construct_default_graph_passes,
    construct_mandatory_graph_passes,
)
from torchtitan.experiments.graph_trainer.precompile import (
    _FX_TRACE_ARTIFACT_KEY,
    compute_config_fingerprint,
    flatten_runtime_inputs,
    get_spmd_precompile_meshes,
    precompile_fx_trace_load,
)
from torchtitan.experiments.graph_trainer.registry import PASS_PIPELINE_REGISTRY
from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter


if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


def make_fwd_bwd_step(model, loss_fn):
    """Return a function that computes loss and explicit parameter gradients.

    Calling convention:
        ``(inputs, labels, global_loss_token_counts, extra_kwargs)``
        ``-> (loss, *parameter_gradients)``

    ``model`` and ``loss_fn`` are captured in the closure so neither shows up
    as a graph input. Pass ``model`` through ``minimal_fx_tracer(fn, module=model)``
    to thread its parameters/buffers as static graph inputs.
    """

    def fwd_bwd_step(inputs, labels, global_loss_token_counts, extra_kwargs):
        pred = model(inputs, **extra_kwargs)
        # The loss function is not a submodule of the model, so
        # annotate_module_fqns won't tag it. Annotate it here so that
        # downstream passes (bucketing, SAC, kernel annotations) can
        # attribute loss nodes in the traced graph.
        loss = compute_annotated_loss(
            loss_fn,
            pred,
            labels,
            {"global_loss_token_counts": global_loss_token_counts},
        )
        named_params = [
            (name, parameter)
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        ]
        grads = compute_parameter_gradients(loss, named_params)
        return [loss, *grads]

    return fwd_bwd_step


@dataclasses.dataclass(slots=True)
class GraphTrainerJointStageGraphs(JointStageGraphs):
    """Execute SPMD without gradient accumulation as one joint graph."""

    traced: TracedResult
    module: nn.Module
    num_param_grads: int
    runtime_meshes: list[DeviceMesh] | None = None
    _run: Callable[..., Any] = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._run = run_traced(
            self.traced,
            module=self.module,
            precompile_meshes=self.runtime_meshes,
        )

    def unshard_params(
        self,
        sharded_param_values: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        return list(sharded_param_values)

    def reduce_grads(
        self,
        unsharded_param_grads: list[Any],
        *,
        runtime_validate: bool = False,
    ) -> list[Any]:
        return list(unsharded_param_grads)

    def _model_input(self, args: tuple[Any, ...]) -> Any:
        if len(args) != 1:
            raise ValueError(
                "SPMD joint forward/backward expects one model input, got "
                f"{len(args)}"
            )
        return args[0]

    def forward_backward(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
        *,
        unsharded_param_values: list[Any],
        buffer_values: list[Any],
        grad_accumulators: list[Any] | None = None,
        runtime_validate: bool = False,
    ) -> tuple[Any, list[Any]]:
        global_loss_token_counts = loss_kwargs["global_loss_token_counts"]
        outputs = self._run(
            self._model_input(args),
            target,
            global_loss_token_counts,
            kwargs,
        )
        if len(outputs) != self.num_param_grads + 1:
            raise ValueError(
                "SPMD joint forward/backward output count mismatch: "
                f"expected {self.num_param_grads + 1}, got {len(outputs)}"
            )
        # Calling convention:
        # (loss, *parameter_gradients) -> (loss, list(parameter_gradients))
        return outputs[0], list(outputs[1:])

    def param_grads_for_accumulation(
        self,
        param_grads: list[Any],
    ) -> list[Any]:
        return param_grads


def construct_joint_train_step_passes(
    traced: TracedResult,
    trainer_config: "GraphTrainer.Config",
    *,
    parallelism_context: ParallelismContext,
    outer_cudagraphs_enabled: bool,
) -> list[Callable]:
    """Construct SPMD without gradient accumulation passes from the full
    config."""
    graphtrainer_cudagraphs_enabled = _graphtrainer_cudagraphs_enabled(
        trainer_config.compile,
        outer_cudagraphs_enabled=outer_cudagraphs_enabled,
    )
    if trainer_config.compile.precompile_artifact_dir:
        if graphtrainer_cudagraphs_enabled:
            return construct_default_graph_passes(
                traced,
                trainer_config,
                parallelism_context=parallelism_context,
            )
        return []
    if not trainer_config.compile.enable_passes:
        return construct_mandatory_graph_passes()

    pipeline_fn = PASS_PIPELINE_REGISTRY.get(trainer_config.compile.pass_pipeline)
    if pipeline_fn is not None:
        passes = pipeline_fn(
            traced, trainer_config, parallelism_context=parallelism_context
        )
        if outer_cudagraphs_enabled:
            passes = _remove_cuda_graph_pass(passes)
        return passes

    if graphtrainer_cudagraphs_enabled:
        return construct_default_graph_passes(
            traced,
            trainer_config,
            parallelism_context=parallelism_context,
        )

    return compile_time_passes(
        traced,
        trainer_config,
        parallelism_context=parallelism_context,
    )


def _trace_joint_stage_graph(
    stage: GraphPipelineStage,
    runtime_args: tuple[Any, Any, Any, dict[str, Any]],
    *,
    loss_fn: Callable,
    compile_config: GraphTrainerCompileConfig,
    parallelism_context: ParallelismContext,
) -> tuple[TracedResult, list[DeviceMesh] | None]:
    """Trace or load the joint forward/backward graph for either SPMD path."""
    runtime_meshes: list[DeviceMesh] | None = None
    if compile_config.precompile_artifact_dir:
        storage: DiskStorageAdapter = DiskStorageAdapter(
            compile_config.precompile_artifact_dir
        )
        if not storage.exists(_FX_TRACE_ARTIFACT_KEY):
            raise ValueError(
                "Precompiled fx_trace artifact not found at "
                f"'{compile_config.precompile_artifact_dir}/"
                f"{_FX_TRACE_ARTIFACT_KEY}.bin'. Run precompile_main first."
            )
        runtime_meshes = get_spmd_precompile_meshes(parallelism_context)
        traced: TracedResult = precompile_fx_trace_load(
            storage,
            expected_fingerprint=compute_config_fingerprint(
                stage.submod,
                compile_config,
                parallelism_context,
            ),
            example_inputs=flatten_runtime_inputs(
                stage.submod,
                runtime_args,
                {},
                precompile_meshes=runtime_meshes,
            ),
        )
        return traced, runtime_meshes

    full_forward_backward_step: Callable[..., Any] = make_fwd_bwd_step(
        stage.submod,
        loss_fn,
    )
    traced = minimal_fx_tracer(
        full_forward_backward_step,
        module=stage.submod,
    )(*runtime_args)
    return traced, runtime_meshes


def _bind_direct_joint_stage_graph(
    stage: GraphPipelineStage,
    traced: TracedResult,
    *,
    trainer_config: "GraphTrainer.Config",
    parallelism_context: ParallelismContext,
    num_param_grads: int,
    runtime_meshes: list[DeviceMesh] | None,
    outer_cudagraphs_enabled: bool,
) -> None:
    """Apply passes and bind the SPMD without gradient accumulation executor."""
    passes: list[Callable] = construct_joint_train_step_passes(
        traced,
        trainer_config,
        parallelism_context=parallelism_context,
        outer_cudagraphs_enabled=outer_cudagraphs_enabled,
    )
    traced.gm = apply_graph_passes(
        traced.gm,
        traced.example_inputs,
        passes,
        compile_config=trainer_config.compile,
        respect_disable_passes=trainer_config.compile.enable_passes,
    )
    stage.graphs = GraphTrainerJointStageGraphs(
        traced=traced,
        module=stage.submod,
        num_param_grads=num_param_grads,
        runtime_meshes=runtime_meshes,
    )


def _trace_spmd_stage_graph(
    stage: GraphPipelineStage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
    *,
    loss_fn: Callable,
    compile_config: GraphTrainerCompileConfig,
    parallelism_context: ParallelismContext,
) -> tuple[TracedResult, list[DeviceMesh] | None, int]:
    """Trace the joint graph for either SPMD path.

    Returns the traced graph, the precompile runtime meshes, and the number of
    parameter gradients.
    """
    if not stage.is_first or not stage.is_last or len(args) != 1:
        raise ValueError(
            "Joint forward/backward requires one SPMD stage and one model input"
        )

    # Calling convention:
    # (model_input, target, global_loss_token_counts, model_kwargs)
    runtime_args: tuple[Any, Any, Any, dict[str, Any]] = (
        args[0],
        target,
        loss_kwargs["global_loss_token_counts"],
        kwargs,
    )
    traced: TracedResult
    runtime_meshes: list[DeviceMesh] | None
    traced, runtime_meshes = _trace_joint_stage_graph(
        stage,
        runtime_args,
        loss_fn=loss_fn,
        compile_config=compile_config,
        parallelism_context=parallelism_context,
    )
    num_param_grads: int = sum(
        parameter.requires_grad
        for _, parameter in stage.submod.named_parameters(remove_duplicate=False)
    )
    return traced, runtime_meshes, num_param_grads


def _build_fwd_bwd_graphs(
    stage: GraphPipelineStage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
    *,
    loss_fn: Callable,
    trainer_config: "GraphTrainer.Config",
    parallelism_context: ParallelismContext,
    outer_cudagraphs_enabled: bool,
) -> None:
    """Build the SPMD without gradient accumulation graph executor."""
    traced, runtime_meshes, num_param_grads = _trace_spmd_stage_graph(
        stage,
        args,
        kwargs,
        target,
        loss_kwargs,
        loss_fn=loss_fn,
        compile_config=trainer_config.compile,
        parallelism_context=parallelism_context,
    )
    _bind_direct_joint_stage_graph(
        stage,
        traced,
        trainer_config=trainer_config,
        parallelism_context=parallelism_context,
        num_param_grads=num_param_grads,
        runtime_meshes=runtime_meshes,
        outer_cudagraphs_enabled=outer_cudagraphs_enabled,
    )
