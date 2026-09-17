# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineScheduleRuntime,
    BACKWARD_WEIGHT,
    FORWARD,
    FULL_BACKWARD,
    get_schedule_class,
    REDUCE_GRAD,
    RESHARD,
    UNSHARD,
)

from torchtitan.components.loss import LossFunction
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.pipeline_parallel import (
    _build_get_mesh_callback,
    _build_pipeline_schedule,
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
    _get_pp_rank_to_stage_indices_mapping,
    _split_module,
)
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    trace_input_preparer_keys,
)
from torchtitan.experiments.graph_trainer.graph_pp.graph_builder import (
    GraphTrainerStageGraphProvider,
)
from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    BACKWARD,
    BACKWARD_WEIGHT_WITH_REDUCE_GRAD,
    BACKWARD_WITH_REDUCE_GRAD,
    FULL_FORWARD_BACKWARD,
    GraphRuntime,
    register_graph_schedule,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import GraphPipelineStage
from torchtitan.experiments.graph_trainer.registry import (
    PASS_PIPELINE_REGISTRY,
    TRACE_CALL_INPUT_PREPARERS,
    TRACE_INPUT_PREPARERS,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ParallelizeFunction


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphRuntimeFSDPPolicy:
    extract_fsdp_param_unshard: bool
    extract_fsdp_grad_reduction: bool


def resolve_graph_runtime_fsdp_policy(
    compile_config: GraphTrainerCompileConfig,
    *,
    pp_enabled: bool,
    fsdp_enabled: bool,
) -> GraphRuntimeFSDPPolicy:
    """Resolve topology-dependent FSDP graph boundaries.

    PP=1 keeps FSDP operations in their compute graphs by default. PP>1
    extracts them into explicit schedule actions by default.
    """
    if not fsdp_enabled:
        if compile_config.fsdp_param_unshard_mode == "extracted_in_schedule_stage":
            raise ValueError("Extracted FSDP parameter unsharding requires FSDP")
        if compile_config.fsdp_gradient_sync_mode == "deferred_as_schedule_stage":
            raise ValueError("Deferred FSDP gradient reduction requires FSDP")
        return GraphRuntimeFSDPPolicy(
            extract_fsdp_param_unshard=False,
            extract_fsdp_grad_reduction=False,
        )

    if not pp_enabled:
        return GraphRuntimeFSDPPolicy(
            extract_fsdp_param_unshard=(
                compile_config.fsdp_param_unshard_mode == "extracted_in_schedule_stage"
            ),
            extract_fsdp_grad_reduction=(
                compile_config.fsdp_gradient_sync_mode == "deferred_as_schedule_stage"
            ),
        )

    if compile_config.fsdp_param_unshard_mode == "in_graph":
        raise ValueError("PP>1 GraphPP requires extracted FSDP parameter unsharding")
    if compile_config.fsdp_gradient_sync_mode == "in_graph":
        raise ValueError("PP>1 GraphPP requires deferred FSDP gradient reduction")
    return GraphRuntimeFSDPPolicy(
        extract_fsdp_param_unshard=True,
        extract_fsdp_grad_reduction=True,
    )


def _validate_spmd_graph_runtime_config(
    compile_config: GraphTrainerCompileConfig,
) -> None:
    if compile_config.mode != "aot_fx_trace":
        raise ValueError("GraphRuntime requires --compile.mode aot_fx_trace")
    if compile_config.precompile_artifact_dir:
        raise ValueError(
            "GraphRuntime does not support --compile.precompile_artifact_dir yet. "
            "Existing artifacts contain one train-step graph, but GraphRuntime "
            "requires separate action graphs."
        )


def _new_spmd_runtime_schedule(
    stage: GraphPipelineStage,
    *,
    num_microbatches: int,
    loss_fn: LossFunction,
) -> _PipelineScheduleRuntime:
    """Create an empty runtime schedule for one SPMD model stage."""

    def scalar_loss_fn(*args: object, **kwargs: object) -> torch.Tensor:
        loss = loss_fn(*args, **kwargs)
        return loss[0] if isinstance(loss, tuple) else loss

    return _PipelineScheduleRuntime(
        [stage],
        n_microbatches=num_microbatches,
        loss_fn=scalar_loss_fn,
        scale_grads=False,
        backward_requires_autograd=False,
    )


def _make_simple_spmd_runtime_schedule(
    stage: GraphPipelineStage,
    *,
    loss_fn: LossFunction,
) -> _PipelineScheduleRuntime:
    """Build the one-action schedule for conventional SPMD execution."""
    schedule = _new_spmd_runtime_schedule(
        stage,
        num_microbatches=1,
        loss_fn=loss_fn,
    )
    actions: list[_Action | None] = [
        _Action(
            0,
            cast(Any, FULL_FORWARD_BACKWARD),
            0,
            (_Action(0, FORWARD, 0), _Action(0, FULL_BACKWARD, 0)),
        )
    ]
    schedule._prepare_schedule_with_comms({0: actions}, format="compute_comms")
    _set_graph_backward_actions(
        schedule,
        extract_fsdp_grad_reduction=False,
    )
    return schedule


def _make_scheduled_spmd_runtime_schedule(
    stage: GraphPipelineStage,
    *,
    num_microbatches: int,
    parallelism: ParallelismConfig,
    loss_fn: LossFunction,
    fsdp_enabled: bool,
    extract_fsdp_param_unshard: bool,
    extract_fsdp_grad_reduction: bool,
) -> _PipelineScheduleRuntime:
    """Build an SPMD schedule with explicit microbatch and FSDP actions."""
    schedule = _new_spmd_runtime_schedule(
        stage,
        num_microbatches=num_microbatches,
        loss_fn=loss_fn,
    )
    fsdp_reshard_after_forward = (
        get_fsdp_reshard_after_forward_policy(
            parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
        )
        if fsdp_enabled
        else None
    )
    reuse_unsharded_parameters = (
        extract_fsdp_param_unshard and fsdp_reshard_after_forward is False
    )
    actions: list[_Action | None] = []
    if reuse_unsharded_parameters:
        actions.append(_Action(0, UNSHARD))
    for microbatch_index in range(num_microbatches):
        if extract_fsdp_param_unshard and not reuse_unsharded_parameters:
            actions.append(_Action(0, UNSHARD))
        actions.extend(
            (
                _Action(0, FORWARD, microbatch_index),
                _Action(0, FULL_BACKWARD, microbatch_index),
            )
        )
        if extract_fsdp_param_unshard and not reuse_unsharded_parameters:
            actions.append(_Action(0, RESHARD))
    if extract_fsdp_grad_reduction:
        actions.append(_Action(0, REDUCE_GRAD))
    if reuse_unsharded_parameters:
        actions.append(_Action(0, RESHARD))
    schedule._prepare_schedule_with_comms({0: actions}, format="compute_comms")
    _set_graph_backward_actions(
        schedule,
        extract_fsdp_grad_reduction=extract_fsdp_grad_reduction,
    )
    return schedule


def _set_graph_backward_actions(
    schedule: _PipelineScheduleRuntime,
    *,
    extract_fsdp_grad_reduction: bool,
) -> None:
    """Make gradient-reduction placement explicit in the runtime schedule."""

    def replace_action(action: _Action) -> _Action:
        computation_type = action.computation_type
        if computation_type == FULL_BACKWARD:
            computation_type = (
                BACKWARD if extract_fsdp_grad_reduction else BACKWARD_WITH_REDUCE_GRAD
            )
        elif computation_type == BACKWARD_WEIGHT and not extract_fsdp_grad_reduction:
            computation_type = BACKWARD_WEIGHT_WITH_REDUCE_GRAD

        sub_actions = (
            None
            if action.sub_actions is None
            else tuple(replace_action(sub_action) for sub_action in action.sub_actions)
        )
        return _Action(
            action.stage_index,
            cast(Any, computation_type),
            action.microbatch_index,
            sub_actions,
        )

    schedule.pipeline_order_with_comms = {
        rank: [replace_action(action) for action in actions]
        for rank, actions in schedule.pipeline_order_with_comms.items()
    }


def _make_pipeline_parallel_runtime_schedule(
    stages: list[GraphPipelineStage],
    *,
    num_microbatches: int,
    parallelism: ParallelismConfig,
    loss_fn: LossFunction,
    extract_fsdp_grad_reduction: bool,
) -> _PipelineScheduleRuntime:
    """Build a real-PP schedule through the upstream schedule implementation."""
    schedule = _build_pipeline_schedule(
        parallelism=parallelism,
        num_microbatches=num_microbatches,
        stages=stages,  # pyrefly: ignore [bad-argument-type]
        loss_fn=loss_fn,
        backward_requires_autograd=False,
    )
    assert isinstance(schedule, _PipelineScheduleRuntime)
    _set_graph_backward_actions(
        schedule,
        extract_fsdp_grad_reduction=extract_fsdp_grad_reduction,
    )
    return schedule


def _validate_graph_pp_config(
    *,
    compile_config: GraphTrainerCompileConfig,
    parallelism: ParallelismConfig,
) -> None:
    if compile_config.mode != "aot_fx_trace":
        raise ValueError("GraphPP requires --compile.mode aot_fx_trace")
    if compile_config.precompile_artifact_dir:
        raise ValueError(
            "GraphPP does not support --compile.precompile_artifact_dir yet. "
            "Trace and graph construction are stage-local runtime operations."
        )
    if parallelism.fsdp_reshard_after_forward == "always":
        raise ValueError(
            "GraphPP assumes ZeRO-2 style FSDP with "
            "--parallelism.fsdp_reshard_after_forward default/never, not always."
        )
    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    if not issubclass(schedule_class, _PipelineScheduleRuntime):
        raise ValueError(
            "GraphPP currently requires a runtime PP schedule such as "
            "Interleaved1F1B, ZBVZeroBubble, or DualPipeV. "
            f"Got {parallelism.pipeline_parallel_schedule}."
        )


def _register_graph_runtime(
    schedule: _PipelineScheduleRuntime,
    *,
    fsdp_policy: GraphRuntimeFSDPPolicy,
    compile_config: GraphTrainerCompileConfig,
    model_config: BaseModel.Config | None,
    parallelism: ParallelismConfig,
    loss_fn: LossFunction,
    pass_config: Any | None,
    parallel_dims: ParallelDims,
    warn_if_cudagraph_pass_requested: bool,
) -> GraphRuntime:
    """Bind GraphTrainer graph construction to an already chosen schedule."""
    graph_provider = GraphTrainerStageGraphProvider(
        loss_fn=loss_fn,
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
        extract_fsdp_param_unshard=fsdp_policy.extract_fsdp_param_unshard,
        extract_fsdp_grad_reduction=fsdp_policy.extract_fsdp_grad_reduction,
        pass_config=pass_config,
        parallel_dims=parallel_dims,
    )
    if warn_if_cudagraph_pass_requested:
        graph_provider._warn_if_cudagraph_pass_requested()
    return register_graph_schedule(schedule, graph_provider=graph_provider)


def _make_simple_spmd_graph_runtime(
    stage: GraphPipelineStage,
    *,
    compile_config: GraphTrainerCompileConfig,
    model_config: BaseModel.Config | None,
    parallelism: ParallelismConfig,
    loss_fn: LossFunction,
    pass_config: Any,
    parallel_dims: ParallelDims,
) -> GraphRuntime:
    """Build conventional single-microbatch SPMD execution."""
    schedule = _make_simple_spmd_runtime_schedule(stage, loss_fn=loss_fn)
    return _register_graph_runtime(
        schedule,
        fsdp_policy=GraphRuntimeFSDPPolicy(False, False),
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
        loss_fn=loss_fn,
        pass_config=pass_config,
        parallel_dims=parallel_dims,
        warn_if_cudagraph_pass_requested=False,
    )


def _make_scheduled_spmd_graph_runtime(
    stage: GraphPipelineStage,
    *,
    num_microbatches: int,
    fsdp_policy: GraphRuntimeFSDPPolicy,
    compile_config: GraphTrainerCompileConfig,
    model_config: BaseModel.Config | None,
    parallelism: ParallelismConfig,
    loss_fn: LossFunction,
    pass_config: Any,
    parallel_dims: ParallelDims,
) -> GraphRuntime:
    """Build SPMD execution with explicit microbatch or FSDP actions."""
    if compile_config.ep_overlap.enabled:
        raise ValueError(
            "GraphRuntime scheduled SPMD does not support "
            "--compile.ep_overlap.enabled yet. Stage tracing does not apply "
            "the EP-overlap trace-input preparers."
        )
    if compile_config.memory_policy == "sac_and_offload":
        raise ValueError(
            "GraphRuntime scheduled SPMD does not support "
            "--compile.memory_policy sac_and_offload yet. Graph partitioning "
            "must preserve offload and reload pairs across the forward/backward "
            "boundary."
        )
    if compile_config.pass_pipeline in PASS_PIPELINE_REGISTRY:
        raise ValueError(
            "GraphRuntime scheduled SPMD does not support custom pass pipelines yet"
        )
    trace_preparer_names = set(trace_input_preparer_keys(compile_config))
    unsupported_preparers = trace_preparer_names.intersection(
        TRACE_INPUT_PREPARERS.keys() | TRACE_CALL_INPUT_PREPARERS.keys()
    )
    if unsupported_preparers:
        raise ValueError(
            "GraphRuntime scheduled SPMD does not support trace-input preparers "
            f"yet: {sorted(unsupported_preparers)}"
        )

    schedule = _make_scheduled_spmd_runtime_schedule(
        stage,
        num_microbatches=num_microbatches,
        parallelism=parallelism,
        loss_fn=loss_fn,
        fsdp_enabled=parallel_dims.fsdp_enabled,
        extract_fsdp_param_unshard=fsdp_policy.extract_fsdp_param_unshard,
        extract_fsdp_grad_reduction=fsdp_policy.extract_fsdp_grad_reduction,
    )
    return _register_graph_runtime(
        schedule,
        fsdp_policy=fsdp_policy,
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
        loss_fn=loss_fn,
        pass_config=pass_config,
        parallel_dims=parallel_dims,
        warn_if_cudagraph_pass_requested=False,
    )


def _make_pipeline_parallel_graph_runtime(
    stages: list[GraphPipelineStage],
    *,
    num_microbatches: int,
    fsdp_policy: GraphRuntimeFSDPPolicy,
    compile_config: GraphTrainerCompileConfig,
    model_config: BaseModel.Config | None,
    parallelism: ParallelismConfig,
    loss_fn: LossFunction,
    parallel_dims: ParallelDims,
) -> GraphRuntime:
    """Build graph execution around a real pipeline-parallel schedule."""
    schedule = _make_pipeline_parallel_runtime_schedule(
        stages,
        num_microbatches=num_microbatches,
        parallelism=parallelism,
        loss_fn=loss_fn,
        extract_fsdp_grad_reduction=fsdp_policy.extract_fsdp_grad_reduction,
    )
    return _register_graph_runtime(
        schedule,
        fsdp_policy=fsdp_policy,
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
        loss_fn=loss_fn,
        pass_config=None,
        parallel_dims=parallel_dims,
        warn_if_cudagraph_pass_requested=True,
    )


def make_graph_runtime(
    stages: list[GraphPipelineStage],
    *,
    num_microbatches: int,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    model_config: BaseModel.Config | None,
    loss_fn: LossFunction,
    pass_config: Any | None,
) -> GraphRuntime:
    """Build the schedule, stage graphs, and runtime for GraphTrainer.

    Descriptive notation
    --------------------
    ``s`` is a stage index and ``m`` is a microbatch index:

    - ``FULL_FORWARD_BACKWARD(s, m)`` runs one joint train graph.
    - ``UNSHARD_FORWARD(s, m)`` runs a forward graph containing FSDP
      parameter all-gather.
    - ``FORWARD(s, m)`` runs a forward graph with all-gather extracted.
    - ``BACKWARD_WITH_REDUCE_GRAD(s, m)`` runs a backward graph containing
      FSDP gradient reduction.
    - ``BACKWARD(s, m)`` runs a backward graph with gradient reduction
      extracted.
    - ``UNSHARD(s)``, ``REDUCE_GRAD(s)``, and ``RESHARD(s)`` are explicit,
      stage-local schedule actions.

    GraphRuntime makes the two backward variants explicit after the upstream
    PP schedule has finished communication and FSDP lowering.

    Runtime lifecycle
    -----------------
    Every path follows the same lifecycle:

    1. Trace a joint graph for each local stage.
    2. Apply graph passes.
    3. Split the joint graph when the schedule needs separate action graphs.
    4. Bind the graphs to schedule actions.
    5. Initialize transient state for each local stage.
    6. Execute the schedule.
    7. Commit final parameter gradients to ``param.grad``.
    8. Clear transient stage state.

    ``param.grad`` remains live until the optimizer step. Parameters, buffers,
    saved activations, and intermediate gradient accumulators in
    ``stage.state`` live for one runtime invocation.

    Action state transitions
    ------------------------
    Before the first action, the runtime binds the stage graphs and records the
    live parameters, buffers, and trainable parameters in ``stage.state``.

    - ``FULL_FORWARD_BACKWARD`` appends the loss to ``stage.output_chunks`` and
      ``schedule._internal_losses``, increments the stage backward counter, and
      accumulates parameter gradients directly into ``param.grad``.
    - ``FORWARD`` waits for and consumes any remote input receive, materializes
      parameter inputs if needed, saves its output and backward values in
      ``stage.fwd_cache[m]``, records last-stage losses, and forwards local
      outputs to the next stage.
    - ``BACKWARD`` and ``BACKWARD_WITH_REDUCE_GRAD`` wait for and consume any
      remote gradient receive, increment the stage backward counter, pop
      ``stage.fwd_cache[m]``, and write input gradients to
      ``stage.bwd_cache[m]`` and, when applicable, the previous local stage.
      The former accumulates unsharded gradients for a later ``REDUCE_GRAD``;
      the latter updates ``param.grad`` directly.
    - ``BACKWARD_INPUT`` performs the input-gradient part of that transition
      and saves weight-backward inputs in
      ``stage.saved_values_for_backward_weight_cache[m]``.
      ``BACKWARD_WEIGHT`` pops that entry and defers gradient reduction;
      ``BACKWARD_WEIGHT_WITH_REDUCE_GRAD`` accumulates directly into
      ``param.grad``.
    - ``OVERLAP_F_B`` applies the same forward and backward transitions to its
      two stages in one multiplexed graph call.
    - ``UNSHARD`` populates ``stage.state.unsharded_param_values``;
      ``RESHARD`` clears it. ``REDUCE_GRAD`` populates
      ``stage.state.sharded_param_grads`` and applies schedule gradient scaling
      once.

    Direct backward results initialize an empty ``param.grad`` with a clone and
    add later microbatches in-place. Deferred reduction instead accumulates
    unsharded graph outputs in ``stage.state`` and commits the reduced result to
    ``param.grad`` after successful schedule execution.

    On exit, including after an exception, the runtime clears ``stage.state``,
    its bound-graph lookup, and per-call loss arguments. Evaluation performs
    the same setup and cleanup without committing gradients.

    Simple SPMD
    -----------
    PP=1 with one microbatch and no extracted FSDP boundaries uses one joint
    graph::

        FULL_FORWARD_BACKWARD(stage=0, microbatch=0)
            stage.output_chunks.append(loss)
            schedule._internal_losses.append(loss)
            param.grad += parameter_grads

    SPMD with gradient accumulation
    -------------------------------
    Gradient accumulation is represented by ``N`` schedule microbatches. With
    the default PP=1 FSDP policy, all-gather and gradient reduction remain in
    each microbatch graph::

        UNSHARD_FORWARD(stage=0, microbatch=0)
            stage.fwd_cache[0] <- (output, saved_values_for_backward)

        BACKWARD_WITH_REDUCE_GRAD(stage=0, microbatch=0)
            pop stage.fwd_cache[0]
            stage.bwd_cache[0] <- input_grads
            param.grad += reduced_param_grads

        ...

        UNSHARD_FORWARD(stage=0, microbatch=N - 1)
            stage.fwd_cache[N - 1] <- (output, saved_values_for_backward)

        BACKWARD_WITH_REDUCE_GRAD(stage=0, microbatch=N - 1)
            pop stage.fwd_cache[N - 1]
            stage.bwd_cache[N - 1] <- input_grads
            param.grad += reduced_param_grads

    Deferred FSDP gradient reduction
    --------------------------------
    Setting ``fsdp_gradient_sync_mode="deferred_as_schedule_stage"`` removes
    gradient reduction from each backward graph. Backward graphs accumulate
    unsharded gradients, and one stage-local ``REDUCE_GRAD`` action reduces the
    result after all microbatches::

        UNSHARD_FORWARD(stage=0, microbatch=0)
        BACKWARD(stage=0, microbatch=0)
            pop stage.fwd_cache[0]
            stage.bwd_cache[0] <- input_grads
            stage.state.unsharded_param_grads += raw_param_grads

        ...

        UNSHARD_FORWARD(stage=0, microbatch=N - 1)
        BACKWARD(stage=0, microbatch=N - 1)
            pop stage.fwd_cache[N - 1]
            stage.bwd_cache[N - 1] <- input_grads
            stage.state.unsharded_param_grads += raw_param_grads

        REDUCE_GRAD(stage=0)
            stage.state.sharded_param_grads <-
                reduce(stage.state.unsharded_param_grads)

    At successful step exit, the runtime commits the sharded gradients to live
    ``param.grad`` and clears ``stage.state``.

    Explicit FSDP parameter unsharding
    -----------------------------------
    Setting ``fsdp_param_unshard_mode="extracted_in_schedule_stage"`` removes
    all-gather from the forward graph. With the ``default`` or ``always``
    reshard policy, every microbatch is wrapped separately::

        UNSHARD(stage=0)
        FORWARD(stage=0, microbatch=0)
        BACKWARD_WITH_REDUCE_GRAD(stage=0, microbatch=0)
        RESHARD(stage=0)

        ...

        UNSHARD(stage=0)
        FORWARD(stage=0, microbatch=N - 1)
        BACKWARD_WITH_REDUCE_GRAD(stage=0, microbatch=N - 1)
        RESHARD(stage=0)

    With ``fsdp_reshard_after_forward="never"``, one unshard is reused by all
    microbatches::

        UNSHARD(stage=0)
        FORWARD(stage=0, microbatch=0)
        BACKWARD_WITH_REDUCE_GRAD(stage=0, microbatch=0)
        ...
        FORWARD(stage=0, microbatch=N - 1)
        BACKWARD_WITH_REDUCE_GRAD(stage=0, microbatch=N - 1)
        RESHARD(stage=0)

    Parameter unsharding and gradient reduction are independent. If both are
    extracted, replace each ``BACKWARD_WITH_REDUCE_GRAD`` above with
    ``BACKWARD`` and add one ``REDUCE_GRAD(stage=0)`` after the final backward.

    Pipeline parallelism
    --------------------
    For PP>1, the upstream schedule owns action ordering and communication.
    The state transitions above apply independently to each physical or virtual
    stage. Current schedules emit one ``REDUCE_GRAD(s)`` after each stage's
    final backward. ``UNSHARD(s)`` and ``RESHARD(s)`` may run more than once per
    stage; none of these actions is global.

    Local stages exchange forward outputs and input gradients through upstream
    stage caches. The upstream schedule creates and waits for remote P2P
    operations.

    Args:
        stages: Local graph stages. PP=1 requires exactly one stage.
        num_microbatches: Trainer accumulation steps for PP=1, or configured
            pipeline microbatches for PP>1.
        parallel_dims: Parallel topology used to select PP=1 or PP>1 behavior.
        parallelism: Parallel configuration used to construct the schedule.
        compile_config: GraphTrainer execution-mode configuration.
        model_config: Model configuration consumed by graph passes.
        loss_fn: Loss function used by the schedule and graph provider.
        pass_config: Full Trainer configuration required by the PP=1 joint
            graph, or ``None`` for PP>1.
    """
    if num_microbatches < 1:
        raise ValueError(
            f"GraphRuntime requires at least one microbatch, got {num_microbatches}"
        )

    pp_enabled = parallel_dims.pp_enabled
    if pp_enabled:
        _validate_graph_pp_config(
            compile_config=compile_config,
            parallelism=parallelism,
        )
    else:
        _validate_spmd_graph_runtime_config(compile_config)
        if len(stages) != 1:
            raise ValueError(f"PP=1 requires one local stage, got {len(stages)}")

    fsdp_policy = resolve_graph_runtime_fsdp_policy(
        compile_config,
        pp_enabled=pp_enabled,
        fsdp_enabled=parallel_dims.fsdp_enabled,
    )

    if pp_enabled:
        return _make_pipeline_parallel_graph_runtime(
            stages,
            num_microbatches=num_microbatches,
            fsdp_policy=fsdp_policy,
            compile_config=compile_config,
            model_config=model_config,
            parallelism=parallelism,
            loss_fn=loss_fn,
            parallel_dims=parallel_dims,
        )

    use_simple_spmd = (
        num_microbatches == 1
        and not fsdp_policy.extract_fsdp_param_unshard
        and not fsdp_policy.extract_fsdp_grad_reduction
    )
    if use_simple_spmd:
        return _make_simple_spmd_graph_runtime(
            stages[0],
            compile_config=compile_config,
            model_config=model_config,
            parallelism=parallelism,
            loss_fn=loss_fn,
            pass_config=pass_config,
            parallel_dims=parallel_dims,
        )

    return _make_scheduled_spmd_graph_runtime(
        stages[0],
        num_microbatches=num_microbatches,
        fsdp_policy=fsdp_policy,
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
        loss_fn=loss_fn,
        pass_config=pass_config,
        parallel_dims=parallel_dims,
    )


def make_spmd_graph_runtime(
    model: nn.Module,
    *,
    gradient_accumulation_steps: int,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    device: torch.device,
    model_config: BaseModel.Config | None,
    loss_fn: LossFunction,
    trainer_config: Any,
) -> GraphRuntime:
    """Represent one SPMD model as a single-stage graph runtime."""
    pp_mesh = parallel_dims.get_optional_mesh("pp", include_singleton_axes=True)
    assert pp_mesh is not None
    stage = GraphPipelineStage(
        model,
        stage_index=0,
        num_stages=1,
        device=device,
        group=pp_mesh.get_group("pp"),
    )
    return make_graph_runtime(
        [stage],
        num_microbatches=gradient_accumulation_steps,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        compile_config=compile_config,
        model_config=model_config,
        loss_fn=loss_fn,
        pass_config=trainer_config,
    )


def graph_pipeline_llm(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    device: torch.device,
    model_config: BaseModel.Config,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[GraphRuntime, list[nn.Module], bool, bool]:
    """Build a GraphPP pipeline schedule for GraphTrainer.

    Args:
        model: The full model before PP stage splitting.
        parallel_dims: TorchTitan parallel dimension helper.
        training: Training config used for local batch size.
        parallelism: Parallelism config used for PP schedule and module split.
        compile_config: GraphTrainer compile config.
        ac_config: Activation checkpointing config forwarded to ``parallelize_fn``.
        dump_folder: Artifact/debug output directory.
        device: Local device for the stage.
        model_config: Model config consumed by stage graph passes.
        parallelize_fn: Model-specific SPMD parallelization function.
        loss_fn: Loss function used by upstream PP metadata and GraphPP tracing.

    Returns:
        A tuple of ``(runtime, model_parts, has_first_stage, has_last_stage)``.
    """
    pp_mesh = parallel_dims.get_mesh("pp")

    (
        num_virtual_stages,
        num_layers,
        input_weight,
        output_weight,
    ) = _get_pipeline_metadata(parallel_dims, parallelism, model_config)

    module_names_per_stage = parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = _generate_llm_fqn_per_model_part(
            num_virtual_stages,
            num_layers,
            input_weight,
            output_weight,
        )
    for index, stage_modules in enumerate(module_names_per_stage):
        logger.debug("GraphPP stage %s modules: %s", index, stage_modules)

    get_mesh_cb = _build_get_mesh_callback(parallel_dims)
    pp_rank_to_stage_indices = _get_pp_rank_to_stage_indices_mapping(
        pp_mesh.get_local_rank(),
        pp_mesh.size(),
        parallelism.pipeline_parallel_schedule,
        len(module_names_per_stage),
    )
    model_parts: list[nn.Module] = []
    stages: list[GraphPipelineStage] = []
    for stage_index in pp_rank_to_stage_indices:
        model_part = _split_module(model, module_names_per_stage[stage_index])
        model_part = parallelize_fn(
            model_part,
            parallel_dims=parallel_dims,
            training=training,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )
        logger.info(
            "PP rank %s is building GraphPP stage_idx %s with modules %s",
            pp_mesh.get_local_rank(),
            stage_index,
            module_names_per_stage[stage_index],
        )
        model_parts.append(model_part)
        stages.append(
            GraphPipelineStage(
                model_part,
                stage_index=stage_index,
                num_stages=len(module_names_per_stage),
                device=device,
                group=pp_mesh.get_group("pp"),
                get_mesh=get_mesh_cb,
            )
        )

    graph_runtime = make_graph_runtime(
        stages,
        num_microbatches=parallelism.num_pp_microbatches,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        compile_config=compile_config,
        model_config=model_config,
        loss_fn=loss_fn,
        pass_config=None,
    )

    return (
        graph_runtime,
        model_parts,
        any(stage.is_first for stage in stages),
        any(stage.is_last for stage in stages),
    )
