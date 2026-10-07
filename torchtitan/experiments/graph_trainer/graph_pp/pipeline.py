# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
import logging
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineScheduleRuntime,
    BACKWARD_WEIGHT,
    FULL_BACKWARD,
    get_schedule_class,
    RESHARD,
)

from torchtitan.components.loss import LossFunction
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
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
    SPMDGradientAccumulationConfig,
)
from torchtitan.experiments.graph_trainer.graph_builder import (
    GraphExecutionPlan,
    GraphTrainerStageGraphProvider,
    ReduceGradPlacement,
    UnshardPlacement,
)
from torchtitan.experiments.graph_trainer.graph_builder_utils import (
    GraphTrainerConfigView,
)
from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    BACKWARD,
    BACKWARD_WEIGHT_WITH_REDUCE_GRAD,
    BACKWARD_WITH_REDUCE_GRAD,
    GraphRuntime,
    register_graph_schedule,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import GraphPipelineStage
from torchtitan.experiments.graph_trainer.registry import PASS_PIPELINE_REGISTRY
from torchtitan.protocols.model import BaseModel


if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


logger = logging.getLogger(__name__)


def _warn_if_spmd_gradient_accumulation_config_ignored(
    gradient_accumulation_config: SPMDGradientAccumulationConfig,
    *,
    num_microbatches: int,
    pp_enabled: bool,
    fsdp_enabled: bool,
) -> None:
    """Warn when ``compile.spmd_gradient_accumulation`` settings do not apply."""
    default_config = SPMDGradientAccumulationConfig()
    if pp_enabled or num_microbatches == 1:
        if gradient_accumulation_config != default_config:
            reason = (
                "PP always runs FSDP collectives as schedule actions without "
                "WGrad accumulation fusion"
                if pp_enabled
                else "SPMD without gradient accumulation runs one microbatch"
            )
            logger.warning(
                "Ignoring compile.spmd_gradient_accumulation=%s: %s",
                gradient_accumulation_config,
                reason,
            )
        return
    if not fsdp_enabled and (
        gradient_accumulation_config.fsdp_param_unshard_mode
        != default_config.fsdp_param_unshard_mode
        or gradient_accumulation_config.fsdp_grad_reduce_mode
        != default_config.fsdp_grad_reduce_mode
    ):
        logger.warning(
            "Ignoring compile.spmd_gradient_accumulation.fsdp_param_unshard_mode=%r "
            "and fsdp_grad_reduce_mode=%r: FSDP is disabled",
            gradient_accumulation_config.fsdp_param_unshard_mode,
            gradient_accumulation_config.fsdp_grad_reduce_mode,
        )


def _resolve_fsdp_placements(
    gradient_accumulation_config: SPMDGradientAccumulationConfig,
    *,
    num_microbatches: int,
    pp_enabled: bool,
    fsdp_enabled: bool,
) -> tuple[UnshardPlacement | None, ReduceGradPlacement | None]:
    """Resolve where each FSDP boundary runs.

    ``gradient_accumulation_config`` only applies to SPMD with gradient
    accumulation. SPMD without gradient accumulation keeps both boundaries in
    the joint graph, and PP extracts both into the schedule.
    """
    if not fsdp_enabled:
        return None, None
    if pp_enabled:
        return "schedule", "schedule"
    if num_microbatches == 1:
        return "every_microbatch", "every_microbatch"

    unshard = gradient_accumulation_config.fsdp_param_unshard_mode
    reduce_grad = gradient_accumulation_config.fsdp_grad_reduce_mode
    if (unshard == "first_microbatch" and reduce_grad == "every_microbatch") or (
        reduce_grad == "last_microbatch" and unshard == "every_microbatch"
    ):
        raise ValueError(
            "First/last-microbatch FSDP boundaries cannot be combined with "
            "the other boundary inside every joint graph"
        )
    return unshard, reduce_grad


def _resolve_fuse_wgrad_accumulation(
    compile_config: GraphTrainerCompileConfig,
    *,
    num_microbatches: int,
    pp_enabled: bool,
    fsdp_enabled: bool,
    reduce_grad: ReduceGradPlacement | None,
) -> bool:
    """Resolve whether WGrad producers accumulate into gradient buffers.

    Only SPMD with gradient accumulation can fuse. PP does not fuse until
    its schedule spans the complete optimizer step.
    """
    if pp_enabled or num_microbatches == 1:
        return False
    fusion_mode = compile_config.spmd_gradient_accumulation.fuse_wgrad_accumulation
    reduce_grad_in_last_microbatch = reduce_grad == "last_microbatch"
    if fusion_mode == "enabled":
        if not compile_config.enable_passes:
            raise ValueError("WGrad accumulation fusion requires graph passes")
        if "fuse_wgrad_accumulation_pass" in compile_config.disable_passes:
            raise ValueError(
                "WGrad accumulation fusion is enabled but its pass is disabled"
            )
        if fsdp_enabled and not reduce_grad_in_last_microbatch:
            raise ValueError(
                "WGrad accumulation fusion with FSDP requires "
                "compile.spmd_gradient_accumulation.fsdp_grad_reduce_mode="
                "'last_microbatch'"
            )

    can_fuse_wgrad = not fsdp_enabled or reduce_grad_in_last_microbatch
    return (
        can_fuse_wgrad
        and compile_config.enable_passes
        and "fuse_wgrad_accumulation_pass" not in compile_config.disable_passes
        and (
            fusion_mode == "enabled"
            or (fusion_mode == "auto" and compile_config.numerics_changing_optim)
        )
    )


def _validate_graph_pp_config(
    *,
    compile_config: GraphTrainerCompileConfig,
    parallelism: ParallelismConfig,
) -> None:
    if compile_config.precompile_artifact_dir:
        raise ValueError(
            "GraphPP does not support compile.precompile_artifact_dir yet. "
            "Trace and graph construction are stage-local runtime operations."
        )
    if parallelism.fsdp_reshard_after_forward == "always":
        raise ValueError(
            "GraphPP assumes ZeRO-2 style FSDP with "
            "parallelism.fsdp_reshard_after_forward='default'/'never', not 'always'."
        )
    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    if not issubclass(schedule_class, _PipelineScheduleRuntime):
        raise ValueError(
            "GraphPP currently requires a runtime PP schedule such as "
            "Interleaved1F1B, ZBVZeroBubble, or DualPipeV. "
            f"Got {parallelism.pipeline_parallel_schedule}."
        )


def _validate_spmd_gradient_accumulation_support(
    compile_config: GraphTrainerCompileConfig,
) -> None:
    """Reject features not yet validated with SPMD with gradient accumulation."""
    if compile_config.precompile_artifact_dir:
        raise ValueError(
            "SPMD with gradient accumulation does not support "
            "compile.precompile_artifact_dir yet"
        )
    if compile_config.ep_overlap.enabled:
        raise ValueError(
            "SPMD with gradient accumulation does not support "
            "compile.ep_overlap.enabled yet. The EP-overlap graph rewrites "
            "have not been validated with per-microbatch joint graphs."
        )
    if compile_config.memory_policy == "sac_and_offload":
        raise ValueError(
            "SPMD with gradient accumulation does not support "
            "compile.memory_policy='sac_and_offload' yet. Joint graph "
            "rewrites must preserve offload and reload pairs."
        )
    if compile_config.pass_pipeline in PASS_PIPELINE_REGISTRY:
        raise ValueError(
            "SPMD with gradient accumulation does not support custom pass "
            "pipelines yet"
        )


def resolve_graph_execution_plan(
    compile_config: GraphTrainerCompileConfig,
    *,
    num_microbatches: int,
    parallelism: ParallelismConfig,
    pp_enabled: bool,
    fsdp_enabled: bool,
) -> GraphExecutionPlan:
    """Resolve and validate every GraphRuntime placement decision once.

    Args:
        compile_config: GraphTrainer compile configuration.
        num_microbatches: Trainer accumulation steps for SPMD, or configured
            pipeline microbatches for PP.
        parallelism: Parallelism configuration.
        pp_enabled: Whether pipeline parallelism is enabled.
        fsdp_enabled: Whether FSDP is enabled.
    """
    if pp_enabled:
        _validate_graph_pp_config(
            compile_config=compile_config,
            parallelism=parallelism,
        )

    _warn_if_spmd_gradient_accumulation_config_ignored(
        compile_config.spmd_gradient_accumulation,
        num_microbatches=num_microbatches,
        pp_enabled=pp_enabled,
        fsdp_enabled=fsdp_enabled,
    )
    unshard, reduce_grad = _resolve_fsdp_placements(
        compile_config.spmd_gradient_accumulation,
        num_microbatches=num_microbatches,
        pp_enabled=pp_enabled,
        fsdp_enabled=fsdp_enabled,
    )
    fuse_wgrad_accumulation = _resolve_fuse_wgrad_accumulation(
        compile_config,
        num_microbatches=num_microbatches,
        pp_enabled=pp_enabled,
        fsdp_enabled=fsdp_enabled,
        reduce_grad=reduce_grad,
    )
    plan = GraphExecutionPlan(
        pp_enabled=pp_enabled,
        num_microbatches=num_microbatches,
        unshard=unshard,
        reduce_grad=reduce_grad,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
    )

    if not pp_enabled and plan.has_gradient_accumulation:
        _validate_spmd_gradient_accumulation_support(compile_config)
    if plan.reuse_unsharded_parameters and (
        parallelism.fsdp_reshard_after_forward == "always"
    ):
        logger.warning(
            "Ignoring parallelism.fsdp_reshard_after_forward='always': "
            "compile.spmd_gradient_accumulation.fsdp_param_unshard_mode=%r "
            "reuses unsharded parameters across microbatches",
            unshard,
        )
    if compile_config.enable_fsdp_dense_region_overlap and (
        plan.split_fsdp_param_unshard or plan.split_fsdp_grad_reduction
    ):
        raise ValueError(
            "FSDP dense-region overlap requires parameter all-gathers and "
            "gradient reductions to remain inside the compute graphs"
        )
    return plan


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


def _make_spmd_runtime_schedule(
    stage: GraphPipelineStage,
    *,
    loss_fn: LossFunction,
    plan: GraphExecutionPlan,
) -> _PipelineScheduleRuntime:
    """Build the schedule for either SPMD path from a resolved plan."""
    schedule = _new_spmd_runtime_schedule(
        stage,
        num_microbatches=plan.num_microbatches,
        loss_fn=loss_fn,
    )
    actions: list[_Action] = [
        _Action(
            0,
            cast(Any, plan.computation_type_for_microbatch(microbatch_index)),
            microbatch_index,
        )
        for microbatch_index in range(plan.num_microbatches)
    ]
    if plan.reuse_unsharded_parameters:
        actions.append(_Action(0, RESHARD))
    # Upstream schedule validation only recognizes separate F/B actions. SPMD
    # has no pipeline communication to lower, so install the already-lowered
    # joint schedule directly instead of misrepresenting FORWARD_BACKWARD
    # as a container of split actions.
    schedule.stage_index_to_group_rank = {0: 0}
    stage.stage_index_to_group_rank = schedule.stage_index_to_group_rank
    schedule.pipeline_order_with_comms = {0: actions}
    return schedule


def _set_graph_backward_actions(
    schedule: _PipelineScheduleRuntime,
    *,
    extract_fsdp_grad_reduction: bool,
) -> None:
    """Replace PyTorch backward actions with explicit GraphRuntime variants."""

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
) -> tuple[_PipelineScheduleRuntime, _PipelineScheduleRuntime]:
    """Build a real-PP schedule through the upstream schedule implementation."""
    schedule = _build_pipeline_schedule(
        parallelism=parallelism,
        num_microbatches=num_microbatches,
        stages=stages,  # pyrefly: ignore [bad-argument-type]
        loss_fn=loss_fn,
        backward_requires_autograd=False,
    )
    assert isinstance(schedule, _PipelineScheduleRuntime)
    liveness_schedule = copy.copy(schedule)
    _set_graph_backward_actions(
        schedule,
        extract_fsdp_grad_reduction=extract_fsdp_grad_reduction,
    )
    return schedule, liveness_schedule


def _register_graph_runtime(
    schedule: _PipelineScheduleRuntime,
    *,
    plan: GraphExecutionPlan,
    config: "GraphTrainer.Config | GraphTrainerConfigView",
    loss_fn: LossFunction,
    parallelism_context: ParallelismContext,
    warn_if_cuda_graph_pass_requested: bool,
    liveness_schedule: _PipelineScheduleRuntime | None = None,
) -> GraphRuntime:
    """Bind GraphTrainer graph construction to an already chosen schedule."""
    graph_provider = GraphTrainerStageGraphProvider(
        loss_fn=loss_fn,
        config=config,
        plan=plan,
        parallelism_context=parallelism_context,
    )
    if warn_if_cuda_graph_pass_requested:
        graph_provider._warn_if_cuda_graph_pass_requested()
    return register_graph_schedule(
        schedule,
        graph_provider=graph_provider,
        is_spmd=not plan.pp_enabled,
        liveness_schedule=liveness_schedule,
    )


def _make_spmd_graph_runtime(
    stage: GraphPipelineStage,
    *,
    plan: GraphExecutionPlan,
    trainer_config: "GraphTrainer.Config",
    loss_fn: LossFunction,
    parallelism_context: ParallelismContext,
) -> GraphRuntime:
    """Build the runtime for either SPMD path."""
    if (
        plan.reuse_unsharded_parameters
        and trainer_config.parallelism.fsdp_reshard_after_forward != "never"
    ):
        # Unsharded parameters live until the final RESHARD, so graph passes
        # must not treat FSDP all-gathers as resharded after forward.
        trainer_config = dataclasses.replace(
            trainer_config,
            parallelism=dataclasses.replace(
                trainer_config.parallelism, fsdp_reshard_after_forward="never"
            ),
        )
    schedule = _make_spmd_runtime_schedule(
        stage,
        loss_fn=loss_fn,
        plan=plan,
    )
    return _register_graph_runtime(
        schedule,
        plan=plan,
        config=trainer_config,
        loss_fn=loss_fn,
        parallelism_context=parallelism_context,
        warn_if_cuda_graph_pass_requested=False,
    )


def _make_pipeline_parallel_graph_runtime(
    stages: list[GraphPipelineStage],
    *,
    plan: GraphExecutionPlan,
    config: "GraphTrainer.Config | GraphTrainerConfigView",
    loss_fn: LossFunction,
    parallelism_context: ParallelismContext,
) -> GraphRuntime:
    """Build graph execution around a real pipeline-parallel schedule."""
    schedule, liveness_schedule = _make_pipeline_parallel_runtime_schedule(
        stages,
        num_microbatches=plan.num_microbatches,
        parallelism=config.parallelism,
        loss_fn=loss_fn,
        extract_fsdp_grad_reduction=plan.extract_fsdp_grad_reduction,
    )
    return _register_graph_runtime(
        schedule,
        plan=plan,
        config=config,
        loss_fn=loss_fn,
        parallelism_context=parallelism_context,
        warn_if_cuda_graph_pass_requested=True,
        liveness_schedule=liveness_schedule,
    )


def make_graph_runtime(
    stages: list[GraphPipelineStage],
    *,
    num_microbatches: int,
    parallelism_context: ParallelismContext,
    config: "GraphTrainer.Config | GraphTrainerConfigView",
    loss_fn: LossFunction,
) -> GraphRuntime:
    """Build the GraphTrainer schedule and runtime with a stage-graph provider.

    Execution paths
    ---------------
    - SPMD without gradient accumulation: no pipeline parallelism, one
      microbatch per step.
    - SPMD with gradient accumulation: no pipeline parallelism, more than one
      microbatch per step.
    - PP: pipeline parallelism.

    Descriptive notation
    --------------------
    ``s`` is a stage index and ``m`` is a microbatch index:

    - ``FORWARD_BACKWARD_NOGRADACCUM(s, 0)`` runs the first SPMD with
      gradient accumulation joint graph. It takes no accumulator inputs and
      returns gradients used as accumulator inputs by later graphs.
    - ``FORWARD_BACKWARD(s, m)`` runs an SPMD with gradient accumulation joint
      graph without FSDP collectives. Repeated calls update the gradient
      accumulator inputs and return the authoritative handles for the next
      graph call.
    - ``FULL_FORWARD_BACKWARD(s, m)`` runs an SPMD joint graph containing both
      FSDP parameter all-gathers and gradient reductions. It is the only graph
      for SPMD without gradient accumulation.
    - ``FORWARD_BACKWARD_FIRST_WITH_UNSHARD(s, 0)`` also unshards parameters
      and returns unsharded parameters and gradients that become gradient
      accumulators.
    - ``FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD(s, N - 1)`` takes unsharded
      parameters and gradient accumulators, accumulates unsharded gradients
      into them in place, and reduces the accumulated gradients.
    - ``FORWARD(s, m)`` runs a PP stage-forward graph.
    - ``BACKWARD_WITH_REDUCE_GRAD(s, m)`` runs a PP stage-backward graph
      containing FSDP gradient reduction.
    - ``BACKWARD(s, m)`` runs a PP stage-backward graph with gradient
      reduction extracted.
    - ``UNSHARD(s)``, ``REDUCE_GRAD(s)``, and ``RESHARD(s)`` are explicit,
      stage-local schedule actions. SPMD with gradient accumulation only uses
      ``RESHARD``.

    For PP, PyTorch pipeline schedules emit ``FULL_BACKWARD``. GraphRuntime
    replaces it with an explicit backward variant before graph construction;
    ``FULL_BACKWARD`` is not part of GraphRuntime IR.

    Action state transitions
    ------------------------
    On the first action, the graph provider prepares any missing stage graphs
    and the runtime binds them, recording parameter and buffer references and
    trainable parameters in ``stage.state``.

    - ``FORWARD_BACKWARD_NOGRADACCUM`` stores its returned gradients in
      ``stage.state.unsharded_param_grads``. These tensors become the
      accumulator inputs to later joint graphs.
    - ``FORWARD_BACKWARD`` and ``FULL_FORWARD_BACKWARD`` append the loss to
      ``stage.output_chunks`` and ``schedule._internal_losses`` and increment
      the stage backward counter. For SPMD without gradient accumulation,
      reduced gradients go directly to ``param.grad``. For SPMD with gradient
      accumulation, the runtime stores each graph's returned accumulator
      handles as the inputs to the next graph.
    - ``FORWARD_BACKWARD_FIRST_WITH_UNSHARD`` initializes the gradient
      accumulators and stores its additional unsharded parameter outputs in
      ``stage.state.unsharded_param_values``.
    - ``FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD`` accumulates its raw
      gradients, reduces the complete accumulators, and returns only
      optimizer-visible sharded gradients in
      ``stage.state.sharded_param_grads``.
    - ``FORWARD`` waits for and consumes any remote input receive, materializes
      parameter inputs if needed, records its output in PyTorch's forward-send
      state, saves explicit backward-graph values separately, records
      last-stage losses, and forwards local outputs to the next stage.
    - ``BACKWARD`` and ``BACKWARD_WITH_REDUCE_GRAD`` wait for and consume any
      remote gradient receive, increment the stage backward counter, retire
      both forward state stores, and write input gradients to
      ``stage.bwd_cache[m]`` and, when applicable, the previous local stage.
      The former produces raw, unsharded gradients for a later
      ``REDUCE_GRAD``. The first backward returns the initial accumulators;
      later backwards update them in graph and return the latest handles.
      A single-microbatch backward with inline reduction writes directly to
      ``param.grad``.
    - ``BACKWARD_INPUT`` performs the input-gradient part of that transition
      and saves weight-backward inputs in
      ``stage.saved_values_for_backward_weight_cache[m]``.
      ``BACKWARD_WEIGHT`` pops that entry and applies the same first/repeated
      accumulation contract before deferred reduction.
    - ``OVERLAP_F_B`` applies the same forward and backward transitions to its
      two stages in one multiplexed graph call.
    - ``UNSHARD`` populates ``stage.state.unsharded_param_values``;
      ``RESHARD`` clears it. ``REDUCE_GRAD`` populates
      ``stage.state.sharded_param_grads`` and applies schedule gradient scaling
      once. ``WAIT_REDUCE_GRAD`` is a no-op because the explicit reduction
      graph returns tensors whose dependencies carry collective ordering; it
      does not create PyTorch's eager FSDP reduction handle.

    Gradient accumulation
    ---------------------
    SPMD and PP use the same accumulator lifecycle. The first gradient graph
    returns its gradients as accumulators. Repeated graphs receive those
    tensors, update them in place, and return the latest handles.

    PP selects the first graph by the first executed backward for each stage,
    which need not be microbatch zero. Its final scheduled ``REDUCE_GRAD``
    consumes the accumulated unsharded gradients once.

    For SPMD with gradient accumulation:
    We do not rely on autograd for gradient accumulation.
    Cross-microbatch accumulation is explicit in the joint graphs.
    First microbatch gradients become gradient accumulators that are passed to
    further microbatches.
    Further microbatches accumulate gradients in place in gradient accumulators,
    optionally fusing accumulation in the gradient producer.
    When reduce grad is in the last microbatch, the last microbatch fuses
    gradient accumulation before reduce grad and reduce grad returns sharded
    gradients.

    Calling convention:
    FSDP collectives in every microbatch, or no FSDP::

        FORWARD_BACKWARD_NOGRADACCUM(0) -> loss, first_grads
        stage.state.unsharded_param_grads = first_grads
        FORWARD_BACKWARD(m) -> loss, latest_grads
        stage.state.unsharded_param_grads = latest_grads
        successful schedule exit -> param.grad += final_param_grads
        step cleanup -> stage.state.clear()

    Calling convention:
    FSDP boundaries in edge microbatches::

        FORWARD_BACKWARD_FIRST_WITH_UNSHARD(0)
            -> loss, first_grads, retained_unsharded_params
            -> stage.state.unsharded_param_values = retained_unsharded_params
            -> stage.state.unsharded_param_grads = first_grads
        FORWARD_BACKWARD(1 ... N - 2)
            -> add_ returns latest_grads
            -> stage.state.unsharded_param_grads = latest_grads
        FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD(N - 1)
            -> add_ consumes latest_grads, then reduce-scatter
            -> stage.state.sharded_param_grads = sharded_grads
        RESHARD -> stage.state.unsharded_param_values = []
        successful schedule exit -> param.grad += final_param_grads
        step cleanup -> stage.state.clear()

    When the schedule returns or raises, the runtime clears ``stage.state``,
    its bound-graph lookup, and per-call loss arguments. Completed training
    steps commit the final gradients to ``param.grad`` for the optimizer step.
    PP evaluation performs the same setup and cleanup without committing
    gradients. Both SPMD paths are currently training-only.

    SPMD without gradient accumulation
    ----------------------------------
    Trace the joint graph, optimize it, and run it once as
    ``FULL_FORWARD_BACKWARD``. With FSDP, this graph contains both parameter
    all-gathers and gradient reductions.

    SPMD with gradient accumulation
    -------------------------------
    ``compile.spmd_gradient_accumulation`` configures this path. With FSDP
    collectives in every microbatch, or without FSDP, the schedule is::

        FORWARD_BACKWARD_NOGRADACCUM(stage=0, microbatch=0)
        FULL_FORWARD_BACKWARD(stage=0, microbatch=1)
        ...
        FULL_FORWARD_BACKWARD(stage=0, microbatch=N - 1)

    With first/last-microbatch FSDP boundaries, unsharded parameters are
    reused across all microbatches until the final ``RESHARD``::

        FORWARD_BACKWARD_FIRST_WITH_UNSHARD(stage=0, microbatch=0)
        FORWARD_BACKWARD(stage=0, microbatch=1)
        ...
        FORWARD_BACKWARD_LAST_WITH_REDUCE_GRAD(stage=0, microbatch=N - 1)
        RESHARD(stage=0)

    The first graph receives sharded parameters and returns both unsharded
    values and gradients retained in runtime state for this step. Middle and
    last graphs receive those values. Their WGrad producers update the first
    microbatch gradients directly; the last graph then reduces them.

    PP
    --
    The upstream schedule owns action ordering and communication.
    This is the only path that partitions the joint graph into forward and
    backward graphs. Gradient accumulation follows the first/repeated graph
    contract above independently for each physical or virtual stage.
    Current schedules emit one ``REDUCE_GRAD(s)`` after each stage's final
    backward. ``UNSHARD(s)`` and ``RESHARD(s)`` may run more than once per stage;
    none of these actions is global.

    Local stages exchange forward outputs and input gradients through upstream
    stage caches. The upstream schedule creates and waits for remote P2P
    operations.

    Args:
        stages: Local graph stages. SPMD requires exactly one stage.
        num_microbatches: Trainer accumulation steps for SPMD, or configured
            pipeline microbatches for PP.
        parallelism_context: Parallel topology used to select SPMD or PP.
        config: Full Trainer configuration for SPMD. PP is entered through
            the generic pipelining API and supplies only its compile,
            parallelism, and model fields.
        loss_fn: Loss function used by the schedule and graph provider.
    """
    pp_enabled = parallelism_context.pp_enabled
    if not pp_enabled and len(stages) != 1:
        raise ValueError(f"SPMD requires one local stage, got {len(stages)}")
    plan = resolve_graph_execution_plan(
        config.compile,
        num_microbatches=num_microbatches,
        parallelism=config.parallelism,
        pp_enabled=pp_enabled,
        fsdp_enabled=parallelism_context.fsdp_enabled,
    )

    if pp_enabled:
        return _make_pipeline_parallel_graph_runtime(
            stages,
            plan=plan,
            config=config,
            loss_fn=loss_fn,
            parallelism_context=parallelism_context,
        )

    if isinstance(config, GraphTrainerConfigView):
        raise ValueError("SPMD requires the full Trainer config")
    return _make_spmd_graph_runtime(
        stages[0],
        plan=plan,
        trainer_config=config,
        loss_fn=loss_fn,
        parallelism_context=parallelism_context,
    )


def make_spmd_graph_runtime(
    model: nn.Module,
    *,
    gradient_accumulation_steps: int,
    parallelism_context: ParallelismContext,
    device: torch.device,
    loss_fn: LossFunction,
    trainer_config: "GraphTrainer.Config",
) -> GraphRuntime:
    """Represent one SPMD model as a single-stage graph runtime."""
    # PipelineStage treats `group=None` as the world group.
    # TODO: Remove this when PipelineStage supports local single-stage execution.
    pp_mesh = parallelism_context.get_optional_mesh("pp", include_singleton_axes=True)
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
        parallelism_context=parallelism_context,
        config=trainer_config,
        loss_fn=loss_fn,
    )


def graph_pipeline_llm(
    model: nn.Module,
    *,
    parallelism_context: ParallelismContext,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    device: torch.device,
    model_config: BaseModel.Config,
    loss_fn: LossFunction,
) -> tuple[GraphRuntime, list[BaseModel], bool, bool]:
    """Build a GraphPP pipeline schedule for GraphTrainer.

    Args:
        model: The full model before PP stage splitting.
        parallelism_context: TorchTitan parallel dimension helper.
        training: Training config used for local batch size.
        parallelism: Parallelism config used for PP schedule and module split.
        compile_config: GraphTrainer compile config.
        ac_config: Activation checkpointing config forwarded to the model.
        dump_folder: Artifact/debug output directory.
        device: Local device for the stage.
        model_config: Model config consumed by stage graph passes.
        loss_fn: Loss function used by upstream PP metadata and GraphPP tracing.

    Returns:
        A tuple of ``(runtime, model_parts, has_first_stage, has_last_stage)``.
    """
    pp_mesh = parallelism_context.get_mesh("pp")

    (
        num_virtual_stages,
        num_layers,
        input_weight,
        output_weight,
    ) = _get_pipeline_metadata(parallelism_context, parallelism, model_config)

    module_names_per_stage = parallelism.pipeline_parallel_module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = _generate_llm_fqn_per_model_part(
            num_virtual_stages,
            num_layers,
            input_weight,
            output_weight,
        )
    for index, stage_modules in enumerate(module_names_per_stage):
        logger.debug("GraphPP stage %s modules: %s", index, stage_modules)

    get_mesh_cb = _build_get_mesh_callback(parallelism_context)
    pp_rank_to_stage_indices = _get_pp_rank_to_stage_indices_mapping(
        pp_mesh.get_local_rank(),
        pp_mesh.size(),
        parallelism.pipeline_parallel_schedule,
        len(module_names_per_stage),
    )
    model_parts: list[BaseModel] = []
    stages: list[GraphPipelineStage] = []
    for stage_index in pp_rank_to_stage_indices:
        model_part = _split_module(model, module_names_per_stage[stage_index])
        model_part = model_part.parallelize(
            parallelism_context=parallelism_context,
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
        parallelism_context=parallelism_context,
        config=GraphTrainerConfigView(
            compile=compile_config,
            parallelism=parallelism,
            model=model_config,
        ),
        loss_fn=loss_fn,
    )

    return (
        graph_runtime,
        model_parts,
        any(stage.is_first for stage in stages),
        any(stage.is_last for stage in stages),
    )
