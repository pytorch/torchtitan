# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineScheduleRuntime,
    FORWARD,
    FULL_BACKWARD,
    get_schedule_class,
    REDUCE_GRAD,
    RESHARD,
    UNSHARD,
)

from torchtitan.components.loss import BaseLoss, LossFunction
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
    GraphPipelineRuntime,
    register_graph_pp_schedule,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import GraphPipelineStage
from torchtitan.experiments.graph_trainer.registry import (
    PASS_PIPELINE_REGISTRY,
    TRACE_CALL_INPUT_PREPARERS,
    TRACE_INPUT_PREPARERS,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ParallelizeFunction
from torchtitan.tools.logging import logger


@dataclass(frozen=True)
class GraphPPRuntimePolicy:
    extract_fsdp_param_unshard: bool
    extract_fsdp_grad_reduction: bool
    accumulate_in_schedule: bool


def resolve_graph_pp_runtime_policy(
    compile_config: GraphTrainerCompileConfig,
    *,
    pp_enabled: bool,
    fsdp_enabled: bool,
) -> GraphPPRuntimePolicy:
    """Resolve topology-dependent GraphPP execution modes."""
    modes = {
        "fsdp_param_unshard_mode": compile_config.fsdp_param_unshard_mode,
        "fsdp_gradient_sync_mode": compile_config.fsdp_gradient_sync_mode,
        "gradient_accumulation_mode": compile_config.gradient_accumulation_mode,
    }
    if compile_config.mode != "aot_fx_trace":
        explicit_modes = [name for name, value in modes.items() if value != "auto"]
        if explicit_modes:
            raise ValueError(
                "GraphPP execution modes require --compile.mode aot_fx_trace: "
                f"{explicit_modes}"
            )
        return GraphPPRuntimePolicy(False, False, False)

    extract_fsdp_param_unshard = (
        pp_enabled
        if compile_config.fsdp_param_unshard_mode == "auto"
        else compile_config.fsdp_param_unshard_mode == "scheduled"
    )
    extract_fsdp_grad_reduction = (
        pp_enabled
        if compile_config.fsdp_gradient_sync_mode == "auto"
        else compile_config.fsdp_gradient_sync_mode == "scheduled"
    )
    accumulate_in_schedule = (
        not pp_enabled
        if compile_config.gradient_accumulation_mode == "auto"
        else compile_config.gradient_accumulation_mode == "scheduled"
    )

    if not fsdp_enabled:
        if compile_config.fsdp_param_unshard_mode == "scheduled":
            raise ValueError("Scheduled FSDP parameter unsharding requires FSDP")
        if compile_config.fsdp_gradient_sync_mode == "scheduled":
            raise ValueError("Scheduled FSDP gradient synchronization requires FSDP")
        extract_fsdp_param_unshard = False
        extract_fsdp_grad_reduction = False
    elif pp_enabled:
        if not extract_fsdp_param_unshard:
            raise ValueError(
                "PP>1 GraphPP requires scheduled FSDP parameter unsharding"
            )
        if not extract_fsdp_grad_reduction:
            raise ValueError("PP>1 GraphPP requires scheduled FSDP gradient sync")

    if pp_enabled and accumulate_in_schedule:
        raise ValueError(
            "PP>1 GraphPP does not yet support folding Trainer gradient "
            "accumulation into the pipeline schedule"
        )

    return GraphPPRuntimePolicy(
        extract_fsdp_param_unshard=extract_fsdp_param_unshard,
        extract_fsdp_grad_reduction=extract_fsdp_grad_reduction,
        accumulate_in_schedule=accumulate_in_schedule,
    )


def _validate_pp1_vpp1_graph_pipeline_compile_config(
    compile_config: GraphTrainerCompileConfig,
) -> None:
    if compile_config.mode != "aot_fx_trace":
        raise ValueError("GraphPipelineRuntime requires --compile.mode aot_fx_trace")
    if compile_config.ep_overlap.enabled:
        raise ValueError(
            "GraphPipelineRuntime does not support --compile.ep_overlap.enabled "
            "yet. GraphPP stage tracing does not apply the EP-overlap trace-input "
            "preparers."
        )
    if compile_config.pass_pipeline in PASS_PIPELINE_REGISTRY:
        raise ValueError(
            "GraphPipelineRuntime does not support custom pass pipelines yet"
        )
    trace_preparer_names = set(trace_input_preparer_keys(compile_config))
    unsupported_preparers = trace_preparer_names.intersection(
        TRACE_INPUT_PREPARERS.keys() | TRACE_CALL_INPUT_PREPARERS.keys()
    )
    if unsupported_preparers:
        raise ValueError(
            "GraphPipelineRuntime does not support trace-input preparers yet: "
            f"{sorted(unsupported_preparers)}"
        )


def make_pp1_vpp1_graph_pipeline_runtime(
    model: nn.Module,
    *,
    num_microbatches: int,
    training: TrainingConfig,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    device: torch.device,
    model_config: BaseModel.Config | None,
    loss_fn: LossFunction,
    loss_config: BaseLoss.Config,
) -> GraphPipelineRuntime:
    """Build the PP=1/VPP=1 runtime to reuse GraphPipelineRuntime,
    to express Gradient Accumulation and Deferred FSDP gradient sync.

    The pipeline runtime represents gradient accumulation as N schedule
    microbatches. With FSDP, ``FULL_BACKWARD`` either keeps its reduction or
    has it extracted according to ``fsdp_gradient_sync_mode``:

    ``fsdp_gradient_sync_mode="in_graph"``::

        FWD(0)
        -> FULL_BACKWARD(0, with FSDP reduce)
        -> ...
        -> FWD(N-1)
        -> FULL_BACKWARD(N - 1, with FSDP reduce)

    ``fsdp_gradient_sync_mode="scheduled"``::

        FWD(0)
        -> FULL_BACKWARD(0, without FSDP reduce)
        -> ...
        -> FWD(N-1)
        -> FULL_BACKWARD(N-1, without FSDP reduce)
        -> REDUCE_GRAD

    FSDP parameter all-gathers remain in each ``FWD`` graph or are represented
    by ``UNSHARD`` actions according to ``fsdp_param_unshard_mode``.
    """
    _validate_pp1_vpp1_graph_pipeline_compile_config(compile_config)
    if num_microbatches < 1:
        raise ValueError(
            "GraphPipelineRuntime requires at least one microbatch, got "
            f"{num_microbatches}"
        )
    runtime_policy = resolve_graph_pp_runtime_policy(
        compile_config,
        pp_enabled=False,
        fsdp_enabled=parallel_dims.fsdp_enabled,
    )
    extract_fsdp_param_unshard = runtime_policy.extract_fsdp_param_unshard
    extract_fsdp_grad_reduction = runtime_policy.extract_fsdp_grad_reduction
    fsdp_reshard_after_forward = (
        get_fsdp_reshard_after_forward_policy(
            parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
        )
        if parallel_dims.fsdp_enabled
        else None
    )

    pp_mesh = parallel_dims.get_optional_mesh("pp", include_singleton_axes=True)
    assert pp_mesh is not None
    stage = GraphPipelineStage(
        model,
        stage_index=0,
        num_stages=1,
        device=device,
        group=pp_mesh.get_group("pp"),
    )

    def scalar_loss_fn(*args: object, **kwargs: object) -> torch.Tensor:
        loss = loss_fn(*args, **kwargs)
        return loss[0] if isinstance(loss, tuple) else loss

    schedule = _PipelineScheduleRuntime(
        [stage],
        n_microbatches=num_microbatches,
        loss_fn=scalar_loss_fn,
        scale_grads=False,
        backward_requires_autograd=False,
    )
    reuse_unsharded_parameters = (
        extract_fsdp_param_unshard and fsdp_reshard_after_forward is False
    )
    actions = []
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

    graph_provider = GraphTrainerStageGraphProvider(
        loss_fn=loss_fn,
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
        extract_fsdp_param_unshard=extract_fsdp_param_unshard,
        extract_fsdp_grad_reduction=extract_fsdp_grad_reduction,
    )
    if compile_config.precompile_artifact_dir:
        from torchtitan.experiments.graph_trainer.make_fx_tracer import (
            extract_module_state,
        )
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
            get_precompile_runtime_meshes,
            precompile_graph_pp_stage_load,
        )
        from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter

        runtime_meshes = get_precompile_runtime_meshes(parallel_dims)
        graph_provider.precompiled_stage_graphs = precompile_graph_pp_stage_load(
            DiskStorageAdapter(compile_config.precompile_artifact_dir),
            expected_fingerprint=compute_config_fingerprint(
                model,
                compile_config,
                parallel_dims,
                loss_config=loss_config,
                model_config=model_config,
                parallelism_config=parallelism,
                training_config=training,
            ),
            expected_state_fqns=list(extract_module_state(model)),
            runtime_meshes=runtime_meshes,
        )
    return register_graph_pp_schedule(
        schedule,
        graph_provider=graph_provider,
    )


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
) -> tuple[GraphPipelineRuntime, list[nn.Module], bool, bool]:
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
    _validate_graph_pp_config(
        compile_config=compile_config,
        parallelism=parallelism,
    )
    runtime_policy = resolve_graph_pp_runtime_policy(
        compile_config,
        pp_enabled=True,
        fsdp_enabled=parallel_dims.fsdp_enabled,
    )
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

    schedule = _build_pipeline_schedule(
        parallelism=parallelism,
        num_microbatches=parallelism.num_pp_microbatches,
        stages=stages,
        loss_fn=loss_fn,
        backward_requires_autograd=False,
    )
    graph_provider = GraphTrainerStageGraphProvider(
        loss_fn=loss_fn,
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
        extract_fsdp_param_unshard=runtime_policy.extract_fsdp_param_unshard,
        extract_fsdp_grad_reduction=runtime_policy.extract_fsdp_grad_reduction,
    )
    graph_provider._warn_if_cudagraph_pass_requested()
    graph_pipeline_runtime = register_graph_pp_schedule(
        schedule,
        graph_provider=graph_provider,
    )

    return (
        graph_pipeline_runtime,
        model_parts,
        any(stage.is_first for stage in stages),
        any(stage.is_last for stage in stages),
    )
