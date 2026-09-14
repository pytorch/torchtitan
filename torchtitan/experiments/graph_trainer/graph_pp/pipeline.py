# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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


def _validate_graph_pipeline_compile_config(
    compile_config: GraphTrainerCompileConfig,
) -> None:
    if compile_config.mode != "aot_fx_trace":
        raise ValueError("GraphPipelineRuntime requires --compile.mode aot_fx_trace")
    if compile_config.precompile_artifact_dir:
        raise ValueError(
            "GraphPipelineRuntime does not support "
            "--compile.precompile_artifact_dir yet. Existing precompiled "
            "artifacts contain one monolithic train-step graph, while the "
            "runtime requires separately bound forward, backward, and FSDP graphs."
        )
    if compile_config.ep_overlap.enabled:
        raise ValueError(
            "GraphPipelineRuntime does not support --compile.ep_overlap.enabled "
            "yet. GraphPP stage tracing does not apply the EP-overlap trace-input "
            "preparers."
        )
    if compile_config.memory_policy == "sac_and_offload":
        raise ValueError(
            "GraphPipelineRuntime does not support "
            "--compile.memory_policy sac_and_offload yet. The GraphPP partition "
            "must preserve offload and reload pairs across the forward/backward "
            "boundary."
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


def graph_train_step_runtime(
    model: nn.Module,
    *,
    num_microbatches: int,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    device: torch.device,
    model_config: BaseModel.Config | None,
    loss_fn: LossFunction,
    use_cuda_graph: bool = False,
) -> GraphPipelineRuntime:
    """Build the non-PP graph path as a one-stage pipeline runtime."""
    _validate_graph_pipeline_compile_config(compile_config)
    if num_microbatches < 1:
        raise ValueError(
            "GraphPipelineRuntime requires at least one microbatch, got "
            f"{num_microbatches}"
        )
    defer_fsdp_gradient_sync = compile_config.enable_deferred_fsdp_gradient_sync
    if defer_fsdp_gradient_sync and not parallel_dims.fsdp_enabled:
        raise ValueError("Deferred FSDP gradient synchronization requires FSDP")
    if defer_fsdp_gradient_sync and num_microbatches < 2:
        raise ValueError(
            "Deferred FSDP gradient synchronization requires at least two "
            "microbatches"
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
    reuse_unsharded_parameters = not get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        pp_enabled=False,
    )
    actions = []
    if reuse_unsharded_parameters:
        actions.append(_Action(0, UNSHARD))
    for microbatch_index in range(num_microbatches):
        if not reuse_unsharded_parameters:
            actions.append(_Action(0, UNSHARD))
        actions.extend(
            (
                _Action(0, FORWARD, microbatch_index),
                _Action(0, FULL_BACKWARD, microbatch_index),
            )
        )
        if not reuse_unsharded_parameters:
            actions.append(_Action(0, RESHARD))
    if defer_fsdp_gradient_sync:
        actions.append(_Action(0, REDUCE_GRAD))
    if reuse_unsharded_parameters:
        actions.append(_Action(0, RESHARD))
    schedule._prepare_schedule_with_comms({0: actions}, format="compute_comms")

    graph_provider = GraphTrainerStageGraphProvider(
        loss_fn=loss_fn,
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
        # The trainer-side GraphPipelineStepRunner owns the CUDA graph policy,
        # including the decision to leave capture disabled.
        cudagraph_managed_by_runtime=True,
        extract_fsdp_grad_reduction=defer_fsdp_gradient_sync,
    )
    return register_graph_pp_schedule(
        schedule,
        graph_provider=graph_provider,
        clone_grads_to_initialize_param_grad=use_cuda_graph,
    )


def _validate_graph_pp_config(
    *,
    compile_config: GraphTrainerCompileConfig,
    parallelism: ParallelismConfig,
) -> None:
    _validate_graph_pipeline_compile_config(compile_config)
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
    )
    graph_pipeline_runtime = register_graph_pp_schedule(
        schedule,
        graph_provider=graph_provider,
    )

    has_first_stage = any(stage.is_first for stage in stages)
    has_last_stage = any(stage.is_last for stage in stages)
    return graph_pipeline_runtime, model_parts, has_first_stage, has_last_stage
