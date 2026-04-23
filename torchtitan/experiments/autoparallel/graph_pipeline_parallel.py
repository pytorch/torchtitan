# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from autoparallel.graph_passes.graph_pp_runner import (
    get_multiplexed_graph_callables,
    GraphCallables,
    GraphMeta,
    GraphPipelineStage,
    GraphPPRunner,
    overlap_fw_bw,
    stage_backward_input,
    stage_backward_weight,
    stage_forward,
    stage_full_backward,
    stage_reduce_grad,
    stage_reshard,
    stage_unshard,
)
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    BACKWARD_INPUT,
    BACKWARD_WEIGHT,
    FORWARD,
    FULL_BACKWARD,
    OVERLAP_F_B,
    REDUCE_GRAD,
    RESHARD,
    ScheduleDualPipeV,
    UNSHARD,
)
from torchtitan.components.loss import LossFunction
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.pipeline_parallel import (
    build_pipeline_schedule,
    generate_llm_fqn_per_model_part,
    get_pipeline_metadata,
    get_pp_rank_to_stage_indices_mapping,
    split_module,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.protocols.model_spec import ParallelizeFunction
from torchtitan.tools.logging import logger


class ModelWithLoss(nn.Module):
    """Wraps a stage model with a loss function so that loss computation
    is included in the compiled graph callables for GraphPP."""

    def __init__(self, model: nn.Module, loss_fn: LossFunction):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        output = self.model(h)
        return self.loss_fn(output, labels)

    def init_weights(self, *args: Any, **kwargs: Any) -> None:
        if hasattr(self.model, "init_weights"):
            self.model.init_weights(*args, **kwargs)


def get_input_generating_fns(
    model_config: BaseModel.Config,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    parallel_dims: ParallelDims,
) -> tuple[Callable, Callable, Callable, Callable]:
    """
    Create tracing input functions for each pipeline stage type.

    Returns:
        Tuple of (first_stage_fn, intermediate_stage_fn, last_stage_fn, target_fn)
    """

    def make_input_fn(
        batch_size: int,
        inp_type: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Callable:
        def input_fn() -> torch.Tensor:
            if inp_type == "tokens":
                return torch.randint(
                    0,
                    model_config.vocab_size,
                    (batch_size, training.seq_len),
                    dtype=dtype,
                    device=device,
                )
            elif inp_type == "embeddings":
                return torch.randn(
                    (batch_size, training.seq_len, model_config.dim),
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            elif inp_type == "logits":
                return torch.randn(
                    (batch_size, training.seq_len, model_config.vocab_size),
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            elif inp_type == "loss":
                return torch.scalar_tensor(
                    1.0,
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                )
            else:
                raise ValueError(f"Unknown input type: {inp_type}")

        return input_fn

    dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
    microbatch_size = parallelism.pipeline_parallel_microbatch_size
    spmd_batch_size = microbatch_size * dp_degree

    device = torch.device("cuda")

    tracing_target_fn = make_input_fn(spmd_batch_size, "tokens", torch.int64, device)
    tracing_input_fn_first_stage = make_input_fn(
        spmd_batch_size, "tokens", torch.int64, device
    )
    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    tracing_input_fn_intermediate_stage = make_input_fn(
        spmd_batch_size, "embeddings", param_dtype, device
    )

    def tracing_input_fn_last_stage():
        return (
            tracing_input_fn_intermediate_stage(),
            tracing_target_fn(),
        )

    return (
        tracing_input_fn_first_stage,
        tracing_input_fn_intermediate_stage,
        tracing_input_fn_last_stage,
        tracing_target_fn,
    )


def get_shape_inference_fns(
    model_config: BaseModel.Config,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    has_loss: bool,
) -> tuple[Callable, Callable, Callable]:
    """
    Create shape inference functions that return meta-device tensors
    with microbatch shapes. Used by GraphPipelineStage constructor for
    input_args / output_args to infer inter-stage activation shapes.

    Returns:
        Tuple of (first_stage_input_fn, intermediate_fn, last_stage_output_fn)
    """
    microbatch_size = parallelism.pipeline_parallel_microbatch_size
    meta_device = torch.device("meta")

    def first_stage_input():
        return torch.randint(
            0,
            model_config.vocab_size,
            (microbatch_size, training.seq_len),
            device=meta_device,
        )

    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]

    def intermediate_stage():
        return torch.randn(
            (microbatch_size, training.seq_len, model_config.dim),
            device=meta_device,
            dtype=param_dtype,
        )

    def last_stage_output():
        if has_loss:
            return torch.scalar_tensor(
                1.0,
                dtype=torch.float32,
                device=meta_device,
            )
        else:
            return torch.randn(
                (microbatch_size, training.seq_len, model_config.vocab_size),
                device=meta_device,
                dtype=param_dtype,
            )

    return first_stage_input, intermediate_stage, last_stage_output


def graph_pipeline_llm(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
    device: torch.device,
    model_config: BaseModel.Config,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    """
    GraphPP-based pipeline parallelism for LLMs.

    Mirrors the structure of pipeline_llm but uses AutoParallelPP for
    per-stage SPMD parallelization, producing compiled graph callables
    that drive execution via GraphPipelineStage and GraphPPRunner.
    """
    pp_mesh = parallel_dims.get_mesh("pp")

    # --- Reused from pipeline_llm ---
    num_virtual_stages, num_layers, input_weight, output_weight = get_pipeline_metadata(
        parallel_dims, parallelism, model_config
    )

    module_names_per_stage = parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )
    for i, stage_ms in enumerate(module_names_per_stage):
        logger.debug(f"Stage {i}: {stage_ms}")

    num_stages = len(module_names_per_stage)

    # --- GraphPP-specific: build SPMD mesh for AutoParallelPP tracing ---
    sparse_names = ["dp_replicate", "efsdp", "ep", "etp"]
    sparse_names = [
        n for n in sparse_names if parallel_dims.get_optional_mesh(n) is not None
    ]
    spmd_mesh = parallel_dims.get_mesh(sparse_names)

    # --- GraphPP-specific: enable inductor + forced balanced routing ---
    use_inductor = getattr(compile_config, "enable", False)
    if use_inductor:
        import autoparallel._testing.models.dsv3 as dsv3_module

        dsv3_module.FORCE_BALANCED_ROUTING = True
        logger.info(
            "Inductor enabled: set FORCE_BALANCED_ROUTING=True for static graph shapes"
        )

    # --- GraphPP-specific: build tracing and shape inference functions ---
    (
        tracing_input_fn_first_stage,
        tracing_input_fn_intermediate_stage,
        tracing_input_fn_last_stage,
        _tracing_target_fn,
    ) = get_input_generating_fns(model_config, training, parallelism, parallel_dims)

    (
        shape_fn_first_stage,
        shape_fn_intermediate_stage,
        shape_fn_last_stage_output,
    ) = get_shape_inference_fns(model_config, training, parallelism, has_loss=True)

    # --- Reused from pipeline_llm: get stage-to-rank mapping ---
    pp_rank = pp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()
    pp_rank_to_stage_indices = get_pp_rank_to_stage_indices_mapping(
        pp_rank, pp_degree, parallelism.pipeline_parallel_schedule, num_stages
    )

    # --- Per-stage loop: split, parallelize, build GraphPipelineStage ---
    stages: list[GraphPipelineStage] = []
    model_parts: list[nn.Module] = []
    stage_graphs: dict[int, GraphCallables] = {}

    for stage_idx in pp_rank_to_stage_indices:
        module_names = module_names_per_stage[stage_idx]
        is_last_stage = stage_idx == num_stages - 1

        # Step 1: Split model — reused from pipeline_parallel.py
        stage_mod = split_module(model, module_names)

        # Step 2: Wrap last stage with loss (GraphPP-specific)
        if is_last_stage:
            stage_mod = ModelWithLoss(stage_mod, loss_fn)

        # Step 2b: Cast model to mixed-precision param dtype (config-driven).
        # The model is created in float32 on meta device; tracing inputs and
        # shape inference use the same dtype derived from training config.
        param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
        stage_mod = stage_mod.to(dtype=param_dtype)

        # Step 3: Pick stage-appropriate tracing input_fn
        if stage_idx == 0:
            input_fn = tracing_input_fn_first_stage
        elif is_last_stage:
            input_fn = tracing_input_fn_last_stage
        else:
            input_fn = tracing_input_fn_intermediate_stage

        # Step 4: Call parallelize_fn with base + extra GraphPP kwargs
        m = parallelize_fn(
            stage_mod,
            parallel_dims=parallel_dims,
            training=training,
            model_converters=model_converters,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
            # Extra GraphPP kwargs
            input_fn=input_fn,
            spmd_mesh=spmd_mesh,
            stage_idx=stage_idx,
            num_stages=num_stages,
            has_loss=is_last_stage,
        )

        # Step 5: Extract graph artifacts from module
        graph_callables = GraphCallables(
            fw=m._graph_callables["fw"],
            full_bw=m._graph_callables["full_bw"],
            bw_dI=m._graph_callables.get("bw_dI"),
            bw_dW=m._graph_callables.get("bw_dW"),
            unshard=m._graph_callables.get("unshard"),
            reduce_grad=m._graph_callables.get("reduce_grad"),
        )
        graph_meta = GraphMeta(
            num_mutate_inputs=m._graph_meta["num_mutate_inputs"],
            num_user_outputs=m._graph_meta["num_user_outputs"],
            num_symints_saved_for_bw=m._graph_meta["num_symints_saved_for_bw"],
            num_params=m._graph_meta["num_params"],
            num_buffers=m._graph_meta["num_buffers"],
            num_input_grads=m._graph_meta["num_input_grads"],
        )

        # Step 6: Build GraphPipelineStage (instead of PipelineStage)
        stage = GraphPipelineStage(
            m,
            graph_callables,
            graph_meta,
            stage_index=stage_idx,
            num_stages=num_stages,
            device=device,
            input_args=(
                shape_fn_first_stage()
                if stage_idx == 0
                else shape_fn_intermediate_stage()
            ),
            output_args=(
                shape_fn_last_stage_output()
                if is_last_stage
                else shape_fn_intermediate_stage()
            ),
            group=pp_mesh.get_group("pp"),
        )

        logger.info(
            f"PP rank {pp_rank} is building GraphPP stage_idx {stage_idx} "
            f"with modules {module_names}"
        )

        stages.append(stage)
        # For model_parts, unwrap ModelWithLoss so the standard
        # register_moe_load_balancing_hook can find `layers` via
        # get_submodule("layers"). CrossEntropyLoss has no parameters,
        # so m.model contains all trainable params.
        if is_last_stage and hasattr(m, "model"):
            model_parts.append(m.model)
        else:
            model_parts.append(m)
        stage_graphs[stage_idx] = graph_callables

    # --- Build pipeline schedule (similar to pipeline_llm, GraphPP-specific params) ---
    pp_schedule = build_pipeline_schedule(
        parallelism=parallelism,
        local_batch_size=training.local_batch_size,
        stages=stages,
        loss_fn=None,
        backward_requires_autograd=False,
    )

    # --- GraphPP-specific: register custom graph functions on the schedule ---
    assert isinstance(pp_schedule, _PipelineScheduleRuntime), (
        f"GraphPP requires a _PipelineScheduleRuntime schedule, "
        f"got {type(pp_schedule).__name__}"
    )
    pp_schedule.register_custom_function(FORWARD, stage_forward)
    pp_schedule.register_custom_function(FULL_BACKWARD, stage_full_backward)
    pp_schedule.register_custom_function(REDUCE_GRAD, stage_reduce_grad)
    pp_schedule.register_custom_function(RESHARD, stage_reshard)
    pp_schedule.register_custom_function(UNSHARD, stage_unshard)
    pp_schedule.register_custom_function(BACKWARD_INPUT, stage_backward_input)
    pp_schedule.register_custom_function(BACKWARD_WEIGHT, stage_backward_weight)

    if isinstance(pp_schedule, ScheduleDualPipeV):
        from autoparallel.graph_passes.graph_multiplex import multiplex_fw_bw_graph

        multiplexed_graph_callables = get_multiplexed_graph_callables(
            stage_graphs,
            partial(multiplex_fw_bw_graph, overlap_with_annotations=True),
        )
        pp_schedule.register_custom_function(
            OVERLAP_F_B, partial(overlap_fw_bw, multiplexed_graph_callables)
        )

    # --- GraphPP-specific: wrap schedule with GraphPPRunner ---
    runner = GraphPPRunner(pp_schedule, inductor=use_inductor)

    # --- Same as pipeline_llm: determine first/last stage flags ---
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return runner, model_parts, has_first_stage, has_last_stage
