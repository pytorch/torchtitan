# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy

import math
import os
from typing import Callable

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    get_schedule_class,
    OVERLAP_F_B,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
)

from torchtitan.components.loss import LossFunction, rescale_accumulated_loss
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.dual_pipe_v import overlap_callback
from torchtitan.protocols.train_spec import BaseModelArgs, ParallelizeFunction
from torchtitan.tools.logging import logger

__all__ = [
    "pipeline_llm",
    "build_pipeline_schedule",
    "generate_llm_fqn_per_model_part",
    "pipeline_module_split",
]

lib = torch.library.Library("aten", "IMPL")


def _override_torch_ops_for_zero_bubble():
    class MmSeparateWeightGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, i, w, real_output):
            ctx.save_for_backward(i)
            return real_output

        @staticmethod
        def backward(ctx, grad_output):
            (i,) = ctx.saved_tensors
            grad_weight = i.t().mm(grad_output)
            return None, grad_weight, None

    class MmSeparateInputGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, i, w, real_output):
            ctx.save_for_backward(w)
            return real_output

        @staticmethod
        def backward(ctx, grad_output):
            (w,) = ctx.saved_tensors
            grad_input = grad_output.mm(w.t())
            return grad_input, None, None

    class MmPassThrough(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mm_output, fake_1, fake_2):
            # we computed the mm earlier, so we could reuse its output shape in the separate input/weight functions
            # but we need to keep this autograd function to connect the fake_* inputs to the autograd graph and pass
            # gradients back to them
            return mm_output

        @staticmethod
        def backward(ctx, gO):
            return None, gO, gO

    def split_mm(i, w):
        # Apply the pass-through node. y is passed to this node so that it can be
        # saved for backward, but detach because we don't want to actually build
        # this edge of the graph
        with torch._C._AutoDispatchBelowAutograd():
            real_output = torch.mm(i.detach(), w.detach()).detach()

        fake_1 = MmSeparateWeightGrad.apply(i.detach(), w, real_output)
        fake_2 = MmSeparateInputGrad.apply(i, w.detach(), real_output)

        return MmPassThrough.apply(real_output, fake_1, fake_2)

    # addmm operator: out = beta * input + alpha * (mat1 @ mat2)
    class AddmmSeparateMat2Grad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mat1, mat2, alpha):
            ctx.save_for_backward(mat1)
            ctx.alpha = alpha
            return mat2

        @staticmethod
        def backward(ctx, grad_output):
            (mat1,) = ctx.saved_tensors
            # Gradient w.r.t. mat2: alpha * mat1.T @ grad_output
            grad_mat2 = mat1.t().mm(grad_output) * ctx.alpha
            return None, grad_mat2, None

    class AddmmSeparateMat1Grad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mat1, mat2, alpha):
            ctx.save_for_backward(mat2)
            ctx.alpha = alpha
            return mat1

        @staticmethod
        def backward(ctx, grad_output):
            (mat2,) = ctx.saved_tensors
            # Gradient w.r.t. mat1: alpha * grad_output @ mat2.T
            grad_mat1 = grad_output.mm(mat2.t()) * ctx.alpha
            return grad_mat1, None, None

    class AddmmSeparateBiasGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, bias, beta):
            ctx.beta = beta
            return bias

        @staticmethod
        def backward(ctx, grad_output):
            # Gradient w.r.t. bias: beta * sum(grad_output, dim=0)
            grad_bias = grad_output.sum(dim=0) * ctx.beta
            return grad_bias, None

    class AddmmPassThrough(torch.autograd.Function):
        @staticmethod
        def forward(ctx, bias, mat1, mat2, beta, alpha):
            with torch._C._AutoDispatchBelowAutograd():
                return torch.addmm(bias, mat1, mat2, beta=beta, alpha=alpha)

        @staticmethod
        def backward(ctx, gO):
            return gO, gO, gO, None, None

    def split_addmm(bias, mat1, mat2, *, beta=1, alpha=1):
        mat2_1 = AddmmSeparateMat2Grad.apply(mat1.detach(), mat2, alpha)
        mat1_1 = AddmmSeparateMat1Grad.apply(mat1, mat2.detach(), alpha)
        bias_1 = AddmmSeparateBiasGrad.apply(bias, beta)
        return AddmmPassThrough.apply(bias_1, mat1_1, mat2_1, beta, alpha)

    # rms_norm operator: RMS normalization
    class RmsNormSeparateWeightGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, normalized_shape, weight, eps, real_output, rstd):
            ctx.save_for_backward(input, rstd)
            return real_output

        @staticmethod
        def backward(ctx, grad_output):
            (input, rstd) = ctx.saved_tensors
            normalized = input * rstd
            # Gradient w.r.t. weight: sum over batch dimension
            grad_weight = (grad_output * normalized).sum(
                dim=tuple(range(grad_output.ndim - 1))
            )
            return None, None, grad_weight, None, None

    class RmsNormSeparateInputGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, normalized_shape, weight, eps, real_output, rstd):
            ctx.save_for_backward(weight, rstd)
            return real_output

        @staticmethod
        def backward(ctx, grad_output):
            weight, rstd = ctx.saved_tensors

            # Gradient w.r.t. input
            if weight is not None:
                grad_input_unnorm = grad_output * weight
            else:
                grad_input_unnorm = grad_output

            mean = (grad_input_unnorm * input).mean(-1, keepdim=True)
            grad_input = (grad_input_unnorm - input * mean * rstd.pow(2)) * rstd

            return grad_input, None, None, None, None

    class RmsNormPassThrough(torch.autograd.Function):
        @staticmethod
        def forward(ctx, real_output, fake_1, fake_2):
            return real_output

        @staticmethod
        def backward(ctx, gO):
            # Pass gradients to fake_1 and fake_2 to trigger their backward methods
            # Return None for real_output since it's already detached
            return None, gO, gO

    def split_rms_norm(input, normalized_shape, weight=None, eps=None):
        # Compute the actual output first
        with torch._C._AutoDispatchBelowAutograd():
            real_output = torch.rms_norm(
                input.detach(),
                normalized_shape,
                weight.detach() if weight is not None else None,
                eps,
            ).detach()
            variance = input.pow(2).mean(-1, keepdim=True)
            rstd = torch.rsqrt(variance + eps).detach()
            rstd2 = rstd.clone().detach()

        weight_1 = RmsNormSeparateWeightGrad.apply(
            input.detach(), normalized_shape, weight, eps, real_output, rstd
        )
        input_1 = RmsNormSeparateInputGrad.apply(
            input,
            normalized_shape,
            weight.detach() if weight is not None else None,
            eps,
            real_output,
            rstd2
        )
        return RmsNormPassThrough.apply(real_output, weight_1, input_1)

    # _grouped_mm operator: Grouped matrix multiplication for MoE
    class GroupedMmSeparateMat2Grad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, mat2, offs, bias, out_dtype, real_output):
            ctx.save_for_backward(input)
            ctx.offs = offs
            return real_output

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            # Gradient w.r.t. mat2 for grouped mm
            grad_mat2 = torch.ops.aten._grouped_mm.default(
                input.transpose(-1, -2), grad_output, offs=ctx.offs
            )
            return None, grad_mat2, None, None, None, None

    class GroupedMmSeparateInputGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, mat2, offs, bias, out_dtype, real_output):
            ctx.save_for_backward(mat2)
            ctx.offs = offs
            return real_output

        @staticmethod
        def backward(ctx, grad_output):
            (mat2,) = ctx.saved_tensors
            # Gradient w.r.t. input for grouped mm
            grad_input = torch.ops.aten._grouped_mm.default(
                grad_output, mat2.transpose(-1, -2), offs=ctx.offs
            )
            return grad_input, None, None, None, None, None

    class GroupedMmPassThrough(torch.autograd.Function):
        @staticmethod
        def forward(ctx, real_output, fake_1, fake_2):
            return real_output

        @staticmethod
        def backward(ctx, gO):
            return None, gO, gO

    def split_grouped_mm(input, mat2, offs=None, bias=None, out_dtype=None):
        with torch._C._AutoDispatchBelowAutograd():
            real_output = torch.ops.aten._grouped_mm.default(
                input, mat2, offs=offs, bias=bias, out_dtype=out_dtype
            ).detach()
        fake_1 = GroupedMmSeparateMat2Grad.apply(
            input.detach(), mat2, offs, bias, out_dtype, real_output
        )
        fake_2 = GroupedMmSeparateInputGrad.apply(
            input, mat2.detach(), offs, bias, out_dtype, real_output
        )
        return GroupedMmPassThrough.apply(real_output, fake_1, fake_2)

    lib.impl("mm", split_mm, "Autograd")
    lib.impl("addmm", split_addmm, "Autograd")
    lib.impl("rms_norm", split_rms_norm, "Autograd")
    lib.impl("_grouped_mm", split_grouped_mm, "Autograd")
    torch.autograd.set_detect_anomaly(True, check_nan=False)

def pipeline_llm(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: torch.device,
    model_args: BaseModelArgs,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    pp_mesh = parallel_dims.get_mesh("pp")

    if True:
        _override_torch_ops_for_zero_bubble()

    # Determine the number of virtual stages based on schedule type
    schedule_class = get_schedule_class(
        job_config.parallelism.pipeline_parallel_schedule
    )
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    layers_per_stage = job_config.parallelism.pipeline_parallel_layers_per_stage
    if hasattr(model_args, "n_layers"):
        num_layers = model_args.n_layers
    else:
        raise ValueError("Model does not have n_layers attribute.")

    # You can adjust these weights based on the computational cost of embeddings and output layers
    # Higher weights mean these modules are treated as "heavier" in the distribution
    input_weight = job_config.parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = job_config.parallelism.pipeline_parallel_last_stage_less_layers

    # Calculate number of virtual stages
    if layers_per_stage is not None:

        # Calculate number of virtual stages needed (using ceiling division)
        # This allows for unequal distribution where stages can differ by at most 1 layer
        num_virtual_stages = math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )

        # Validation: check stages per rank based on schedule type
        model_config_info = f"Model has {num_layers} layers with pipeline_parallel_layers_per_stage={layers_per_stage}"
        stage_distribution_info = (
            f"resulting in {num_virtual_stages=} across {parallel_dims.pp} PP ranks"
        )

        if num_virtual_stages % parallel_dims.pp != 0:
            raise ValueError(
                f"Number of virtual stages ({num_virtual_stages}) must be divisible by "
                f"pipeline parallel size ({parallel_dims.pp}). "
                f"{model_config_info}. "
                f"Please adjust pipeline_parallel_layers_per_stage to a value that results in a number of stages "
                f"divisible by {parallel_dims.pp}."
            )

        stages_per_rank = num_virtual_stages // parallel_dims.pp

        if is_single_stage_schedule and stages_per_rank != 1:
            raise ValueError(
                f"Single stage schedule requires exactly 1 stage per rank, but got {stages_per_rank} stages per rank. "
                f"{model_config_info}, {stage_distribution_info}. "
                f"Please increase pipeline_parallel_layers_per_stage to {num_layers // parallel_dims.pp} or higher "
                f"to achieve 1 stage per rank."
            )

        if not is_single_stage_schedule and stages_per_rank < 2:
            raise ValueError(
                f"Multi-stage schedule requires at least 2 stages per rank, but got {stages_per_rank} stages per rank. "
                f"{model_config_info}, {stage_distribution_info}. "
                f"Please decrease pipeline_parallel_layers_per_stage to achieve at least 2 stages per rank."
            )
    else:
        # Fallback to default behavior when layers_per_stage is not provided
        # For multi-stage schedules, default is 2 virtual stages per rank
        # For single-stage schedules, default is 1 virtual stage per rank
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = parallel_dims.pp * stages_per_rank

    module_names_per_stage = job_config.parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )
    for i, stage_ms in enumerate(module_names_per_stage):
        logger.debug(f"Stage {i}: {stage_ms}")

    stages, model_parts = pipeline_module_split(
        model,
        pp_mesh,
        job_config.parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
    )

    # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
    # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
    # optimizer, and checkpointing
    for i, m in enumerate(model_parts):
        # apply SPMD-style PT-D techniques
        m = parallelize_fn(m, parallel_dims, job_config)
        model_parts[i] = m
        # NOTE: this is to update the model in the stage
        #       in case the model is modified e.g. by torch.compile
        stages[i].submod = m

    pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)

    # This is used in the train loop to determine whether to pass in the input_ids and labels
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_parts, has_first_stage, has_last_stage


def build_pipeline_schedule(
    job_config: JobConfig, stages: list[PipelineStage], loss_fn: Callable
) -> _PipelineSchedule:
    """Builds a pipeline schedule for the given job configuration and stages.

    Args:
        job_config (JobConfig): The job configuration.
        stages (list[PipelineStage]): The stages to be scheduled.
        loss_fn (Callable): The loss function.

    Returns:
        _PipelineSchedule: The pipeline schedule for the given stages.
    """
    pp_schedule_csv = job_config.parallelism.pipeline_parallel_schedule_csv

    # Validate that pp_schedule_csv is a valid path
    if pp_schedule_csv:
        if not os.path.isfile(pp_schedule_csv):
            raise FileNotFoundError(
                f"The specified path {pp_schedule_csv} does not exist or is not a file."
            )
        schedule_class = _PipelineScheduleRuntime
    else:
        schedule_class = get_schedule_class(
            job_config.parallelism.pipeline_parallel_schedule
        )

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    microbatch_size = job_config.parallelism.pipeline_parallel_microbatch_size
    batch_size = job_config.training.local_batch_size
    # validate that the batch size is divisible by the microbatch_size otherwise we'll hang or error during training
    if batch_size % microbatch_size != 0:
        raise ValueError(
            f"Batch size {job_config.training.local_batch_size} must be divisible by microbatch_size {microbatch_size}. "
            "Update the config arguments for either batch_size or pipeline_parallel_microbatch_size."
        )
    n_microbatches = batch_size // microbatch_size
    # We expect that the number of local stages (`len(stages)`) is the same across all ranks
    num_total_stages = job_config.parallelism.pipeline_parallel_degree * len(stages)
    if n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number "
            f"of stages ({num_total_stages}) which may result in a bubble in the pipeline."
        )

    # pyrefly: ignore [bad-instantiation]
    schedule = schedule_class(
        # pyrefly: ignore [bad-argument-type]
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=rescale_accumulated_loss(loss_fn, n_microbatches),
        scale_grads=False,
    )
    logger.info(
        f"Using pipeline schedule {job_config.parallelism.pipeline_parallel_schedule} "
        f"with {n_microbatches} microbatches and {num_total_stages} stages."
    )

    if job_config.parallelism.pipeline_parallel_expert_parallel_overlap and isinstance(
        schedule, ScheduleDualPipeV
    ):
        schedule.register_custom_function(OVERLAP_F_B, overlap_callback)

    if pp_schedule_csv:
        assert schedule_class in [
            PipelineScheduleSingle,
            PipelineScheduleMulti,
            _PipelineScheduleRuntime,
        ], (
            "Only PipelineScheduleSingle (single stage), PipelineScheduleMulti (multistage), "
            "and _PipelineScheduleRuntime support csv schedules"
        )
        # pyrefly: ignore [missing-attribute]
        schedule._load_csv(pp_schedule_csv)

    return schedule


def generate_llm_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[list[str]]:
    """
    Programmatically generates module names model part, focused on LLMs models.

    Args:
        num_stages: Number of pipeline stages
        num_layers: Total number of transformer layers in the model
        input_weight: Weight for input modules (tok_embeddings) in layer calculation
        output_weight: Weight for output modules (norm + output) in layer calculation

    Returns:
        List of lists containing module names for each model part

    Example:
        generate_llm_fqn_per_model_part(2, 3, input_weight=2, output_weight=2)
        treats embeddings as 2 layers and norm+output as 2 layers for distribution
    """
    if num_stages < 1:
        raise ValueError("Number of stages must be at least 1")

    if num_stages == 1:
        # Single stage gets everything
        layer_names = [f"layers.{i}" for i in range(num_layers)]
        return [["tok_embeddings"] + layer_names + ["norm", "output"]]

    # Calculate effective layers including weights
    num_effective_layers = num_layers + input_weight + output_weight

    if num_stages > num_effective_layers:
        raise ValueError(
            f"Number of stages ({num_stages}) cannot be greater than effective layers ({num_effective_layers})"
        )

    # Calculate layers per stage (distribute evenly)
    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    # Feasibility check: Ensure at least 1 layer in each PP stage
    if layers_per_stage == 0:
        raise ValueError(
            f"Configuration would result in empty stages. "
            f"With {num_stages} stages and {num_effective_layers} effective layers "
            f"(num_layers={num_layers} + input_weight={input_weight} + output_weight={output_weight}), "
            f"each stage would get {layers_per_stage} layers on average. "
            f"Reduce num_stages or increase num_layers/weights."
        )

    # Balance check: Ensure weights don't exceed minimum layers per stage
    if input_weight > layers_per_stage:
        raise ValueError(
            f"input_weight ({input_weight}) exceeds minimum layers per stage ({layers_per_stage})."
        )
    if output_weight > layers_per_stage:
        raise ValueError(
            f"output_weight ({output_weight}) exceeds minimum layers per stage ({layers_per_stage})."
        )

    module_names_per_stage = []
    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules = []

        # Calculate effective layers for this stage
        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:
            effective_layers_for_stage += 1

        # First stage: handle input modules with weighting
        if stage_idx == 0:
            stage_modules.append("tok_embeddings")
            # Account for input weight in layer distribution
            remaining_layers_for_stage = effective_layers_for_stage - input_weight

            # Add transformer layers
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

        # Last stage: handle output modules with weighting
        elif stage_idx == num_stages - 1:
            # Account for output weight in layer distribution
            remaining_layers_for_stage = effective_layers_for_stage - output_weight

            # Add transformer layers
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

            # Add output modules
            stage_modules.extend(["norm", "output"])

        # Middle stages: only transformer layers
        else:
            for _ in range(effective_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def pipeline_module_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    pp_schedule: str,
    device: torch.device,
    module_names_per_stage: list[list[str]],
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """
    This API creates pipeline stages based on specified module names for each stage.

    Some model restrictions include:
    - forward() method should tolerate deleted layers
    - weight initialization methods should tolerate deleted layers
    - Does not support nested moduledict and modulelist structures

    Args:
        whole_model: The complete model to be split
        pp_mesh: Pipeline parallel device mesh
        pp_schedule: Name of pipeline parallelism schedule
        device: Device
        module_names_per_stage: List of lists, where each inner list contains the module names
                               that should be included in that stage. Module names should be
                               dot-separated paths. Examples:
                               - "tok_embeddings" for token embeddings
                               - "layers.0", "layers.1" for specific transformer layers
                               - "norm" for the final normalization layer
                               - "output" for the output projection layer

    Returns:
        Tuple of (stages, models) where stages are PipelineStage objects and models are the
        corresponding model chunks

    Example usage:
        module_names_per_stage = [
            ["tok_embeddings", "layers.0"],     # Stage 0: embeddings + first layer
            ["layers.1", "layers.2"],           # Stage 1: middle layers
            ["norm", "output"]                  # Stage 2: final norm + output
        ]
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()

    def _build_stage_from_modules(
        stage_idx: int, module_names: list[str], num_stages: int
    ) -> tuple[PipelineStage, nn.Module]:
        model = copy.deepcopy(whole_model)

        # Create a set of modules to keep for faster lookup
        modules_to_keep = set(module_names)
        for module_name, module_value in model.named_children():
            # Handle layer-like structures (e.g., "layers.0", "layers.1")
            if isinstance(module_value, (nn.ModuleDict, nn.ModuleList)):
                layers_to_keep = {
                    name.split(".", 1)[1]
                    for name in modules_to_keep
                    if name.startswith(f"{module_name}.")
                }
                if layers_to_keep:
                    # Keep only specified layers
                    if isinstance(module_value, nn.ModuleDict):
                        for layer_name in list(module_value.keys()):
                            if layer_name not in layers_to_keep:
                                del module_value[layer_name]
                    elif isinstance(module_value, nn.ModuleList):
                        indices_to_keep = {
                            int(idx) for idx in layers_to_keep if idx.isdigit()
                        }
                        new_layers = nn.ModuleList(
                            [
                                layer
                                for i, layer in enumerate(module_value)
                                if i in indices_to_keep
                            ]
                        )
                        setattr(model, module_name, new_layers)
                else:
                    # No layers from this structure needed, set to empty structure
                    if isinstance(module_value, nn.ModuleDict):
                        setattr(model, module_name, nn.ModuleDict())
                    elif isinstance(module_value, nn.ModuleList):
                        setattr(model, module_name, nn.ModuleList())
            # Handle simple module attributes (e.g., "linear", "norm")
            elif module_name not in modules_to_keep:
                # Replace with None
                setattr(model, module_name, None)

        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group("pp"),
        )
        return stage, model

    num_stages = len(module_names_per_stage)
    stages = []
    models = []

    schedule_class = get_schedule_class(pp_schedule)
    style = (
        "v" if schedule_class in (ScheduleZBVZeroBubble, ScheduleDualPipeV) else "loop"
    )

    def _get_stage_indices() -> tuple[int, ...]:
        """
        Compute the stage ids for the stages that will run on this pp rank
        for either a looped or V style schedule
        """
        assert (
            num_stages % pp_degree == 0
        ), f"num_stages {num_stages} must be evenly divisible by pp_degree {pp_degree}"
        stages_per_rank = num_stages // pp_degree
        if style == "loop":
            return tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))
        elif style == "v":
            assert (
                stages_per_rank == 2
            ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
            stage_v_pairs = list(
                zip(range(pp_degree), range(num_stages - 1, pp_degree - 1, -1))
            )
            return stage_v_pairs[pp_rank]
        else:
            raise ValueError(f"Unknown style {style}")

    for stage_idx in _get_stage_indices():
        module_names = module_names_per_stage[stage_idx]
        stage, model_chunk = _build_stage_from_modules(
            stage_idx,
            module_names,
            num_stages,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx} "
            f"with modules {module_names}"
        )
        stages.append(stage)
        models.append(model_chunk)

    return stages, models
