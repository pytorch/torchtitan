# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D pipeline parallelism to the Llama model.

import copy

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    get_schedule_class,
    PipelineScheduleSingle,
    ScheduleZBVZeroBubble,
)

from torchtitan.components.loss import LossFunction
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.pipeline import build_pipeline_schedule, stage_ids_this_rank
from torchtitan.protocols.train_spec import ParallelizeFunction
from torchtitan.tools.logging import logger

from ..model.args import DeepSeekV3ModelArgs


def generate_module_names_per_stage(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[list[str]]:
    """
    Programmatically generates module names per stage for pipeline parallelism with weighting.

    Args:
        num_stages: Number of pipeline stages
        num_layers: Total number of transformer layers in the model
        input_weight: Weight for input modules (tok_embeddings) in layer calculation
        output_weight: Weight for output modules (norm + output) in layer calculation

    Returns:
        List of lists containing module names for each stage

    Example:
        generate_module_names_per_stage(2, 3, input_weight=2, output_weight=2)
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

    # Ensure each stage gets at least the weight of input/output modules
    if layers_per_stage < max(input_weight, output_weight):
        raise ValueError(
            f"Layers per stage ({layers_per_stage}) must be >= max(input_weight={input_weight}, output_weight={output_weight})"
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


def pipeline_deepseekv3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: torch.device,
    model_config: DeepSeekV3ModelArgs,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    pp_mesh = parallel_dims.world_mesh["pp"]

    # Determine the number of virtual stages based on schedule type
    schedule_class = get_schedule_class(
        job_config.parallelism.pipeline_parallel_schedule
    )
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)

    # For multi-stage schedules, default is 2 virtual stages per rank
    # For single-stage schedules, default is 1 virtual stage per rank
    stages_per_rank = 1 if is_single_stage_schedule else 2
    num_virtual_stages = parallel_dims.pp * stages_per_rank

    # Generate module names per stage programmatically with weighting
    num_layers = model_config.n_layers

    # You can adjust these weights based on the computational cost of embeddings and output layers
    # Higher weights mean these modules are treated as "heavier" in the distribution
    input_weight = 1  # Weight for tok_embeddings
    output_weight = 1  # Weight for norm + output layers

    module_names_per_stage = generate_module_names_per_stage(
        num_virtual_stages, num_layers, input_weight, output_weight
    )
    for i, stage_ms in enumerate(module_names_per_stage):
        logger.info(f"Stage {i}: {stage_ms}")

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


def pipeline_module_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    pp_schedule: str,
    device: torch.device,
    module_names_per_stage: list[list[str]],
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """
    This API creates pipeline stages based on specified module names for each stage.

    Args:
        whole_model: The complete model to be split
        pp_mesh: Pipeline parallel device mesh
        pp_schedule: Name of pipeline parallelism schedule
        device: Device type
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
    pp_size = pp_mesh.size()

    def _build_stage_from_modules(
        stage_idx: int, module_names: list[str], num_stages: int
    ) -> tuple[PipelineStage, nn.Module]:
        model = copy.deepcopy(whole_model)

        # Create a set of modules to keep for faster lookup
        modules_to_keep = set(module_names)
        print(f"Stage {stage_idx}: Modules to keep: {modules_to_keep}")
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
    style = "v" if schedule_class == ScheduleZBVZeroBubble else "loop"

    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, style=style):
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
