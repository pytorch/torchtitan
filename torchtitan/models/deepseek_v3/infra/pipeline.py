# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D pipeline parallelism to the Llama model.

import copy

import torch.nn as nn
import torch
from torch.distributed import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    get_schedule_class,
    ScheduleZBVZeroBubble,
)

from torchtitan.components.loss import LossFunction
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.pipeline import (
    build_pipeline_schedule,
    stage_ids_this_rank,
)
from torchtitan.protocols.train_spec import DeviceType, ParallelizeFunction
from torchtitan.tools.logging import logger

# Should I use BaseModelArgs instead of DeepSeekV3ModelArgs?
from ..model.args import DeepSeekV3ModelArgs


def _pipeline_friendly_forward(self, tokens: torch.Tensor):
    """
    Pipeline friendly forward pass for the DeepSeekV3 model. 
    This method is only used when pipeline parallelism is enabled.
    If model attributes are None, they are skipped in the forward pass.

    Args:
        tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).

    Returns:
        torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
    """
    h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
    # h: (batch_size, seq_len, dim)
    for layer in self.layers.values():
        h = layer(h, self.freqs_cis)
    h = self.norm(h) if self.norm is not None else h
    output = self.output(h) if self.output is not None else h
    return output


def _patch_model_for_pipeline(model: nn.Module):
    """
    Patches the model's forward method to be pipeline-friendly.
    This only affects models used in pipeline parallelism.
    
    Args:
        model: The model to patch
    """
    # Store the original forward method
    if not hasattr(model, '_original_forward'):
        model._original_forward = model.forward
        # Replace with pipeline-friendly version
        model.forward = _pipeline_friendly_forward.__get__(model, model.__class__)


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
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: DeepSeekV3ModelArgs,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    pp_mesh = world_mesh["pp"]
    
    # Determine the number of virtual stages based on schedule type
    schedule_class = get_schedule_class(job_config.parallelism.pipeline_parallel_schedule)
    is_single_stage_schedule = schedule_class.__name__ in ["PipelineScheduleSingle"]
    
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
    
    stages, model_parts = pipeline_deepseekv3_module_split(
        model, pp_mesh, parallel_dims, job_config, device, module_names_per_stage)

    # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
    # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
    # optimizer, and checkpointing
    for i, m in enumerate(model_parts):
        # apply SPMD-style PT-D techniques
        m = parallelize_fn(m, world_mesh, parallel_dims, job_config)
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

def pipeline_deepseekv3_module_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    module_names_per_stage: list[list[str]],
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """
    This API creates pipeline stages based on specified module names for each stage.
    
    Args:
        whole_model: The complete DeepSeekV3Model to be split
        pp_mesh: Pipeline parallel device mesh
        parallel_dims: Parallel dimensions configuration
        job_config: Job configuration
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
    parallelism_config = job_config.parallelism

    def _build_stage_from_modules(
        stage_idx: int,
        module_names: list[str],
        is_first: bool = False,
        is_last: bool = False,
    ) -> tuple[PipelineStage, nn.Module]:
        model = copy.deepcopy(whole_model)
        
        # Patch the model to use pipeline-friendly forward method
        _patch_model_for_pipeline(model)
        
        # Create a set of modules to keep for faster lookup
        modules_to_keep = set(module_names)
        
        # Handle embeddings - remove if not in this stage and not first stage
        if "tok_embeddings" not in modules_to_keep:
            model.tok_embeddings = None
        
        # Handle layers - remove layers not specified for this stage
        layers_to_keep = set()
        for name in modules_to_keep:
            if name.startswith("layers."):
                # Extract layer number (e.g., "layers.0" -> "0")
                layer_num = name.split(".", 1)[1]
                layers_to_keep.add(layer_num)
        
        # Remove layers not in this stage
        for layer_name in list(model.layers.keys()):
            if layer_name not in layers_to_keep:
                del model.layers[layer_name]
        
        # Handle final normalization layer
        if "norm" not in modules_to_keep:
            model.norm = None
            
        # Handle output projection layer
        if "output" not in modules_to_keep:
            model.output = None
        
        stage = PipelineStage(
            model,
            stage_idx,
            len(module_names_per_stage),
            device,
            group=pp_mesh.get_group("pp"),
        )
        return stage, model

    num_stages = len(module_names_per_stage)
    stages = []
    models = []

    schedule_class = get_schedule_class(parallelism_config.pipeline_parallel_schedule)
    style = "v" if schedule_class == ScheduleZBVZeroBubble else "loop"

    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, style=style):
        module_names = module_names_per_stage[stage_idx]
        stage, model_chunk = _build_stage_from_modules(
            stage_idx,
            module_names,
            is_first=stage_idx == 0,
            is_last=stage_idx == num_stages - 1,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx} "
            f"with modules {module_names}"
        )
        stages.append(stage)
        models.append(model_chunk)
    
    return stages, models
