# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D pipeline parallelism to the Llama model.

import math

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    get_schedule_class,
    PipelineScheduleSingle,
)

from torchtitan.components.loss import LossFunction
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.pipeline import (
    build_pipeline_schedule,
    generate_llm_fqn_per_model_part,
    pipeline_module_split,
)

from torchtitan.protocols.train_spec import BaseModelArgs, ParallelizeFunction
from torchtitan.tools.logging import logger


def pipeline_llama(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: torch.device,
    model_args: BaseModelArgs,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    if job_config.parallelism.pipeline_parallel_split_points != []:
        raise ValueError(
            "pipeline_parallel_split_points is deprecated. Please use module_fqns_per_model_part instead."
            "You can generate module_fqns_per_model_part programmatically with generate_llm_fqn_per_model_part"
        )

    pp_mesh = parallel_dims.world_mesh["pp"]

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
