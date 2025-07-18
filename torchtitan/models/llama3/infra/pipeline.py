# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D pipeline parallelism to the Llama model.

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    get_schedule_class,
    PipelineScheduleSingle,
)

from torchtitan.components.loss import LossFunction
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.pipeline import (
    build_pipeline_schedule,
    generate_module_names_per_stage,
    pipeline_module_split,
)
from torchtitan.protocols.train_spec import ParallelizeFunction
from torchtitan.tools.logging import logger

from ..model.args import TransformerModelArgs


def pipeline_llama(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: torch.device,
    model_config: TransformerModelArgs,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    if job_config.parallelism.pipeline_parallel_split_points != []:
        raise ValueError(
            "pipeline_parallel_split_points is deprecated. Please use module_names_per_model_chunk instead."
            "You can generate module_names_per_model_chunk programmatically with generate_module_names_per_stage"
        )

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

    module_names_per_stage = job_config.parallelism.module_names_per_model_chunk
    if module_names_per_stage == []:
        module_names_per_stage = generate_module_names_per_stage(
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
