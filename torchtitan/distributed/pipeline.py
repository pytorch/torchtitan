# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math
import os
from typing import Callable, Optional

from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.pipelining.stage import PipelineStage

from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger


__all__ = ["build_pipeline_schedule", "generate_split_points", "stage_ids_this_rank"]


# TODO: It's unclear if this API is general enough to be used by other models.
# If not, we should move it to a Transformer-specific directory.
def generate_split_points(
    schedule_str: str,
    layers_per_stage: Optional[int],
    pp_dim: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[str]:
    """
    Generate a list of split points based on the number of layers and
    pipeline parallel dimension, ensuring the first and last stages have the least layers.

    Args:
        schedule_str (str): The string of the schedule name.
        layers_per_stage (int): The number of layers per stage.
        pp_dim (int): The pipeline parallel dimension.
        num_layers (int): The number of layers in the model.
        input_output_weight (int): The number of layers to consider the input/output modules in the layer calculation.

    Returns:
        list[str]: A list of split point FQNs.
    """

    schedule_class = get_schedule_class(schedule_str)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    num_stages_per_rank = 1 if is_single_stage_schedule else 2

    if layers_per_stage is not None:
        total_stages = math.ceil(num_layers / layers_per_stage)
        if total_stages % pp_dim != 0:
            raise ValueError(
                f"Number of stages ({total_stages}) must be divisible by the pipeline parallel dimension ({pp_dim})."
                f"Each rank should have the same number of stages. "
            )
        num_stages_per_rank = total_stages // pp_dim

        if is_single_stage_schedule and num_stages_per_rank != 1:
            raise ValueError(
                f"Number of stages per rank ({num_stages_per_rank}) must be 1 for single stage schedules."
            )
        elif not is_single_stage_schedule and num_stages_per_rank < 2:
            raise ValueError(
                f"Number of stages per rank ({num_stages_per_rank}) must be >= 2 for multi stage schedules."
            )
    else:
        total_stages = pp_dim * num_stages_per_rank
        if total_stages > num_layers:
            raise ValueError("Total stages cannot be greater than the number of layers")

    # Calculate effective number of layers including input and output weights
    effective_num_layers = num_layers + input_weight + output_weight
    base_layers_per_stage = effective_num_layers // total_stages

    splits = [""] * (total_stages - 1)
    current_layer_index = 0

    # First stage
    layers_on_first_stage = max(0, base_layers_per_stage - input_weight)
    current_layer_index += layers_on_first_stage
    splits[0] = "layers." + str(current_layer_index)

    # Last stage
    layers_on_last_stage = max(0, base_layers_per_stage - output_weight)
    splits[-1] = "layers." + str(num_layers - layers_on_last_stage)

    # Middle stages
    remaining_layers = num_layers - layers_on_first_stage - layers_on_last_stage - 1
    middle_stages = len(splits) - 2
    layers_per_middle_stage = remaining_layers // middle_stages
    # split remainder evenly across middle stages
    remainder = remaining_layers % middle_stages

    for i in range(1, middle_stages + 1):
        current_layer_index += layers_per_middle_stage
        if remainder > 0:
            current_layer_index += 1
            remainder -= 1
        splits[i] = "layers." + str(current_layer_index)

    logger.info(
        f"No 'pipeline_parallel_split_points' provided so the generated splits are: {splits} "
        "This may be sub-optimal as the number of layers per stage may be unbalanced."
    )
    return splits


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
    n_microbatches = job_config.parallelism.pipeline_parallel_microbatches
    # We expect that the number of local stages (`len(stages)`) is the same across all ranks
    num_total_stages = job_config.parallelism.pipeline_parallel_degree * len(stages)
    if n_microbatches is None:
        n_microbatches = num_total_stages
    elif n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number "
            f"of stages ({num_total_stages}) which may result in a bubble in the pipeline."
        )

    # validate that the batch size is divisible by the number of microbatches otherwise we'll hang or error during training
    if job_config.training.batch_size % n_microbatches != 0:
        raise ValueError(
            f"Batch size {job_config.training.batch_size} must be divisible by number of microbatches {n_microbatches}. "
            "Update the config arguments for either batch_size or pipeline_parallel_microbatches."
        )

    schedule = schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )
    logger.info(
        f"Using pipeline schedule {job_config.parallelism.pipeline_parallel_schedule} "
        f"with {n_microbatches} microbatches and {num_total_stages} stages."
    )

    if pp_schedule_csv:
        assert schedule_class in [
            PipelineScheduleSingle,
            PipelineScheduleMulti,
            _PipelineScheduleRuntime,
        ], (
            "Only PipelineScheduleSingle (single stage), PipelineScheduleMulti (multistage), "
            "and _PipelineScheduleRuntime support csv schedules"
        )
        schedule._load_csv(pp_schedule_csv)

    return schedule


# TODO(whc) should this be a utility inside torch.pipelining?
def stage_ids_this_rank(
    pp_rank: int, pp_size: int, num_stages: int, style: str = "loop"
) -> tuple[int]:
    """Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule"""
    assert (
        num_stages % pp_size == 0
    ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
    stages_per_rank = num_stages // pp_size
    if style == "loop":
        return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
    elif style == "v":
        assert (
            stages_per_rank == 2
        ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        stage_v_pairs = list(
            zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1))
        )
        return stage_v_pairs[pp_rank]
