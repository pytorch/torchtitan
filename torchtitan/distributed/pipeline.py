# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Callable

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
    pp_degree: int,
    num_layers: int,
    num_layers_per_stage: int | None,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[str]:
    """
    Generate a list of split points based on the input configs. In this function,
    the number of effective layers considered is the summation of num_layers,
    input_weight, and output_weight.

    If num_layers_per_virtual_stage is given, we require rigid fit of the
    effective layers (regular layers + weighted input + weighted output)
    onto pipeline stages and ranks, with several assertions. It is the users'
    responsibility to figure out the input weight, output weight, and the
    number of regular layers, so that they can be arranged neatly.

    If num_layers_per_virtual_stage is None, we by default set each pipeline rank
    to have 1 stage if schedule_str is a single-stage schedule, or 2 virtual stages
    if it is a multi-stage schedule, and try to distribute all effective layers
    evenly onto the PP stages. If there are extra layers, we disperse them in
    the starting stages.

    Args:
        schedule_str (str): The string of the schedule name.
        pp_degree (int): The pipeline parallel dimension.
        num_layers (int): The number of layers in the model.
        input_weight (int): The number of layers to consider the input modules in layer calculation.
        output_weight (int): The number of layers to consider the output modules in layer calculation.
        num_layers_per_stage (int): The number of layers per (virtual) pipeline stage.

    Returns:
        list[str]: A list of split point FQNs.
    """

    schedule_class = get_schedule_class(schedule_str)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)

    num_effective_layers = num_layers + input_weight + output_weight

    if num_layers_per_stage is not None:
        # If num_layers_per_stage is provided, we require a rigid fit of the effective layers
        assert num_effective_layers % pp_degree == 0
        num_layers_per_pipeline_rank = num_effective_layers // pp_degree

        assert num_layers_per_pipeline_rank % num_layers_per_stage == 0
        num_stages_per_rank = num_layers_per_pipeline_rank // num_layers_per_stage

        num_total_virtual_stages = num_stages_per_rank * pp_degree
        num_extra_layers = 0

        if is_single_stage_schedule:
            assert (
                num_stages_per_rank == 1
            ), f"Number of stages per rank ({num_stages_per_rank}) must be 1 for single-stage schedules."
        else:
            assert (
                num_stages_per_rank >= 2
            ), f"Number of stages per rank ({num_stages_per_rank}) must be >= 2 for multi-stage schedules."
    else:
        # In a multi-stage schedule, if num_layers_per_stage is not
        # provided, by default each pipeline rank has 2 virtual stages.
        num_stages_per_rank = 1 if is_single_stage_schedule else 2
        num_total_virtual_stages = pp_degree * num_stages_per_rank

        if num_total_virtual_stages > num_effective_layers:
            raise ValueError(
                "The number of total stages cannot be greater than the number of effective layers."
            )

        num_layers_per_stage = num_effective_layers // num_total_virtual_stages
        num_extra_layers = num_effective_layers % num_total_virtual_stages

    assert num_layers_per_stage >= max(input_weight, output_weight)

    splits = []
    current_layer = 0
    for i in range(num_total_virtual_stages - 1):
        if i == 0:
            current_layer += num_layers_per_stage - input_weight
        else:
            current_layer += num_layers_per_stage
        # extra layers will be dispersed to the first stages
        if num_extra_layers > 0:
            current_layer += 1
            num_extra_layers -= 1
        splits.append("layers." + str(current_layer))

    logger.info(
        "No 'pipeline_parallel_split_points' provided. Here is the auto-generated split, "
        f"which may be sub-optimal: {splits}."
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
    microbatch_size = job_config.parallelism.pipeline_parallel_microbatch_size
    batch_size = job_config.training.batch_size
    # validate that the batch size is divisible by the microbatch_size otherwise we'll hang or error during training
    if batch_size % microbatch_size != 0:
        raise ValueError(
            f"Batch size {job_config.training.batch_size} must be divisible by number of microbatches {n_microbatches}. "
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
