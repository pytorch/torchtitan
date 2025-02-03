# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import List, Tuple

from torch.distributed.pipelining.schedules import (
    _PipelineScheduleRuntime,
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torchtitan.config_manager import JobConfig
from torchtitan.logging import logger
from torchtitan.models.llama.model import ModelArgs


def generate_split_points(
    job_config: JobConfig, pp_dim: int, model_config: ModelArgs
) -> List[str]:
    """
    Generate a default split point based on the number of layers and
    pipeline parallel dimension.
    """

    schedule_class = get_schedule_class(
        job_config.experimental.pipeline_parallel_schedule
    )
    if issubclass(schedule_class, PipelineScheduleSingle):
        num_stages_per_rank = 1
    elif issubclass(schedule_class, PipelineScheduleMulti):
        # Multi-stage schedules support more than 2 stages per rank, but this is the default if
        # no pipeline split is specified
        num_stages_per_rank = 2
    else:
        raise ValueError(
            f"Unsupported pipeline schedule: {job_config.experimental.pipeline_parallel_schedule}"
        )
    total_stages = pp_dim * num_stages_per_rank
    num_layers = model_config.n_layers
    if total_stages > num_layers:
        raise ValueError("Total stages cannot be greater than the number of layers")

    base_interval = num_layers // total_stages
    extra_layers = num_layers % total_stages

    splits = []
    current_layer = 0
    for i in range(total_stages - 1):
        if i == 0:
            current_layer += base_interval
        else:
            # Middle stages get an extra layer if there are any remaining
            if extra_layers > 0:
                current_layer += base_interval + 1
                extra_layers -= 1
            else:
                current_layer += base_interval
        splits.append("layers." + str(current_layer))
    logger.info(
        f"No 'pipeline_parallel_split_points' provided so the generated splits are: {splits} \
This may be sub-optimal as the number of layers per stage may be unbalanced."
    )
    return splits


def build_pipeline_schedule(job_config, stages, loss_fn):
    pp_schedule_csv = job_config.experimental.pipeline_parallel_schedule_csv

    # Validate that pp_schedule_csv is a valid path
    if pp_schedule_csv:
        if not os.path.isfile(pp_schedule_csv):
            raise FileNotFoundError(
                f"The specified path {pp_schedule_csv} does not exist or is not a file."
            )
        schedule_class = _PipelineScheduleRuntime
    else:
        schedule_class = get_schedule_class(
            job_config.experimental.pipeline_parallel_schedule
        )

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    n_microbatches = job_config.experimental.pipeline_parallel_microbatches
    # We expect that the number of local stages (`len(stages)`) is the same across all ranks
    num_total_stages = job_config.experimental.pipeline_parallel_degree * len(stages)
    if n_microbatches is None:
        n_microbatches = num_total_stages
    elif n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number \
of stages ({num_total_stages}) which may result in a bubble in the pipeline."
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
        f"Using pipeline schedule {job_config.experimental.pipeline_parallel_schedule} \
with {n_microbatches} microbatches and {num_total_stages} stages."
    )

    if pp_schedule_csv:
        assert schedule_class in [
            PipelineScheduleSingle,
            PipelineScheduleMulti,
            _PipelineScheduleRuntime,
        ], "Only PipelineScheduleSingle (single stage), PipelineScheduleMulti (multistage), \
            and _PipelineScheduleRuntime support csv schedules"
        schedule._load_csv(pp_schedule_csv)

    return schedule


# TODO(whc) should this be a utility inside torch.pipelining?
def stage_ids_this_rank(
    pp_rank: int, pp_size: int, num_stages: int, style: str = "loop"
) -> Tuple[int]:
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
