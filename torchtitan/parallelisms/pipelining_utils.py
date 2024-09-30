# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import pathlib
from typing import Tuple

from .pipelining import (
    Schedule1F1B,
    ScheduleFlexibleInterleaved1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
)
from .pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torchtitan.logging import logger

PARALLELISM_DIR = pathlib.Path(__file__).parent.resolve()


def build_pipeline_schedule(job_config, stages, loss_fn):
    looped_schedule = False
    if job_config.experimental.pipeline_parallel_schedule == "1f1b":
        schedule_class = Schedule1F1B
    elif job_config.experimental.pipeline_parallel_schedule == "gpipe":
        schedule_class = ScheduleGPipe
    elif job_config.experimental.pipeline_parallel_schedule == "interleaved_1f1b":
        schedule_class = ScheduleInterleaved1F1B
        looped_schedule = True
    elif (
        job_config.experimental.pipeline_parallel_schedule
        == "flexible_interleaved_1f1b"
    ):
        schedule_class = ScheduleFlexibleInterleaved1F1B
        looped_schedule = True
    elif job_config.experimental.pipeline_parallel_schedule == "zb_v":
        looped_schedule = True
        schedule_class = PipelineScheduleMulti
    elif job_config.experimental.pipeline_parallel_schedule == "zb":
        looped_schedule = True
        schedule_class = ScheduleFlexibleInterleaved1F1B
    else:
        raise NotImplementedError(
            f"{job_config.experimental.pipeline_parallel_schedule} is not implemented"
        )
    logger.info(
        f"Using pipeline schedule {job_config.experimental.pipeline_parallel_schedule}"
    )
    n_microbatches = job_config.experimental.pipeline_parallel_microbatches
    if n_microbatches is None:
        n_microbatches = job_config.experimental.pipeline_parallel_degree

    # Validation that the stages are compatible with the schedule
    if isinstance(schedule_class, PipelineScheduleSingle):
        if len(stages) != 1:
            raise ValueError(
                f"PipelineScheduleSingle requires exactly one stage, got {len(stages)}"
            )
    elif isinstance(schedule_class, PipelineScheduleMulti):
        if len(stages) < 2:
            raise ValueError(
                f"PipelineScheduleMulti requires at least two stages, got {len(stages)}"
            )

    if job_config.experimental.pipeline_parallel_schedule == "zb_v":
        # TODO: hardcoded and only used for V-shaped zero bubble
        stage_index_to_group_rank = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 3,
            5: 2,
            6: 1,
            7: 0,
        }
        schedule = schedule_class(
            stages if looped_schedule else stages[0],
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            stage_index_to_group_rank=stage_index_to_group_rank,
        )
        # TODO(whc) if we allow creating PipelineScheduleMulti directly from csv, we have some ux refactoring to do
        schedule.use_full_backward = False
        schedule._load_csv(os.path.join(PARALLELISM_DIR, "zb.csv"))
    elif job_config.experimental.pipeline_parallel_schedule == "zb":
        schedule = schedule_class(
            stages if looped_schedule else stages[0],
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            enable_zero_bubble=True,
        )
    else:
        schedule = schedule_class(
            stages if looped_schedule else stages[0],
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
        )

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
