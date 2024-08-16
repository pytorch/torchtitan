# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import pathlib
from typing import Tuple

from torch.distributed.pipelining import (
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
)
from torch.distributed.pipelining.schedules import PipelineScheduleMulti

from torchtitan.logging_utils import logger

PARALLELISM_DIR = pathlib.Path(__file__).parent.resolve()


def build_pipeline_schedule(job_config, parallel_dims, stages, loss_fn):

    looped_schedule = False
    zb_schedule = False
    if job_config.experimental.pipeline_parallel_schedule == "1f1b":
        schedule_class = Schedule1F1B
    elif job_config.experimental.pipeline_parallel_schedule == "gpipe":
        schedule_class = ScheduleGPipe
    elif job_config.experimental.pipeline_parallel_schedule == "interleaved_1f1b":
        schedule_class = ScheduleInterleaved1F1B
        looped_schedule = True
    elif job_config.experimental.pipeline_parallel_schedule == "zb":
        looped_schedule = True
        schedule_class = PipelineScheduleMulti
        zb_schedule = True
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

    if job_config.experimental.pipeline_parallel_schedule == "zb":
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
    else:
        stage_index_to_group_rank = None

    schedule = schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        stage_index_to_group_rank=stage_index_to_group_rank,
    )

    if zb_schedule:
        # TODO(whc) if we allow creating PipelineScheduleMulti directly from csv, we have some ux refactoring to do
        schedule.use_full_backward = False
        schedule.load_csv(os.path.join(PARALLELISM_DIR, "zb.csv"))

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
