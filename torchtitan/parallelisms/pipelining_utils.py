# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torch.distributed.pipelining import (
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
)
from torchtitan.logging_utils import logger


def build_pipeline_schedule(job_config, parallel_dims, stages, loss_fn):

    looped_schedule = False
    if job_config.experimental.pipeline_parallel_schedule == "1f1b":
        schedule_class = Schedule1F1B
    elif job_config.experimental.pipeline_parallel_schedule == "gpipe":
        schedule_class = ScheduleGPipe
    elif job_config.experimental.pipeline_parallel_schedule == "interleaved_1f1b":
        schedule_class = ScheduleInterleaved1F1B
        looped_schedule = True
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

    return schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )
