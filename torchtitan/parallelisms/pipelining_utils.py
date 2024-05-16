# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.pipelining import Schedule1F1B, ScheduleGPipe

from torchtitan.logging_utils import logger


def build_pipeline_schedule(job_config, parallel_dims, stage, loss_fn):
    if job_config.experimental.pipeline_parallel_schedule == "1f1b":
        schedule_class = Schedule1F1B
    elif job_config.experimental.pipeline_parallel_schedule == "gpipe":
        schedule_class = ScheduleGPipe
    else:
        raise NotImplementedError(
            f"{job_config.experimental.pipeline_parallel_schedule} is not implemented"
        )
    logger.info(
        f"Using pipeline schedule {job_config.experimental.pipeline_parallel_schedule}"
    )
    return schedule_class(
        stage,
        n_microbatches=parallel_dims.pp,
        loss_fn=loss_fn,
    )


def split_stage_fqns(fqns, split_points, stage_id):
    """Helper for splitting ordered list of layer names into layers per stage.

    split_points is a list of layer names, each layer will be the first layer in a stage
    """
    stages = []
    cur = []

    for name in fqns:
        if name in split_points:
            assert len(
                cur
            ), f"{name} is not a valid split point, do not specify the first layer of stage 0"
            stages.append(cur)
            cur = []
        cur.append(name)

    stages.append(cur)
    return stages[stage_id]
