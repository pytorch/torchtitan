# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO(whc) this can be removed after pippy migration into pytorch core is complete.
try:
    from pippy import Schedule1F1B, ScheduleGPipe
except ImportError as exc:
    raise ImportError(
        "pippy is not installed. Please install it to use pipeline parallelism. "
        "`pip install git+https://github.com/pytorch/pippy`"
    ) from exc


def build_pipeline_schedule(job_config, parallel_dims, stage, loss_fn):
    if job_config.experimental.pipeline_parallel_schedule == "1f1b":
        schedule_class = Schedule1F1B
    elif job_config.experimental.pipeline_parallel_schedule == "gpipe":
        schedule_class = ScheduleGPipe
    else:
        raise NotImplementedError(
            f"{job_config.experimental.pipeline_parallel_schedule} is not implemented"
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
