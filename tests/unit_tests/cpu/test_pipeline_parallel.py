# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
from torch.distributed.pipelining.schedules import PipelineScheduleMulti

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.pipeline_parallel import _build_pipeline_schedule


@pytest.mark.parametrize("defer_pp_recv", [False, True])
def test_looped_pipeline_schedule_forwards_deferred_receive_config(
    defer_pp_recv: bool,
) -> None:
    captured_kwargs = {}

    class _Schedule(PipelineScheduleMulti):
        def __init__(self, stages, **kwargs):
            captured_kwargs.update(kwargs)

    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="test",
        pipeline_parallel_defer_recv=defer_pp_recv,
    )
    with patch(
        "torchtitan.distributed.pipeline_parallel.get_schedule_class",
        return_value=_Schedule,
    ):
        _build_pipeline_schedule(
            parallelism=parallelism,
            num_microbatches=4,
            stages=[object(), object()],
            loss_fn=lambda prediction, target: (prediction, target),
        )

    assert captured_kwargs["defer_pp_recv"] is defer_pp_recv
