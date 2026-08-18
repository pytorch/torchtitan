# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock, patch

import torch
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    _PipelineScheduleRuntime,
    PipelineScheduleMulti,
)

from torchtitan.config import ParallelismConfig
from torchtitan.distributed import pipeline_parallel


def test_build_pipeline_schedule_forwards_max_active_stages():
    captured_kwargs: dict[str, object] = {}

    class FakeRuntimeSchedule(_PipelineScheduleRuntime):
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)

    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="FakeRuntimeSchedule",
        pipeline_parallel_max_active_stages=4,
    )

    with patch.object(
        pipeline_parallel,
        "get_schedule_class",
        return_value=FakeRuntimeSchedule,
    ):
        stages: list[PipelineStage] = [MagicMock(spec=PipelineStage)]
        pipeline_parallel._build_pipeline_schedule(
            parallelism=parallelism,
            local_batch_size=4,
            stages=stages,
            loss_fn=lambda *args, **kwargs: (torch.tensor(0.0), {}),
        )

    assert captured_kwargs["max_active_stages"] == 4


def test_build_pipeline_schedule_warns_when_max_active_stages_is_ignored():
    class FakeMultiSchedule(PipelineScheduleMulti):
        def __init__(self, *args, **kwargs):
            pass

    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="FakeMultiSchedule",
        pipeline_parallel_max_active_stages=4,
    )
    stages: list[PipelineStage] = [MagicMock(spec=PipelineStage)]

    with (
        patch.object(
            pipeline_parallel,
            "get_schedule_class",
            return_value=FakeMultiSchedule,
        ),
        patch.object(pipeline_parallel.logger, "warning") as warning,
    ):
        pipeline_parallel._build_pipeline_schedule(
            parallelism=parallelism,
            local_batch_size=4,
            stages=stages,
            loss_fn=lambda *args, **kwargs: (torch.tensor(0.0), {}),
        )

    warning.assert_called_once()
