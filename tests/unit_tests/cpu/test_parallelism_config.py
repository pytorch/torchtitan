# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.config import ParallelismConfig


def test_parallelism_config_default_load_balancer() -> None:
    assert ParallelismConfig().context_parallel_load_balancer == "headtail"


def test_parallelism_config_accepts_none_load_balancer() -> None:
    config = ParallelismConfig(context_parallel_load_balancer=None)
    assert config.context_parallel_load_balancer is None


def test_parallelism_config_accepts_ptrr_when_cp_disabled() -> None:
    config = ParallelismConfig(
        context_parallel_degree=1,
        context_parallel_load_balancer="ptrr",
    )
    assert config.context_parallel_load_balancer == "ptrr"


def test_parallelism_config_rejects_unknown_load_balancer() -> None:
    with pytest.raises(
        ValueError,
        match=r"must be one of: None, 'headtail', 'ptrr' \(got 'foo'\)",
    ):
        ParallelismConfig(context_parallel_load_balancer="foo")


def test_parallelism_config_rejects_empty_string_load_balancer() -> None:
    with pytest.raises(ValueError, match="cannot be an empty string"):
        ParallelismConfig(context_parallel_load_balancer="")


def test_parallelism_config_rejects_unknown_load_balancer_when_cp_disabled() -> None:
    with pytest.raises(
        ValueError,
        match=r"must be one of: None, 'headtail', 'ptrr' \(got 'foo'\)",
    ):
        ParallelismConfig(
            context_parallel_degree=1,
            context_parallel_load_balancer="foo",
        )


def test_parallelism_config_default_schedule() -> None:
    assert ParallelismConfig().pipeline_parallel_schedule == "1F1B"


def test_parallelism_config_accepts_interleaved_1f1b() -> None:
    config = ParallelismConfig(pipeline_parallel_schedule="Interleaved1F1B")
    assert config.pipeline_parallel_schedule == "Interleaved1F1B"


def test_parallelism_config_accepts_pipeline_schedule_multi() -> None:
    config = ParallelismConfig(pipeline_parallel_schedule="PipelineScheduleMulti")
    assert config.pipeline_parallel_schedule == "PipelineScheduleMulti"


@pytest.mark.parametrize("schedule", ["foo", "Interleved1F1B", ""])
def test_parallelism_config_rejects_invalid_schedule(schedule: str) -> None:
    with pytest.raises(
        ValueError,
        match=rf"pipeline_parallel_schedule {schedule!r}",
    ):
        ParallelismConfig(pipeline_parallel_schedule=schedule)


def test_parallelism_config_rejects_unknown_schedule_when_pp_disabled() -> None:
    with pytest.raises(ValueError, match=r"pipeline_parallel_schedule 'foo'"):
        ParallelismConfig(
            pipeline_parallel_degree=1,
            pipeline_parallel_schedule="foo",
        )
