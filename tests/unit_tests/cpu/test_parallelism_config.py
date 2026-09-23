# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.config import FSDPSymmMemScope, ParallelismConfig
from torchtitan.distributed.context_parallel import HeadTailLoadBalancer, PTRRLoadBalancer


def test_parallelism_config_default_load_balancer() -> None:
    assert isinstance(
        ParallelismConfig().context_parallel_load_balancer,
        HeadTailLoadBalancer.Config,
    )


def test_parallelism_config_accepts_ptrr_when_cp_disabled() -> None:
    config = ParallelismConfig(
        context_parallel_degree=1,
        context_parallel_load_balancer=PTRRLoadBalancer.Config(),
    )
    assert isinstance(config.context_parallel_load_balancer, PTRRLoadBalancer.Config)


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


def test_parallelism_config_disables_fsdp_symm_mem_by_default() -> None:
    assert ParallelismConfig().fsdp_symm_mem_scope is None


@pytest.mark.parametrize("scope", ["all", "dense"])
def test_parallelism_config_accepts_fsdp_symm_mem_scopes(
    scope: FSDPSymmMemScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (9, 0))

    config = ParallelismConfig(fsdp_symm_mem_scope=scope)

    assert config.fsdp_symm_mem_scope == scope


def test_parallelism_config_rejects_unknown_fsdp_symm_mem_scope() -> None:
    with pytest.raises(
        ValueError,
        match=r"fsdp_symm_mem_scope must be one of: .* \(got 'sparse'\)",
    ):
        ParallelismConfig(
            fsdp_symm_mem_scope="sparse"  # pyrefly: ignore [bad-argument-type]
        )
