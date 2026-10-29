# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys

import pytest

from torchtitan.config.parallelism import FSDPSymmMemScope, ParallelismConfig
from torchtitan.distributed.context_parallel import PTRRFlexAttentionCPLoadBalancer


def test_cp_import_before_parallelism_config() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from torchtitan.distributed.context_parallel import "
            "ContextParallelLoadBalancer\n"
            "from torchtitan.config.parallelism import ParallelismConfig\n"
            "from typing import get_type_hints\n"
            "load_balancer_type = get_type_hints(ParallelismConfig)["
            "'context_parallel_load_balancer']\n"
            "assert load_balancer_type == ContextParallelLoadBalancer.Config | None\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_parallelism_config_accepts_ptrr_when_cp_disabled() -> None:
    config = ParallelismConfig(
        context_parallel_degree=1,
        context_parallel_load_balancer=(PTRRFlexAttentionCPLoadBalancer.Config()),
    )
    assert isinstance(
        config.context_parallel_load_balancer,
        PTRRFlexAttentionCPLoadBalancer.Config,
    )


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


def test_parallelism_config_accepts_unset_layers_per_stage() -> None:
    assert ParallelismConfig().pipeline_parallel_layers_per_stage is None


def test_parallelism_config_accepts_positive_layers_per_stage() -> None:
    config = ParallelismConfig(pipeline_parallel_layers_per_stage=2)
    assert config.pipeline_parallel_layers_per_stage == 2


@pytest.mark.parametrize("layers_per_stage", [0, -1])
def test_parallelism_config_rejects_non_positive_layers_per_stage(
    layers_per_stage: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"pipeline_parallel_layers_per_stage must be positive.*{layers_per_stage}",
    ):
        ParallelismConfig(pipeline_parallel_layers_per_stage=layers_per_stage)


def test_parallelism_config_rejects_startup_cache_release_without_pp() -> None:
    with pytest.raises(ValueError, match="requires pipeline_parallel_degree > 1"):
        ParallelismConfig(pipeline_parallel_empty_cache_before_first_backward=True)


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
