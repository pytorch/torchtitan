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
from torchtitan.distributed.parallelism_context import ParallelismContext


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


@pytest.mark.parametrize(
    ("dp_replicate", "edp_replicate", "ep", "expected"),
    [
        (4, None, 2, 4),
        (4, 2, 2, 2),
        (4, 1, 2, 1),
        # Without EP, only the dense replicate degree is allowed.
        (4, None, 1, 4),
        (4, 4, 1, 4),
    ],
)
def test_parallelism_config_accepts_expert_replicate_degree(
    dp_replicate: int, edp_replicate: int | None, ep: int, expected: int
) -> None:
    config = ParallelismConfig(
        data_parallel_replicate_degree=dp_replicate,
        expert_parallel_degree=ep,
        expert_data_parallel_replicate_degree=edp_replicate,
    )
    assert (
        ParallelismContext.resolve_edp_replicate(
            config.data_parallel_replicate_degree,
            config.expert_data_parallel_replicate_degree,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("dp_replicate", "edp_replicate", "ep", "match"),
    [
        # None is the only way to inherit the dense replicate degree.
        (4, -1, 2, r">= 1, or None .*got -1"),
        (4, 0, 2, r">= 1, or None .*got 0"),
        (4, 3, 2, r"\(3\) must divide data_parallel_replicate_degree \(4\)"),
        (2, 4, 2, r"\(4\) must divide data_parallel_replicate_degree \(2\)"),
        (4, 2, 1, r"requires expert_parallel_degree > 1"),
    ],
)
def test_parallelism_config_rejects_invalid_expert_replicate_degree(
    dp_replicate: int, edp_replicate: int, ep: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ParallelismConfig(
            data_parallel_replicate_degree=dp_replicate,
            expert_parallel_degree=ep,
            expert_data_parallel_replicate_degree=edp_replicate,
        )
