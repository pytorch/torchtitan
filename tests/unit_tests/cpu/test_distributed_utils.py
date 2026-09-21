# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
from spmd_types import SpmdType
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config import CommConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.spmd_types import (
    current_module_input_spmd_type,
    current_module_output_spmd_type,
    set_current_module_spmd_types,
)
from torchtitan.distributed.utils import init_distributed


def test_bf16x9_is_enabled_on_future_nvidia_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matmul = SimpleNamespace(fp32_precision="ieee")
    monkeypatch.setattr(dist_utils, "device_type", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.backends.cuda, "matmul", matmul)

    dist_utils.enable_fp32_matmul_emulation_with_bf16x9()

    assert matmul.fp32_precision == "bfx9"


def test_fake_pg_uses_requested_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NGPU", "8")
    monkeypatch.setenv("RANK", "6")
    with (
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torchtitan.distributed.utils.init_fake_mode") as init_fake_mode,
    ):
        assert init_distributed(CommConfig(mode="fake_backend")) == 8
    init_fake_mode.assert_called_once_with(8, rank=6)


def test_fake_pg_rejects_out_of_range_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NGPU", "8")
    monkeypatch.setenv("RANK", "8")
    with (
        patch("torch.distributed.is_initialized", return_value=False),
        pytest.raises(ValueError, match=r"RANK must be in \[0, 8\)"),
    ):
        init_distributed(CommConfig(mode="fake_backend"))


def test_dist_sum_tensor_keeps_local_result_as_tensor():
    value = torch.tensor(3, dtype=torch.int64)

    result = dist_utils.dist_sum_tensor(value)

    assert result is value


def test_dist_sum_tensor_waits_for_distributed_result():
    value = torch.tensor(3, dtype=torch.int64)
    reduced = torch.tensor(8, dtype=torch.int64)
    mesh = cast(DeviceMesh, object())

    with (
        patch.object(dist_utils.funcol, "all_reduce", return_value=reduced) as reduce,
        patch.object(dist_utils.funcol, "wait_tensor", return_value=reduced) as wait,
    ):
        result = dist_utils.dist_sum_tensor(value, mesh)

    assert result is reduced
    reduce.assert_called_once_with(value, reduceOp="SUM", group=mesh)
    wait.assert_called_once_with(reduced)


def test_module_spmd_context_exposes_boundary_types() -> None:
    outer_input = SpmdType({"tp": spmd.S(0)})
    outer_output = SpmdType({"tp": spmd.S(0)})
    inner_input = SpmdType({"tp": spmd.I})
    inner_output = SpmdType({"tp": spmd.P})

    with set_current_module_spmd_types(
        input_types={"x": outer_input},
        output_type=outer_output,
    ):
        assert current_module_input_spmd_type("x", "tp") == spmd.S(0)
        assert current_module_output_spmd_type("tp") == spmd.S(0)
        with set_current_module_spmd_types(
            input_types={"input": inner_input},
            output_type=inner_output,
        ):
            assert current_module_input_spmd_type("input", "tp") == spmd.I
            assert current_module_output_spmd_type("tp") == spmd.P
        assert current_module_input_spmd_type("x", "tp") == spmd.S(0)
