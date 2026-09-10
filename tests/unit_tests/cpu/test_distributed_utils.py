# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config import CommConfig
from torchtitan.distributed import utils as dist_utils
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


def test_real_pp_fake_spmd_init_maps_physical_to_logical_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    process_group = torch.distributed.ProcessGroup(2, 4)
    store = MagicMock()
    pp_mesh = cast(DeviceMesh, MagicMock())

    with (
        patch.object(dist_utils, "_real_pp_fake_spmd_state", None),
        patch.object(dist_utils, "init_fake_mode") as init_fake_mode,
        patch.object(
            dist_utils.dist,
            "rendezvous",
            return_value=iter([(store, 2, 4)]),
        ),
        patch.object(
            dist_utils.c10d,
            "_new_process_group_helper",
            return_value=(process_group, store),
        ) as new_process_group,
        patch.object(dist_utils.DeviceMesh, "from_group", return_value=pp_mesh),
        patch.dict(dist_utils.c10d._world.pg_group_ranks, {}, clear=False),
    ):
        assert dist_utils._init_real_pp_fake_spmd(16, timedelta(seconds=30)) == 16

        init_fake_mode.assert_called_once_with(16, rank=8)
        assert new_process_group.call_args.kwargs["global_ranks_in_group"] == [
            0,
            4,
            8,
            12,
        ]
        assert dist_utils.c10d._world.pg_group_ranks[process_group] == {
            0: 0,
            4: 1,
            8: 2,
            12: 3,
        }
        assert dist_utils.get_real_pp_mesh(4) is pp_mesh


def test_real_pp_fake_spmd_requires_one_process_per_pp_rank() -> None:
    state = dist_utils._RealPPFakeSpmdState(
        pp_mesh=cast(DeviceMesh, MagicMock()),
        physical_world_size=4,
    )
    with (
        patch.object(dist_utils, "_real_pp_fake_spmd_state", state),
        pytest.raises(ValueError, match="one physical process per PP rank"),
    ):
        dist_utils.get_real_pp_mesh(2)


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
