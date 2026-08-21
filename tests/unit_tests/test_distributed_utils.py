# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config import CommConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.spmd_types import (
    set_current_spmd_mesh,
    set_spmd_meshes,
    spmd_distribute_tensor,
)
from torchtitan.distributed.utils import init_distributed


def test_fake_pg_uses_requested_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NGPU", "8")
    monkeypatch.setenv("RANK", "6")
    with (
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torchtitan.distributed.utils.init_fake_mode") as init_fake_mode,
    ):
        assert init_distributed(CommConfig(mode="fake_backend")) == 8
    init_fake_mode.assert_called_once_with(8, "fake_backend", rank=6)


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


class TestSpmdLocalGradNorm(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 8

    @property
    def device_type(self) -> str:
        return "cpu"

    @with_comms
    def test_multiple_meshes_and_placements(self) -> None:
        dense_mesh = init_device_mesh(
            "cpu",
            (2, 4),
            mesh_dim_names=("dense_0", "dense_1"),
        )
        sparse_mesh = init_device_mesh(
            "cpu",
            (4, 2),
            mesh_dim_names=("sparse_0", "sparse_1"),
        )
        set_spmd_meshes(
            dense_mesh=dense_mesh,
            dense_storage_mesh=dense_mesh,
            sparse_mesh=sparse_mesh,
            sparse_storage_mesh=sparse_mesh,
        )

        # Construct eight parameters across two meshes. Each mesh has two R,R
        # parameters and two R,S(0) parameters.
        model = nn.Module()
        model.dense_params = nn.ParameterList()
        model.sparse_params = nn.ParameterList()
        global_grads = []
        torch.manual_seed(42)
        second_axis_sharded_placements = (False, True)
        num_params_per_placement = 2

        for mesh, params in (
            (dense_mesh, model.dense_params),
            (sparse_mesh, model.sparse_params),
        ):
            assert mesh.mesh_dim_names is not None
            mesh_axes = tuple(
                spmd.MeshAxis.of(mesh.get_group(axis_name))
                for axis_name in mesh.mesh_dim_names
            )
            for second_axis_sharded in second_axis_sharded_placements:
                for _ in range(num_params_per_placement):
                    global_grad = torch.randn(8, 8, dtype=torch.float32)
                    axis_types = {
                        mesh_axes[0]: spmd.R,
                        mesh_axes[1]: (spmd.S(0) if second_axis_sharded else spmd.R),
                    }
                    local_grad = spmd_distribute_tensor(
                        global_grad.clone(), mesh, spmd.SpmdType(axis_types)
                    )

                    parameter = nn.Parameter(local_grad.clone())
                    with set_current_spmd_mesh(mesh):
                        spmd.assert_type(
                            parameter,
                            {
                                mesh.mesh_dim_names[0]: spmd.R,
                                mesh.mesh_dim_names[1]: (
                                    spmd.S(0) if second_axis_sharded else spmd.R
                                ),
                            },
                        )
                    parameter.grad = local_grad.clone()
                    params.append(parameter)
                    global_grads.append(global_grad)

        # compare against globally computed grad norm
        expected_norm = (
            torch.stack([global_grad.square().sum() for global_grad in global_grads])
            .sum()
            .sqrt()
        )
        # clip_grad_norm_spmd_ should only issue 1 all-reduce per mesh.
        expected_groups = [
            dense_mesh.get_group("dense_1"),
            sparse_mesh.get_group("sparse_1"),
        ]

        with patch.object(
            dist_utils.dist,
            "all_reduce",
            wraps=dist_utils.dist.all_reduce,
        ) as all_reduce:
            actual_norm = dist_utils.clip_grad_norm_spmd_(
                list(model.parameters()),
                max_norm=float("inf"),
                foreach=True,
            )

        self.assertEqual(actual_norm, expected_norm, rtol=1e-5, atol=1e-5)
        self.assertEqual(all_reduce.call_count, 2)
        self.assertEqual(
            [call.kwargs["group"] for call in all_reduce.call_args_list],
            expected_groups,
        )
