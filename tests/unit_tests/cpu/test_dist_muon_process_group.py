# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for DistMuon redistribution process groups."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.components.optim.optimizer import DistMuon
from torchtitan.distributed.flex_shard import (
    BucketConfig,
    DistMuon as FlexShardDistMuon,
)
from torchtitan.distributed.parallelism_context import ParallelismContext


def _parallelism_context() -> ParallelismContext:
    return ParallelismContext(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=1,
        enable_sequence_parallel=False,
    )


def test_redistribution_max_ctas_is_opt_in() -> None:
    config = DistMuon.Config(
        pattern=".*",
        compute_sharding_by_fqn={},
        bucket_configs=(),
    )

    assert config.redistribution_max_ctas is None


def test_redistribution_max_ctas_must_be_positive() -> None:
    common = {
        "pattern": ".*",
        "compute_sharding_by_fqn": {},
        "bucket_configs": (),
    }
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            DistMuon.Config(redistribution_max_ctas=value, **common)


@pytest.mark.parametrize("shard_axis", ["dp_shard", "edp_shard"])
def test_redistribution_mesh_splits_shard_axis_once(shard_axis: str) -> None:
    replicate_group = MagicMock(spec=dist.ProcessGroup)
    shard_group = MagicMock(spec=dist.ProcessGroup)
    child_group = MagicMock(spec=dist.ProcessGroup)
    storage_mesh = MagicMock(spec=DeviceMesh)
    storage_mesh.mesh_dim_names = ("dp_replicate", shard_axis)
    storage_mesh.mesh = torch.tensor([[0, 1]])
    storage_mesh.device_type = "cuda"
    storage_mesh.get_group.side_effect = {
        "dp_replicate": replicate_group,
        shard_axis: shard_group,
    }.__getitem__
    redistribution_mesh = MagicMock(spec=DeviceMesh)
    options = MagicMock()
    context = _parallelism_context()

    with (
        patch.object(dist, "get_backend", return_value=dist.Backend.NCCL),
        patch.object(dist, "get_world_size", return_value=2),
        patch.object(dist, "split_group", return_value=child_group) as split_group,
        patch.object(dist, "ProcessGroupNCCL", create=True) as process_group_nccl,
        patch.object(
            DeviceMesh,
            "from_group",
            return_value=redistribution_mesh,
        ) as from_group,
    ):
        process_group_nccl.Options.return_value = options
        first = context.get_redistribution_mesh(storage_mesh, max_ctas=8)
        second = context.get_redistribution_mesh(storage_mesh, max_ctas=8)

    assert first is redistribution_mesh
    assert second is redistribution_mesh
    assert options.config.max_ctas == 8
    assert options.config.split_share == 0
    split_group.assert_called_once_with(
        parent_pg=shard_group,
        split_ranks=[[0, 1]],
        pg_options=options,
        group_desc=f"optimizer_redistribution_{shard_axis}",
        backend="nccl",
    )
    from_group.assert_called_once_with(
        [replicate_group, child_group],
        "cuda",
        mesh=storage_mesh.mesh,
        mesh_dim_names=("dp_replicate", shard_axis),
    )


def test_fake_process_group_uses_storage_mesh(caplog) -> None:
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
    try:
        storage_mesh = init_device_mesh(
            "cpu",
            (2,),
            mesh_dim_names=("dp_shard",),
        )
        resolved = _parallelism_context().get_redistribution_mesh(
            storage_mesh,
            max_ctas=8,
        )
    finally:
        dist.destroy_process_group()

    assert resolved is storage_mesh
    assert "not modeled by the fake process group" in caplog.text


def test_non_nccl_process_group_is_rejected() -> None:
    parent_group = MagicMock(spec=dist.ProcessGroup)
    storage_mesh = MagicMock(spec=DeviceMesh)
    storage_mesh.mesh_dim_names = ("dp_shard",)
    storage_mesh.get_group.return_value = parent_group

    with (
        patch.object(dist, "get_backend", return_value=dist.Backend.GLOO),
        pytest.raises(ValueError, match="requires an NCCL process group"),
    ):
        _parallelism_context().get_redistribution_mesh(storage_mesh, max_ctas=8)


def test_dist_muon_attaches_redistribution_mesh_to_matching_bucket() -> None:
    storage_mesh = MagicMock(spec=DeviceMesh)
    redistribution_mesh = MagicMock(spec=DeviceMesh)
    parameter = MagicMock(spec=DTensor)
    parameter.device_mesh = storage_mesh
    parallelism_context = MagicMock(spec=ParallelismContext)
    parallelism_context.get_redistribution_mesh.return_value = redistribution_mesh
    bucket = BucketConfig(patterns=("layers.*.weight",))
    config = DistMuon.Config(
        pattern=".*",
        compute_sharding_by_fqn={},
        bucket_configs=(bucket,),
        redistribution_max_ctas=8,
    )

    with patch.object(FlexShardDistMuon, "__init__", return_value=None) as init:
        config.build_optimizer(
            params=[
                {
                    "params": [parameter],
                    "param_names": ["layers.0.weight"],
                }
            ],
            parallelism_context=parallelism_context,
        )

    parallelism_context.get_redistribution_mesh.assert_called_once_with(
        storage_mesh,
        max_ctas=8,
    )
    (resolved_bucket,) = init.call_args.kwargs["bucket_configs"]
    assert resolved_bucket.mesh is redistribution_mesh
    assert bucket.mesh is None
