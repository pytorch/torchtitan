# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    build_dist_muon,
    ComputeLayout,
    dist_muon,
)
from torchtitan.distributed.flex_shard._optimizer_reshard_runtime import (
    _BucketedRedistributionRuntime,
)


class TestMuonPlanConstruction(unittest.TestCase):
    def test_identical_layers_build_one_plan(self):
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard",))
        num_layers = 3
        num_matrices = 3
        matrix_rows = matrix_cols = 4
        names = [f"layers.{layer}.weight" for layer in range(num_layers)]
        # Six-row storage shards split the middle four-row matrix, requiring
        # the same redistribution plan for each layer.
        params = [
            torch.nn.Parameter(
                DTensor.from_local(
                    torch.empty(num_matrices * matrix_rows // 2, matrix_cols),
                    mesh,
                    (Shard(0),),
                    shape=torch.Size((num_matrices * matrix_rows, matrix_cols)),
                    stride=(matrix_cols, 1),
                )
            )
            for _ in names
        ]

        with (
            # CPU stream setup does not support the runtime's device argument.
            patch.object(_BucketedRedistributionRuntime, "reserve_buffers"),
            patch.object(
                dist_muon,
                "_build_parameter_redistribution_plan",
                wraps=dist_muon._build_parameter_redistribution_plan,
            ) as build_plan,
        ):
            build_dist_muon(
                [{"params": params, "param_names": names}],
                compute_sharding_by_fqn={
                    name: ComputeLayout(
                        {"dp_shard": BlockShard(dim=0, block_size=matrix_rows)}
                    )
                    for name in names
                },
                bucket_configs=[BucketConfig(patterns=(name,)) for name in names],
            )
        build_plan.assert_called_once()


if __name__ == "__main__":
    unittest.main()
