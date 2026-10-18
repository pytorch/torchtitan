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
    ComputeLayout,
    DistMuon,
)
from torchtitan.distributed.flex_shard._optimizer_reshard_runtime import (
    _BucketedRedistributionRuntime,
)


class TestMuonPlanConstruction(unittest.TestCase):
    def test_identical_placements_reuse_plans(self):
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard",))
        num_layers = 3
        num_matrices = 3
        matrix_rows = matrix_cols = 4
        names = [f"layers.{layer}.weight" for layer in range(num_layers)]
        # The same chunk geometry is reused across layers, but cumulative work
        # can change which rank receives the larger chunk.
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

        # CPU stream setup does not support the runtime's device argument.
        with patch.object(_BucketedRedistributionRuntime, "reserve_buffers"):
            optimizer = DistMuon(
                [{"params": params, "param_names": names}],
                compute_sharding_by_fqn={
                    name: ComputeLayout(
                        {"dp_shard": BlockShard(dim=0, block_sizes=(matrix_rows,))},
                        compute_dtype=torch.bfloat16,
                    )
                    for name in names
                },
                bucket_configs=[BucketConfig(patterns=(name,)) for name in names],
            )
        plans = [bucket.redistribution_plans[0] for bucket in optimizer._bucket_plans]
        self.assertIs(plans[0], plans[2])
        self.assertIsNot(plans[0], plans[1])
        self.assertEqual(plans[0].storage_partitions, plans[1].storage_partitions)
        self.assertEqual(
            plans[0].compute_partition(0).logical_regions,
            plans[1].compute_partition(1).logical_regions,
        )


if __name__ == "__main__":
    unittest.main()
