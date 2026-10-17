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
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    DistMuon,
    Owned,
)
from torchtitan.distributed.flex_shard._optimizer_reshard_runtime import (
    _BucketedRedistributionRuntime,
)


class TestComputePlacement(unittest.TestCase):
    def test_joint_placement_accounts_for_fixed_local_work(self):
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=8)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp_shard", "ep"))
        specifications = {
            "layers.0.blocks": (
                (12, 8),
                (Replicate(), Shard(0)),
                {"ep": BlockShard(dim=0, block_sizes=(8, 4))},
            ),
            "layers.0.experts": (
                (10, 4, 4),
                (Shard(0), Shard(0)),
                {"dp_shard": Shard(0), "ep": Shard(0)},
            ),
            "layers.0.native": (
                (1, 8, 8),
                (Replicate(), Shard(1)),
                {"ep": Shard(0)},
            ),
            "layers.0.owned": (
                (8, 8),
                (Replicate(), Shard(0)),
                {"ep": Owned()},
            ),
        }
        params, layouts = [], {}
        for name, (shape, placements, shardings) in specifications.items():
            local_shape, _ = compute_local_shape_and_global_offset(
                shape, mesh, placements
            )
            params.append(
                torch.nn.Parameter(
                    DTensor.from_local(
                        torch.empty(local_shape),
                        mesh,
                        placements,
                        shape=torch.Size(shape),
                        stride=torch.empty(shape).stride(),
                    )
                )
            )
            layouts[name] = ComputeLayout(shardings, compute_dtype=torch.bfloat16)

        # CPU stream setup does not support the runtime's device argument.
        with patch.object(_BucketedRedistributionRuntime, "reserve_buffers"):
            optimizer = DistMuon(
                [{"params": params, "param_names": list(specifications)}],
                compute_sharding_by_fqn=layouts,
                bucket_configs=[BucketConfig(patterns=("layers.0.*",))],
                ns_steps=5,
            )
        (bucket,) = optimizer._bucket_plans
        self.assertEqual(bucket.group.participants, (0, 1, 2, 3))
        self.assertEqual(
            [item.fqn for item in bucket.unredistributed_items],
            ["layers.0.experts"],
        )
        assignments = {
            item.fqn: [
                (part.participant, region.offsets, region.shape)
                for part in plan.compute_partitions
                for region in part.logical_regions
                if region.numel
            ]
            for item, plan in zip(
                bucket.redistributed_items, bucket.redistribution_plans, strict=True
            )
        }
        # Repeated sharding leaves [2, 2, 1, 0] fixed experts on ranks 0-3.
        # Equal-cost jobs go to ranks 3, 2, 0; the smaller block goes to rank 1.
        # The original planner put all three large jobs on rank 0.
        self.assertEqual(assignments["layers.0.owned"], [(0, (0, 0), (8, 8))])
        self.assertEqual(assignments["layers.0.native"], [(2, (0, 0, 0), (1, 8, 8))])
        self.assertEqual(
            assignments["layers.0.blocks"],
            [(1, (8, 0), (4, 8)), (3, (0, 0), (8, 8))],
        )


if __name__ == "__main__":
    unittest.main()
