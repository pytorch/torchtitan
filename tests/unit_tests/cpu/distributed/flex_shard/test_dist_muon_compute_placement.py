# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
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
    def test_joint_placement_preserves_chunks_and_accounts_for_fixed_work(self):
        dist.init_process_group("fake", store=FakeStore(), rank=1, world_size=4)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp_shard", "ep"))
        specifications = {
            "experts": (
                (2, 16, 8),
                (Shard(0), Shard(0)),
                {"dp_shard": Shard(0), "ep": Shard(0)},
            ),
            "native": ((3, 8, 4), (Replicate(), Shard(1)), {"ep": Shard(0)}),
            "blocks": (
                (6, 4),
                (Replicate(), Shard(0)),
                {"ep": BlockShard(dim=0, block_sizes=(4, 2))},
            ),
            "owned": ((16, 4), (Replicate(), Shard(0)), {"ep": Owned()}),
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
                bucket_configs=[BucketConfig(patterns=("*",))],
                ns_steps=5,
            )
        (bucket,) = optimizer._bucket_plans
        self.assertEqual(bucket.group.participants, (0, 1))
        self.assertEqual(
            [item.fqn for item in bucket.unredistributed_items], ["experts"]
        )

        # Repeated Shard gives fixed costs [12800, 0], not equal costs.
        # All movable work costs 8840, so every original chunk goes to rank 1.
        expected_views = {
            "native": [((2, 8, 4), 0), ((1, 8, 4), 64)],
            "blocks": [((1, 4, 4), 0), ((1, 2, 4), 16)],
            "owned": [((16, 4), 0)],
        }
        for item, plan in zip(
            bucket.redistributed_items, bucket.redistribution_plans, strict=True
        ):
            self.assertEqual(plan.compute_partition(0).logical_regions, ())
            partition = plan.compute_partition(1)
            self.assertEqual(partition.tensor_shape, plan.logical_shape)

            # With all chunks on one rank, packed and logical addresses agree.
            for route in plan.storage_to_compute_routes:
                self.assertEqual(route.destination.participants, (1,))
                self.assertEqual(route.destination.tensor_region, route.logical_region)

            views = optimizer._matrix_views_by_fqn[item.fqn]
            self.assertEqual(
                [(tuple(view.shape), view.offset) for view in views],
                expected_views[item.fqn],
            )
            packed = torch.arange(
                math.prod(plan.logical_shape), dtype=torch.float32
            ).reshape(plan.logical_shape)
            for view, region in zip(views, partition.logical_regions, strict=True):
                start = region.offsets[0]
                torch.testing.assert_close(
                    view.view_as_matrix_batch(packed).reshape(region.shape),
                    packed[start : start + region.shape[0]],
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
