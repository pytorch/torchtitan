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
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
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
    def test_fixed_work_changes_placement_and_plan_reuse(self):
        dist.init_process_group("fake", store=FakeStore(), rank=1, world_size=2)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard",))
        num_layers = 3
        params, names, layouts = [], [], {}
        for layer in range(num_layers):
            specifications = [("weight", (6, 4), BlockShard(dim=0, block_sizes=(4, 2)))]
            if layer != 1:
                specifications.append(("fixed", (1, 8, 4), Shard(0)))
            for suffix, shape, sharding in specifications:
                name = f"layers.{layer}.{suffix}"
                local_shape, _ = compute_local_shape_and_global_offset(
                    shape, mesh, (Shard(0),)
                )
                params.append(
                    torch.nn.Parameter(
                        DTensor.from_local(
                            torch.empty(local_shape),
                            mesh,
                            (Shard(0),),
                            shape=torch.Size(shape),
                            stride=torch.empty(shape).stride(),
                        )
                    )
                )
                names.append(name)
                layouts[name] = ComputeLayout(
                    {"dp_shard": sharding}, compute_dtype=torch.bfloat16
                )

        # CPU stream setup does not support the runtime's device argument.
        with patch.object(_BucketedRedistributionRuntime, "reserve_buffers"):
            optimizer = DistMuon(
                [{"params": params, "param_names": names}],
                compute_sharding_by_fqn=layouts,
                bucket_configs=[
                    BucketConfig(patterns=(f"layers.{layer}.*",))
                    for layer in range(num_layers)
                ],
                ns_steps=5,
            )
        plans = [bucket.redistribution_plans[0] for bucket in optimizer._bucket_plans]
        # Fixed work [1600, 0] exceeds both chunks together (960 + 200).
        # Layers 0 and 2 put both chunks on rank 1; layer 1 uses both ranks.
        self.assertIs(plans[0], plans[2])
        self.assertIsNot(plans[0], plans[1])
        self.assertEqual(plans[0].storage_partitions, plans[1].storage_partitions)
        self.assertEqual(plans[0].compute_partition(0).logical_regions, ())
        self.assertEqual(plans[0].compute_partition(1).tensor_shape, (6, 4))
        self.assertEqual(
            [
                (tuple(view.shape), view.offset)
                for view in optimizer._matrix_views_by_fqn["layers.0.weight"]
            ],
            [((1, 4, 4), 0), ((1, 2, 4), 16)],
        )


if __name__ == "__main__":
    unittest.main()
