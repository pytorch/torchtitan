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
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    dist_muon,
    DistMuon,
)
from torchtitan.distributed.flex_shard._optimizer_reshard_runtime import (
    _BucketedRedistributionRuntime,
)
from torchtitan.distributed.flex_shard._optimizer_reshard_schedule import (
    _RedistributionBucketPlan,
)


class TestMuonPlanConstruction(unittest.TestCase):
    def test_singleton_expert_shards_balance_within_bucket(self):
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=8)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("edp_shard", "ep"))
        names = []
        params = []
        for layer in (1, 2):
            for projection, shape in (
                ("w13", (4, 2, 4, 4)),
                ("w2", (4, 4, 2)),
            ):
                local_shape = (1, *shape[1:-2], 2, shape[-1])
                global_tensor = torch.empty(shape)
                params.append(
                    torch.nn.Parameter(
                        DTensor.from_local(
                            torch.empty(local_shape),
                            mesh,
                            (Shard(len(shape) - 2), Shard(0)),
                            shape=global_tensor.shape,
                            stride=global_tensor.stride(),
                        )
                    )
                )
                names.append(f"layers.{layer}.routed_experts.{projection}")
        with patch.object(_BucketedRedistributionRuntime, "reserve_buffers"):
            optimizer = DistMuon(
                [{"params": params, "param_names": names}],
                compute_sharding_by_fqn={
                    name: ComputeLayout(
                        {"edp_shard": Shard(0), "ep": Shard(0)},
                        shard_order_by_tensor_dim={0: ("ep", "edp_shard")},
                        compute_dtype=torch.bfloat16,
                    )
                    for name in names
                },
                bucket_configs=[BucketConfig(patterns=tuple(names))],
            )

        (bucket,) = optimizer._bucket_plans
        assert isinstance(bucket, _RedistributionBucketPlan)
        costs = {rank: 0 for rank in bucket.group.participants}
        plans_by_projection = {}
        for item, plan in zip(
            bucket.redistributed_items, bucket.redistribution_plans, strict=True
        ):
            owners = [
                rank
                for rank in bucket.group.participants
                if math.prod(plan.compute_partition(rank).tensor_shape)
            ]
            self.assertEqual(len(owners), 1)
            (owner,) = owners
            compute_shape = plan.compute_partition(owner).tensor_shape
            self.assertEqual(compute_shape, (1, *item.param.shape[1:]))
            costs[owner] += dist_muon._estimate_muon_compute_cost(
                torch.Size(compute_shape), optimizer.param_groups[0]["ns_steps"]
            )
            self.assertEqual(
                {
                    rank
                    for route in plan.storage_to_compute_routes
                    for rank in route.destination.participants
                },
                {owner},
            )
            plans_by_projection.setdefault(item.fqn.rsplit(".", 1)[1], []).append(plan)
        self.assertEqual(set(costs), {0, 4})
        self.assertEqual(costs[0], costs[4])
        for plans in plans_by_projection.values():
            # Equal-shaped layers assigned to different ranks need distinct plans.
            self.assertIsNot(plans[0], plans[1])

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
            DistMuon(
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
        build_plan.assert_called_once()


if __name__ == "__main__":
    unittest.main()
