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
    BucketConfig,
    ComputeLayout,
    DistMuon,
    Owned,
)
from torchtitan.distributed.flex_shard._optimizer_reshard_runtime import (
    _BucketedRedistributionRuntime,
)


def _param(shape, mesh, placement):
    local_shape = list(shape)
    local_shape[placement.dim] //= mesh.size()
    return torch.nn.Parameter(
        DTensor.from_local(
            torch.empty(local_shape),
            mesh,
            (placement,),
            shape=torch.Size(shape),
            stride=torch.empty(shape).stride(),
        )
    )


def _compute_ranks(plan):
    return tuple(
        partition.participant
        for partition in plan.compute_partitions
        if partition.tensor_shape and all(partition.tensor_shape)
    )


class TestComputePlacement(unittest.TestCase):
    def setUp(self):
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)
        self.addCleanup(dist.destroy_process_group)
        self.mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("dp_shard",))

    def _build(self, params, layouts):
        names = list(layouts)
        # CPU stream setup does not support the runtime's device argument.
        with patch.object(_BucketedRedistributionRuntime, "reserve_buffers"):
            optimizer = DistMuon(
                [{"params": params, "param_names": names}],
                compute_sharding_by_fqn=layouts,
                bucket_configs=[BucketConfig(patterns=("layers.0.*",))],
            )
        (bucket,) = optimizer._bucket_plans
        return dict(
            zip(
                (item.fqn for item in bucket.redistributed_items),
                bucket.redistribution_plans,
            )
        )

    def test_chunks_and_owned_tensors_spread_over_distinct_ranks(self):
        # Two [16, 8] chunks, one [8, 16] owned tensor, one smaller owned
        # tensor: four jobs for four ranks, each rank computes one matrix.
        shard0 = ComputeLayout({"dp_shard": Shard(0)})
        owned = ComputeLayout({"dp_shard": Owned()})
        layouts = {
            "layers.0.w13": shard0,
            "layers.0.w2": owned,
            "layers.0.wo": owned,
        }
        params = [
            _param((2, 16, 8), self.mesh, Shard(1)),
            _param((8, 16), self.mesh, Shard(0)),
            _param((8, 8), self.mesh, Shard(0)),
        ]
        plans = self._build(params, layouts)
        w13_ranks = _compute_ranks(plans["layers.0.w13"])
        self.assertEqual(len(w13_ranks), 2)
        owners = (
            *w13_ranks,
            *_compute_ranks(plans["layers.0.w2"]),
            *_compute_ranks(plans["layers.0.wo"]),
        )
        self.assertEqual(sorted(owners), [0, 1, 2, 3])

    def test_fully_populated_sharding_keeps_storage_order(self):
        layouts = {"layers.0.experts": ComputeLayout({"dp_shard": Shard(0)})}
        params = [_param((8, 4, 4), self.mesh, Shard(1))]
        (plan,) = self._build(params, layouts).values()
        offsets = {
            partition.participant: partition.logical_regions[0].offsets[0]
            for partition in plan.compute_partitions
        }
        self.assertEqual(offsets, {0: 0, 1: 2, 2: 4, 3: 6})

    def test_owned_tensor_avoids_ranks_with_compute_ready_load(self):
        # Six chunks over four ranks are compute-ready on ranks 0-2 (two
        # chunks each) and leave rank 3 idle, so the owned tensor goes there.
        layouts = {
            "layers.0.experts": ComputeLayout({"dp_shard": Shard(0)}),
            "layers.0.wo": ComputeLayout({"dp_shard": Owned()}),
        }
        params = [
            torch.nn.Parameter(
                DTensor.from_local(
                    torch.empty(2, 4, 4),
                    self.mesh,
                    (Shard(0),),
                    shape=torch.Size((6, 4, 4)),
                    stride=(16, 4, 1),
                )
            ),
            _param((8, 8), self.mesh, Shard(0)),
        ]
        plans = self._build(params, layouts)
        self.assertNotIn("layers.0.experts", plans)
        self.assertEqual(_compute_ranks(plans["layers.0.wo"]), (3,))


if __name__ == "__main__":
    unittest.main()
