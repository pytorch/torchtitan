# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import _StridedShard
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


def _param(shape, mesh, *placements):
    local_shape, _ = compute_local_shape_and_global_offset(shape, mesh, placements)
    return torch.nn.Parameter(
        DTensor.from_local(
            torch.empty(local_shape),
            mesh,
            placements,
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

    def _build(self, params, layouts, *, patterns=("layers.0.*",)):
        names = list(layouts)
        # CPU stream setup does not support the runtime's device argument.
        with patch.object(_BucketedRedistributionRuntime, "reserve_buffers"):
            optimizer = DistMuon(
                [{"params": params, "param_names": names}],
                compute_sharding_by_fqn=layouts,
                bucket_configs=[
                    BucketConfig(patterns=(pattern,)) for pattern in patterns
                ],
            )
        return {
            item.fqn: plan
            for bucket in optimizer._bucket_plans
            for item, plan in zip(
                bucket.redistributed_items, bucket.redistribution_plans, strict=True
            )
        }

    def _layout(self, placement):
        return ComputeLayout({"dp_shard": placement}, compute_dtype=torch.bfloat16)

    def test_chunks_and_owned_tensors_spread_over_distinct_ranks(self):
        # Two [16, 8] chunks, one [8, 16] owned tensor, one smaller owned
        # tensor: four jobs for four ranks, each rank computes one matrix.
        shard0 = self._layout(Shard(0))
        owned = self._layout(Owned())
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

    def test_existing_shard_counts_and_order_are_preserved(self):
        for num_matrices, counts in (
            (4, [1, 1, 1, 1]),
            (6, [2, 2, 2, 0]),
            (8, [2, 2, 2, 2]),
        ):
            with self.subTest(num_matrices=num_matrices):
                layouts = {"layers.0.experts": self._layout(Shard(0))}
                params = [_param((num_matrices, 4, 4), self.mesh, Shard(1))]
                (plan,) = self._build(params, layouts).values()
                self.assertEqual(
                    [part.tensor_shape[0] for part in plan.compute_partitions], counts
                )

    def test_owned_tensor_avoids_ranks_with_compute_ready_load(self):
        # Six chunks over four ranks are compute-ready on ranks 0-2 (two
        # chunks each) and leave rank 3 idle, so the owned tensor goes there.
        layouts = {
            "layers.0.experts": self._layout(Shard(0)),
            "layers.0.wo": self._layout(Owned()),
        }
        params = [
            _param((6, 4, 4), self.mesh, Shard(0)),
            _param((8, 8), self.mesh, Shard(0)),
        ]
        plans = self._build(params, layouts)
        self.assertNotIn("layers.0.experts", plans)
        self.assertEqual(_compute_ranks(plans["layers.0.wo"]), (3,))

    def test_owned_tensor_accounts_for_repeated_sharding_of_one_dimension(self):
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp_shard", "ep"))
        layouts = {
            "layers.0.experts": ComputeLayout(
                {"dp_shard": Shard(0), "ep": Shard(0)},
                compute_dtype=torch.bfloat16,
            ),
            "layers.0.owned": ComputeLayout(
                {"ep": Owned()}, compute_dtype=torch.bfloat16
            ),
        }
        params = [
            _param((6, 4, 4), mesh, Shard(0), Shard(0)),
            _param((4, 4), mesh, Replicate(), Shard(0)),
        ]
        plans = self._build(params, layouts)
        # The dp_shard slice has three experts, split as two on rank 0 and
        # one on rank 1. The additional matrix belongs on rank 1.
        self.assertNotIn("layers.0.experts", plans)
        self.assertEqual(_compute_ranks(plans["layers.0.owned"]), (1,))

    def test_fixed_work_contributes_to_cross_bucket_ties(self):
        layouts = {
            "layers.0.experts": self._layout(Shard(0)),
            "layers.0.owned": self._layout(Owned()),
            "layers.1.owned": self._layout(Owned()),
        }
        params = [
            _param((6, 4, 4), self.mesh, Shard(0)),
            _param((4, 4), self.mesh, Shard(0)),
            _param((4, 4), self.mesh, Shard(0)),
        ]
        plans = self._build(params, layouts, patterns=("layers.0.*", "layers.1.*"))
        self.assertEqual(_compute_ranks(plans["layers.0.owned"]), (3,))
        self.assertEqual(_compute_ranks(plans["layers.1.owned"]), (3,))

    def test_compute_ready_strided_shards_contribute_their_actual_load(self):
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp_shard", "ep"))
        layouts = {
            "layers.0.experts": ComputeLayout(
                {"dp_shard": Shard(0), "ep": Shard(0)},
                shard_order_by_tensor_dim={0: ("ep", "dp_shard")},
                compute_dtype=torch.bfloat16,
            ),
            "layers.0.owned": ComputeLayout(
                {"dp_shard": Owned()}, compute_dtype=torch.bfloat16
            ),
        }
        params = [
            _param((6, 4, 4), mesh, _StridedShard(0, split_factor=2), Shard(0)),
            _param((4, 4), mesh, Shard(0), Replicate()),
        ]
        plans = self._build(params, layouts)
        # EP-first storage gives ranks 0/2 two/one experts respectively.
        self.assertNotIn("layers.0.experts", plans)
        self.assertEqual(_compute_ranks(plans["layers.0.owned"]), (2,))

    def test_block_shards_and_owned_tensors_share_the_assignment(self):
        for block_sizes in ((4,), (4, 2)):
            with self.subTest(block_sizes=block_sizes):
                num_rows = 8 if len(block_sizes) == 1 else 6
                layouts = {
                    "layers.0.blocks": self._layout(
                        BlockShard(dim=0, block_sizes=block_sizes)
                    ),
                    "layers.0.owned0": self._layout(Owned()),
                    "layers.0.owned1": self._layout(Owned()),
                }
                params = [
                    _param((num_rows, 4), self.mesh, Shard(0)),
                    _param((8, 4), self.mesh, Shard(0)),
                    _param((4, 4), self.mesh, Shard(0)),
                ]
                plans = self._build(params, layouts)
                self.assertEqual(
                    sorted(
                        rank for plan in plans.values() for rank in _compute_ranks(plan)
                    ),
                    [0, 1, 2, 3],
                )
                regions = sorted(
                    (region.offsets[0], region.shape[0])
                    for part in plans["layers.0.blocks"].compute_partitions
                    for region in part.logical_regions
                    if region.numel
                )
                self.assertEqual(regions, [(0, 4), (4, num_rows - 4)])

    def test_single_block_parameters_do_not_all_use_the_first_rank(self):
        layouts = {
            f"layers.0.matrix{index}": self._layout(BlockShard(dim=0, block_sizes=(8,)))
            for index in range(4)
        }
        params = [_param((8, 4), self.mesh, Shard(0)) for _ in layouts]
        plans = self._build(params, layouts)
        self.assertEqual(
            sorted(rank for plan in plans.values() for rank in _compute_ranks(plan)),
            [0, 1, 2, 3],
        )

    def test_fixed_sharding_preserves_noncanonical_mesh_order(self):
        mesh = DeviceMesh("cpu", [2, 0, 3, 1], mesh_dim_names=("dp_shard",))
        layouts = {"layers.0.experts": self._layout(Shard(0))}
        params = [_param((8, 4, 4), mesh, Shard(1))]
        (plan,) = self._build(params, layouts).values()
        self.assertEqual(
            {
                part.participant: part.logical_regions[0].offsets[0]
                for part in plan.compute_partitions
            },
            {2: 0, 0: 2, 3: 4, 1: 6},
        )


if __name__ == "__main__":
    unittest.main()
