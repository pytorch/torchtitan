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
        specifications = {
            "layers.0.blocks": (
                (6, 4),
                (Replicate(), Shard(0)),
                {"ep": BlockShard(dim=0, block_sizes=(4, 2))},
            ),
            "layers.0.experts": (
                (2, 8, 4),
                (Shard(0), Shard(0)),
                {"dp_shard": Shard(0), "ep": Shard(0)},
            ),
            "layers.0.native": (
                (3, 8, 4),
                (Replicate(), Shard(1)),
                {"ep": Shard(0)},
            ),
            "layers.0.owned": (
                (16, 4),
                (Replicate(), Shard(0)),
                {"ep": Owned()},
            ),
            "layers.0.small": (
                (1, 4, 4),
                (Replicate(), Shard(1)),
                {"ep": Shard(0)},
            ),
        }
        expected_assignments = {
            "layers.0.blocks": [(0, (0, 0), (4, 4)), (0, (4, 0), (2, 4))],
            "layers.0.native": [
                (1, (0, 0, 0), (2, 8, 4)),
                (1, (2, 0, 0), (1, 8, 4)),
            ],
            "layers.0.owned": [(0, (0, 0), (16, 4))],
            "layers.0.small": [(1, (0, 0, 0), (1, 4, 4))],
        }

        def region_view(tensor, region):
            return tensor[
                tuple(
                    slice(offset, offset + size)
                    for offset, size in zip(region.offsets, region.shape, strict=True)
                )
            ]

        for rank in (0, 1):
            with self.subTest(rank=rank):
                dist.init_process_group(
                    "fake", store=FakeStore(), rank=rank, world_size=4
                )
                try:
                    mesh = init_device_mesh(
                        "cpu", (2, 2), mesh_dim_names=("dp_shard", "ep")
                    )
                    params, layouts, local_regions = [], {}, {}
                    for name, (shape, placements, shardings) in specifications.items():
                        local_shape, offsets = compute_local_shape_and_global_offset(
                            shape, mesh, placements
                        )
                        local_regions[name] = (offsets, local_shape)
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
                        layouts[name] = ComputeLayout(
                            shardings, compute_dtype=torch.bfloat16
                        )

                    # CPU stream setup does not support the runtime's device argument.
                    with patch.object(
                        _BucketedRedistributionRuntime, "reserve_buffers"
                    ):
                        optimizer = DistMuon(
                            [{"params": params, "param_names": list(specifications)}],
                            compute_sharding_by_fqn=layouts,
                            bucket_configs=[BucketConfig(patterns=("layers.0.*",))],
                            ns_steps=5,
                        )
                    (bucket,) = optimizer._bucket_plans
                    self.assertEqual(bucket.group.participants, (0, 1))
                    self.assertEqual(
                        [item.fqn for item in bucket.unredistributed_items],
                        ["layers.0.experts"],
                    )
                    self.assertEqual(
                        local_regions["layers.0.experts"][1],
                        (1 if rank == 0 else 0, 8, 4),
                    )

                    # Fixed work is [1600, 0]. All chunks join LPT, including
                    # fewer, equal, and more logical units than the two ranks.
                    # Sibling chunks may share a rank without merging NS calls.
                    for item, plan in zip(
                        bucket.redistributed_items,
                        bucket.redistribution_plans,
                        strict=True,
                    ):
                        self.assertEqual(
                            [
                                (part.participant, region.offsets, region.shape)
                                for part in plan.compute_partitions
                                for region in part.logical_regions
                                if region.numel
                            ],
                            expected_assignments[item.fqn],
                        )
                        labels = torch.arange(
                            math.prod(plan.logical_shape), dtype=torch.float32
                        ).reshape(plan.logical_shape)
                        (storage_region,) = plan.storage_partition(rank).logical_regions
                        self.assertEqual(
                            (storage_region.offsets, storage_region.shape),
                            local_regions[item.fqn],
                        )
                        storage = {}
                        for part in plan.storage_partitions:
                            (region,) = part.logical_regions
                            storage[part.participant] = region_view(
                                labels, region
                            ).clone()
                            self.assertEqual(
                                tuple(storage[part.participant].shape),
                                part.tensor_shape,
                            )
                        compute = {
                            part.participant: torch.full(
                                part.tensor_shape, float("nan")
                            )
                            for part in plan.compute_partitions
                        }
                        for route in plan.storage_to_compute_routes:
                            source = region_view(
                                storage[route.source.participants[0]],
                                route.source.tensor_region,
                            )
                            for participant in route.destination.participants:
                                destination = region_view(
                                    compute[participant],
                                    route.destination.tensor_region,
                                )
                                destination.copy_(source.reshape(destination.shape))
                        for part in plan.compute_partitions:
                            regions = [
                                region_view(labels, region)
                                for region in part.logical_regions
                                if region.numel
                            ]
                            expected = (
                                torch.cat(regions, dim=0)
                                if regions
                                else torch.empty(part.tensor_shape)
                            )
                            torch.testing.assert_close(
                                compute[part.participant], expected, rtol=0, atol=0
                            )

                        views = optimizer._matrix_views_by_fqn[item.fqn]
                        if item.fqn == "layers.0.native" and rank == 1:
                            self.assertEqual(
                                [
                                    (tuple(view.shape), view.strides, view.offset)
                                    for view in views
                                ],
                                [
                                    ((2, 8, 4), (32, 4, 1), 0),
                                    ((1, 8, 4), (32, 4, 1), 64),
                                ],
                            )
                            for view, expected in zip(
                                views, (labels[:2], labels[2:]), strict=True
                            ):
                                torch.testing.assert_close(
                                    view.view_as_matrix_batch(compute[rank]),
                                    expected,
                                    rtol=0,
                                    atol=0,
                                )
                        if item.fqn == "layers.0.blocks" and rank == 0:
                            self.assertEqual(
                                [
                                    (tuple(view.shape), view.strides, view.offset)
                                    for view in views
                                ],
                                [
                                    ((1, 4, 4), (24, 4, 1), 0),
                                    ((1, 2, 4), (24, 4, 1), 16),
                                ],
                            )
                            for view, expected in zip(
                                views, (labels[:4], labels[4:]), strict=True
                            ):
                                torch.testing.assert_close(
                                    view.view_as_matrix_batch(compute[rank]),
                                    expected.unsqueeze(0),
                                    rtol=0,
                                    atol=0,
                                )

                        restored = {
                            participant: torch.full_like(tensor, float("nan"))
                            for participant, tensor in storage.items()
                        }
                        for route in plan.compute_to_storage_routes:
                            source = region_view(
                                compute[route.source.participants[0]],
                                route.source.tensor_region,
                            )
                            for participant in route.destination.participants:
                                destination = region_view(
                                    restored[participant],
                                    route.destination.tensor_region,
                                )
                                destination.copy_(source.reshape(destination.shape))
                        for participant, original in storage.items():
                            torch.testing.assert_close(
                                restored[participant],
                                original,
                                rtol=0,
                                atol=0,
                            )
                finally:
                    dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
