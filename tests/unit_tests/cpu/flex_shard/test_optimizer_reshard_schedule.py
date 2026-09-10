# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard

from torchtitan.distributed.flex_shard._optimizer_reshard_schedule import (
    _build_dim0_shard_redistribution_plan,
    _build_owned_redistribution_plan,
    _dtensor_storage_regions,
    _TensorRegion,
)


class TestHSDPStorageRegions(unittest.TestCase):
    @staticmethod
    def make_storage(shape, placements, ranks):
        # Planning needs only DTensor metadata; no process group or device is used.
        tensor = mock.Mock(spec=DTensor)
        tensor.shape = torch.Size(shape)
        tensor.ndim = len(shape)
        tensor.placements = placements
        tensor.device_mesh.mesh = ranks
        tensor.device_mesh.shape = ranks.shape
        tensor.device_mesh.ndim = ranks.ndim
        tensor.device_mesh.size.side_effect = ranks.size
        return tensor

    def test_owned_cartesian_group_restores_all_storage_holders(self):
        ranks = torch.tensor([[3, 1], [2, 0]])
        tensor = self.make_storage((5, 3), (Replicate(), Shard(0)), ranks)
        participants = (0, 1, 2, 3)
        shape, regions = _dtensor_storage_regions(
            tensor, participants, required_storage_mesh_axes=(0, 1)
        )
        self.assertEqual(shape, (5, 3))
        self.assertEqual(
            dict(regions),
            {
                (0, 1): _TensorRegion((3, 0), (2, 3)),
                (2, 3): _TensorRegion((0, 0), (3, 3)),
            },
        )
        plan = _build_owned_redistribution_plan(
            regions, participants=participants, owner_rank=1, logical_shape=shape
        )
        self.assertEqual(plan.compute_partition(1).tensor_shape, shape)
        for rank in (0, 2, 3):
            self.assertEqual(plan.compute_partition(rank).tensor_shape, (0,))
        self.assertEqual(
            {
                rank
                for route in plan.compute_to_storage_routes
                for rank in route.destination.participants
            },
            set(participants),
        )

    def test_replica_sharding_preserves_uneven_and_empty_storage_shards(self):
        ranks = torch.tensor([[3, 1], [2, 0]])
        for num_matrices in (5, 1):
            tensor = self.make_storage(
                (num_matrices, 4, 3), (Replicate(), Shard(0)), ranks
            )
            for shard_coordinate, participants in enumerate(((2, 3), (0, 1))):
                with self.subTest(num_matrices=num_matrices, shard=shard_coordinate):
                    shape, regions = _dtensor_storage_regions(
                        tensor, participants, required_storage_mesh_axes=(0,)
                    )
                    expected_size = (
                        (num_matrices + 1) // 2
                        if shard_coordinate == 0
                        else num_matrices // 2
                    )
                    self.assertEqual(shape, (expected_size, 4, 3))
                    self.assertEqual(
                        regions, ((participants, _TensorRegion((0, 0, 0), shape)),)
                    )
                    plan = _build_dim0_shard_redistribution_plan(
                        regions,
                        participants=participants,
                        shard_participants=tuple(reversed(participants)),
                        logical_shape=shape,
                    )
                    self.assertEqual(
                        sum(part.tensor_shape[0] for part in plan.compute_partitions),
                        expected_size,
                    )
                    # Rank ordering follows mesh coordinates, not sorted global ranks.
                    self.assertEqual(
                        plan.compute_partition(participants[-1]).tensor_shape[0],
                        (expected_size + 1) // 2,
                    )

    def test_multi_axis_group_rejects_incomplete_participants(self):
        tensor = self.make_storage(
            (5, 3), (Replicate(), Shard(0)), torch.arange(4).reshape(2, 2)
        )
        with self.assertRaisesRegex(ValueError, "participants do not match"):
            _dtensor_storage_regions(tensor, (0, 1), required_storage_mesh_axes=(0, 1))

    def test_multi_axis_group_preserves_outer_replicated_axis(self):
        tensor = self.make_storage(
            (5, 3),
            (Replicate(), Replicate(), Shard(0)),
            torch.arange(8).reshape(2, 2, 2),
        )
        shape, regions = _dtensor_storage_regions(
            tensor, (4, 5, 6, 7), required_storage_mesh_axes=(1, 2)
        )
        self.assertEqual(shape, (5, 3))
        self.assertEqual(
            {rank for holders, _ in regions for rank in holders}, {4, 5, 6, 7}
        )


if __name__ == "__main__":
    unittest.main()
