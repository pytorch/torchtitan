# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torchtitan.components.distributed_optimizers.muon import DistributedMuon, Owned
from torchtitan.components.distributed_optimizers.muon_parameter_prep import (
    BatchedMatrixComputeView,
    build_distributed_muon,
    MuonComputeSharding,
)


class TestMuonParameterPrep(unittest.TestCase):
    def test_batched_matrix_view_validation(self):
        for num_matrices in (0, -1, True, 1.5):
            with self.subTest(num_matrices=num_matrices):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    BatchedMatrixComputeView(num_matrices, 0)
        for matrices_flattened_into_dim in (True, "0"):
            with self.subTest(
                matrices_flattened_into_dim=matrices_flattened_into_dim
            ):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    BatchedMatrixComputeView(3, matrices_flattened_into_dim)
        with self.assertRaisesRegex(
            ValueError, "only matrices_flattened_into_dim=0"
        ):
            BatchedMatrixComputeView(3, 1)

    def test_builder_compiles_layout_without_mutating_caller_group(self):
        view = BatchedMatrixComputeView(
            num_matrices=3, matrices_flattened_into_dim=0
        )
        compute_sharding = MuonComputeSharding(
            view_before_placement=view,
            placement=Shard(0),
        )
        storage = torch.arange(24).reshape(6, 4)
        other_storage = torch.empty(9, 5)
        group = {
            "params": [storage, other_storage],
            "param_names": [
                "layers.0.wq.weight",
                "layers.0.wkv_b.weight",
            ],
            "compute_sharding": compute_sharding,
        }
        identity_storage = torch.empty(4, 3)
        identity_group = {
            "params": [identity_storage],
            "param_names": ["layers.0.wkv_a.weight"],
            "compute_sharding": MuonComputeSharding(placement=Owned()),
        }
        bucket_spec = ()

        with mock.patch.object(DistributedMuon, "__init__", return_value=None) as init:
            optimizer = build_distributed_muon(
                [group, identity_group],
                bucket_spec=bucket_spec,
                lr=0.1,
            )

        self.assertIsInstance(optimizer, DistributedMuon)
        core_groups = init.call_args.args[0]
        prepared = init.call_args.kwargs["_prepared_compute_views"]
        init.assert_called_once_with(
            core_groups,
            bucket_spec=bucket_spec,
            _prepared_compute_views=prepared,
            lr=0.1,
        )
        self.assertIsNot(core_groups[0], group)
        self.assertIsNot(core_groups[1], identity_group)
        self.assertIs(group["compute_sharding"], compute_sharding)
        self.assertNotIn("compute_sharding", core_groups[0])
        self.assertEqual(core_groups[0]["_compute_placement"], Shard(0))
        self.assertEqual(core_groups[1]["_compute_placement"], Owned())
        self.assertFalse(any(value is view for value in core_groups[0].values()))
        self.assertEqual(
            prepared["layers.0.wq.weight"].global_compute_shape,
            torch.Size((3, 2, 4)),
        )
        self.assertEqual(
            prepared["layers.0.wq.weight"].local_compute_tensor.shape,
            torch.Size((3, 2, 4)),
        )
        self.assertEqual(
            prepared["layers.0.wkv_b.weight"].global_compute_shape,
            torch.Size((3, 3, 5)),
        )
        self.assertEqual(
            prepared["layers.0.wkv_b.weight"].local_compute_tensor.shape,
            torch.Size((3, 3, 5)),
        )
        self.assertEqual(
            prepared["layers.0.wq.weight"].local_compute_tensor.data_ptr(),
            storage.data_ptr(),
        )
        self.assertEqual(
            prepared["layers.0.wkv_a.weight"].global_compute_shape,
            identity_storage.shape,
        )
        self.assertEqual(
            prepared["layers.0.wkv_a.weight"].local_compute_tensor.shape,
            identity_storage.shape,
        )
        self.assertIs(
            prepared["layers.0.wkv_a.weight"].local_compute_tensor,
            identity_storage,
        )

    def test_builder_validates_global_shape_and_aligned_names(self):
        for shape, message in (
            ((2, 3, 4), "requires rank-2 storage"),
            ((5, 4), "is not divisible"),
        ):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, message):
                    build_distributed_muon(
                        [
                            {
                                "params": [torch.empty(shape)],
                                "param_names": ["layers.0.wq.weight"],
                                "compute_sharding": MuonComputeSharding(
                                    view_before_placement=BatchedMatrixComputeView(
                                        2, 0
                                    ),
                                    placement=Shard(0),
                                ),
                            }
                        ],
                        bucket_spec=(),
                    )

        with self.assertRaisesRegex(ValueError, "must be aligned"):
            build_distributed_muon(
                [
                    {
                        "params": [torch.empty(6, 4)],
                        "param_names": [],
                        "compute_sharding": MuonComputeSharding(
                            placement=Shard(0)
                        ),
                    }
                ],
                bucket_spec=(),
            )

    def test_builder_requires_compute_sharding(self):
        with self.assertRaisesRegex(TypeError, "named parameter groups"):
            build_distributed_muon([torch.empty(2, 2)], bucket_spec=())

        with self.assertRaisesRegex(TypeError, "DTensor parameters"):
            build_distributed_muon(
                [
                    {
                        "params": [torch.empty(2, 2)],
                        "param_names": ["weight"],
                        "compute_sharding": MuonComputeSharding(
                            placement=Owned()
                        ),
                    }
                ],
                bucket_spec=(),
            )

        with self.assertRaisesRegex(TypeError, "must be a MuonComputeSharding"):
            build_distributed_muon(
                [{"params": [], "param_names": [], "compute_sharding": object()}],
                bucket_spec=(),
            )

    def test_builder_rejects_strided_storage_shard_for_batched_matrices(self):
        param = mock.Mock(spec=DTensor)
        param.shape = torch.Size((6, 4))
        param.placements = (
            _StridedShard(0, split_factor=2),
            Shard(0),
        )

        with self.assertRaisesRegex(ValueError, "Shard or Replicate"):
            build_distributed_muon(
                [
                    {
                        "params": [param],
                        "param_names": ["layers.0.wq.weight"],
                        "compute_sharding": MuonComputeSharding(
                            view_before_placement=BatchedMatrixComputeView(3),
                            placement=Shard(0),
                        ),
                    }
                ],
                bucket_spec=(),
            )


if __name__ == "__main__":
    unittest.main()
