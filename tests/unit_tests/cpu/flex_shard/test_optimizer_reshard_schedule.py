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
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    build_dist_muon,
    ComputeLayout,
    dist_muon,
)
from torchtitan.distributed.flex_shard._optimizer_reshard_runtime import (
    _BucketedRedistributionRuntime,
    _tensor_region_view,
)
from torchtitan.distributed.flex_shard._optimizer_reshard_schedule import (
    _RedistributionBucketPlan,
)


class TestMuonPlanConstruction(unittest.TestCase):
    def test_native_row_shards_route_complete_matrices(self):
        for mesh_shape, rank, matrix_rows in (
            ((2,), 0, 8),
            # Rank 2 stores rows but owns no complete matrix for compute.
            ((4,), 2, 5),
            ((2, 2), 2, 5),
        ):
            with self.subTest(
                mesh_shape=mesh_shape, rank=rank, matrix_rows=matrix_rows
            ):
                dist.init_process_group(
                    "fake",
                    store=FakeStore(),
                    rank=rank,
                    world_size=math.prod(mesh_shape),
                )
                try:
                    mesh_axis_names = (
                        ("dp_replicate", "dp_shard")
                        if len(mesh_shape) == 2
                        else ("dp_shard",)
                    )
                    mesh = init_device_mesh(
                        "cpu", mesh_shape, mesh_dim_names=mesh_axis_names
                    )
                    placements = (
                        (Replicate(), Shard(1)) if len(mesh_shape) == 2 else (Shard(1),)
                    )
                    num_participants = mesh_shape[-1]
                    rows_per_shard = (
                        matrix_rows + num_participants - 1
                    ) // num_participants
                    value = (
                        torch.arange(2 * matrix_rows * 3)
                        .reshape(2, matrix_rows, 3)
                        .float()
                    )
                    storage = tuple(
                        value[
                            :, index * rows_per_shard : (index + 1) * rows_per_shard
                        ].clone()
                        for index in range(num_participants)
                    )
                    parameter = torch.nn.Parameter(
                        DTensor.from_local(
                            storage[mesh["dp_shard"].get_local_rank()],
                            mesh,
                            placements,
                            shape=value.shape,
                            stride=value.stride(),
                        )
                    )
                    # CPU stream setup does not support the runtime's device argument.
                    with patch.object(
                        _BucketedRedistributionRuntime, "reserve_buffers"
                    ):
                        optimizer = build_dist_muon(
                            [{"params": [parameter], "param_names": ["w13.weight"]}],
                            compute_sharding_by_fqn={
                                "w13.weight": ComputeLayout({"dp_shard": Shard(0)})
                            },
                            bucket_configs=[BucketConfig(patterns=("w13.weight",))],
                        )
                    layout = optimizer._parameter_compute_layouts[0]
                    self.assertIsNone(layout.compute_view)
                    self.assertEqual(layout.global_compute_shape, value.shape)
                    self.assertEqual(layout.compute_sharding, Shard(0))
                    self.assertFalse(layout.storage_is_compute_ready)
                    bucket = optimizer._bucket_plans[0]
                    assert isinstance(bucket, _RedistributionBucketPlan)
                    plan = bucket.redistribution_plans[0]
                    self.assertEqual(plan.logical_shape, tuple(value.shape))
                    compute = tuple(
                        value[index : index + 1] for index in range(num_participants)
                    )
                    for sources, expected, partitions, routes in (
                        (
                            storage,
                            compute,
                            plan.compute_partitions,
                            plan.storage_to_compute_routes,
                        ),
                        (
                            compute,
                            storage,
                            plan.storage_partitions,
                            plan.compute_to_storage_routes,
                        ),
                    ):
                        actual = {
                            partition.participant: torch.zeros(partition.tensor_shape)
                            for partition in partitions
                        }
                        coverage = {
                            participant: torch.zeros_like(tensor)
                            for participant, tensor in actual.items()
                        }
                        source_by_participant = dict(
                            zip(plan.participants, sources, strict=True)
                        )
                        for route in routes:
                            (source,) = route.source.participants
                            (destination,) = route.destination.participants
                            _tensor_region_view(
                                actual[destination], route.destination.tensor_region
                            ).copy_(
                                _tensor_region_view(
                                    source_by_participant[source],
                                    route.source.tensor_region,
                                )
                            )
                            _tensor_region_view(
                                coverage[destination], route.destination.tensor_region
                            ).add_(1)
                        for participant, expected_tensor in zip(
                            plan.participants, expected, strict=True
                        ):
                            torch.testing.assert_close(
                                actual[participant], expected_tensor, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                coverage[participant],
                                torch.ones_like(expected_tensor),
                                rtol=0,
                                atol=0,
                            )
                    local_index = plan.participants.index(rank)
                    self.assertEqual(
                        bucket.storage_to_compute_schedule.output_buffer_numel,
                        compute[local_index].numel(),
                    )
                    self.assertEqual(
                        bucket.compute_to_storage_schedule.output_buffer_numel,
                        storage[local_index].numel(),
                    )
                finally:
                    dist.destroy_process_group()

    def test_native_compute_shards_preserve_complete_matrices(self):
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard",))
        parameter = DTensor.from_local(
            torch.empty(2, 3, 4),
            mesh,
            (Shard(1),),
            shape=torch.Size((2, 6, 4)),
            stride=(24, 4, 1),
        )
        for compute_dim in (1, 2):
            with self.subTest(compute_dim=compute_dim):
                with self.assertRaisesRegex(
                    ValueError, "sharded only on tensor dimension 0"
                ):
                    dist_muon._resolve_storage_to_compute_transition(
                        "w13.weight",
                        parameter,
                        parameter.shape,
                        None,
                        ComputeLayout({"dp_shard": Shard(compute_dim)}),
                    )

    def test_native_column_to_batch_sharding_remains_unsupported(self):
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard",))
        parameter = DTensor.from_local(
            torch.empty(2, 6, 2),
            mesh,
            (Shard(2),),
            shape=torch.Size((2, 6, 4)),
            stride=(24, 4, 1),
        )
        with self.assertRaisesRegex(
            NotImplementedError, "cannot yet change tensor sharding"
        ):
            dist_muon._resolve_storage_to_compute_transition(
                "w13.weight",
                parameter,
                parameter.shape,
                None,
                ComputeLayout({"dp_shard": Shard(0)}),
            )

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
            build_dist_muon(
                [{"params": params, "param_names": names}],
                compute_sharding_by_fqn={
                    name: ComputeLayout(
                        {"dp_shard": BlockShard(dim=0, block_size=matrix_rows)}
                    )
                    for name in names
                },
                bucket_configs=[BucketConfig(patterns=(name,)) for name in names],
            )
        build_plan.assert_called_once()


if __name__ == "__main__":
    unittest.main()
