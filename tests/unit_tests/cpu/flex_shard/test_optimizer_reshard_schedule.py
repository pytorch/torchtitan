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
    _FQN = "layers.0.weight"

    def _make_1d_parameter(self) -> tuple[DeviceMesh, torch.nn.Parameter]:
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard",))
        parameter = torch.nn.Parameter(
            DTensor.from_local(
                torch.empty(6, 4),
                mesh,
                (Shard(0),),
                shape=torch.Size((12, 4)),
                stride=(4, 1),
            )
        )
        return mesh, parameter

    def _build_single_parameter_optimizer(
        self,
        parameter: torch.nn.Parameter,
        *,
        bucket_config: BucketConfig,
        block_sizes: tuple[int, ...] = (4,),
    ) -> DistMuon:
        with patch.object(_BucketedRedistributionRuntime, "reserve_buffers"):
            return DistMuon(
                [{"params": [parameter], "param_names": [self._FQN]}],
                compute_sharding_by_fqn={
                    self._FQN: ComputeLayout(
                        {
                            "dp_shard": BlockShard(
                                dim=0,
                                block_sizes=block_sizes,
                            )
                        }
                    )
                },
                bucket_configs=[bucket_config],
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
            DistMuon(
                [{"params": params, "param_names": names}],
                compute_sharding_by_fqn={
                    name: ComputeLayout(
                        {"dp_shard": BlockShard(dim=0, block_sizes=(matrix_rows,))}
                    )
                    for name in names
                },
                bucket_configs=[BucketConfig(patterns=(name,)) for name in names],
            )
        build_plan.assert_called_once()

    def test_dedicated_process_group_owns_redistribution_schedules(self):
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh(
            "cpu",
            (1, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        dedicated_group = dist.new_group(
            ranks=[0, 1],
            backend="fake",
            use_local_synchronization=True,
        )
        configured_mesh = DeviceMesh.from_group(
            [mesh.get_group("dp_replicate"), dedicated_group],
            "cpu",
            mesh=mesh.mesh,
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        parameter = torch.nn.Parameter(
            DTensor.from_local(
                torch.empty(6, 4),
                mesh,
                (Replicate(), Shard(0)),
                shape=torch.Size((12, 4)),
                stride=(4, 1),
            )
        )
        fqn = "layers.0.weight"

        with patch.object(_BucketedRedistributionRuntime, "reserve_buffers"):
            optimizer = DistMuon(
                [{"params": [parameter], "param_names": [fqn]}],
                compute_sharding_by_fqn={
                    fqn: ComputeLayout(
                        {"dp_shard": BlockShard(dim=0, block_sizes=(4,))}
                    )
                },
                bucket_configs=[BucketConfig(patterns=(fqn,), mesh=configured_mesh)],
            )
            optimizer.load_state_dict(optimizer.state_dict())

        (plan,) = optimizer._bucket_plans
        self.assertIsInstance(plan, _RedistributionBucketPlan)
        self.assertIs(plan.group.process_group, dedicated_group)
        self.assertIs(
            plan.storage_to_compute_schedule.process_group,
            dedicated_group,
        )
        self.assertIs(
            plan.compute_to_storage_schedule.process_group,
            dedicated_group,
        )

    def test_bucket_mesh_requires_matching_named_axis(self):
        mesh, parameter = self._make_1d_parameter()
        wrong_axis_mesh = DeviceMesh.from_group(
            [mesh.get_group()],
            "cpu",
            mesh=mesh.mesh,
            mesh_dim_names=("other",),
        )

        with self.assertRaisesRegex(ValueError, "missing redistribution axis"):
            self._build_single_parameter_optimizer(
                parameter,
                bucket_config=BucketConfig(
                    patterns=(self._FQN,),
                    mesh=wrong_axis_mesh,
                ),
            )

    def test_bucket_mesh_requires_matching_rank_order(self):
        mesh, parameter = self._make_1d_parameter()
        reversed_mesh = DeviceMesh.from_group(
            [mesh.get_group()],
            "cpu",
            mesh=mesh.mesh.flip(0),
            mesh_dim_names=("dp_shard",),
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            self._build_single_parameter_optimizer(
                parameter,
                bucket_config=BucketConfig(
                    patterns=(self._FQN,),
                    mesh=reversed_mesh,
                ),
            )

    def test_bucket_without_redistribution_warns(self):
        _mesh, parameter = self._make_1d_parameter()

        with self.assertLogs(
            "torchtitan.distributed.flex_shard._optimizer_reshard_schedule",
            level="WARNING",
        ) as logs:
            self._build_single_parameter_optimizer(
                parameter,
                bucket_config=BucketConfig(
                    patterns=(self._FQN,),
                    name="local",
                ),
                block_sizes=(6,),
            )

        self.assertIn("has no parameters requiring redistribution", logs.output[0])


if __name__ == "__main__":
    unittest.main()
