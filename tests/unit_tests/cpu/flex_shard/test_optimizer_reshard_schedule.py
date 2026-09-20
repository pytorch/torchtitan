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
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    build_dist_muon,
    ComputeLayout,
    dist_muon,
    Owned,
)
from torchtitan.distributed.flex_shard._optimizer_reshard_runtime import (
    _BucketedRedistributionRuntime,
)


class TestMuonPlanConstruction(unittest.TestCase):
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


class TestNativeMatrixViewBinding(unittest.TestCase):
    def _make_mesh(self, mesh_shape, mesh_axis_names, *, rank=0):
        dist.init_process_group(
            "fake",
            store=FakeStore(),
            rank=rank,
            world_size=torch.Size(mesh_shape).numel(),
        )
        self.addCleanup(dist.destroy_process_group)
        return init_device_mesh("cpu", mesh_shape, mesh_dim_names=mesh_axis_names)

    def _make_optimizer(self, parameters, compute_layouts):
        with patch.object(_BucketedRedistributionRuntime, "reserve_buffers"):
            return build_dist_muon(
                [
                    {
                        "params": list(parameters.values()),
                        "param_names": list(parameters),
                    }
                ],
                compute_sharding_by_fqn=compute_layouts,
                bucket_configs=[BucketConfig(patterns=("layers.0.*",))],
            )

    def _assert_identity_view(self, optimizer, layout, compute):
        views = optimizer._matrix_views_by_fqn[layout.fqn]
        if compute.numel() == 0:
            self.assertEqual(views, ())
            return
        (view,) = views
        matrix_batch = view.view_as_matrix_batch(compute)
        self.assertEqual(matrix_batch.shape, compute.shape)
        torch.testing.assert_close(matrix_batch, compute, rtol=0, atol=0)
        self.assertEqual(matrix_batch.data_ptr(), compute.data_ptr())

    def test_local_owned_view_preserves_matrix_rank(self):
        mesh = self._make_mesh((1,), ("dp_shard",))
        value = torch.arange(12).view(4, 3).float()
        fqn = "layers.0.weight"
        parameter = torch.nn.Parameter(DTensor.from_local(value, mesh, (Replicate(),)))
        optimizer = self._make_optimizer(
            {fqn: parameter}, {fqn: ComputeLayout({"dp_shard": Owned()})}
        )
        (layout,) = optimizer._parameter_compute_layouts
        self.assertTrue(layout.storage_is_compute_ready)
        self._assert_identity_view(optimizer, layout, value)
        self.assertIs(optimizer._bucket_plans[0].items[0], layout)

    def test_mixed_bucket_views_follow_native_compute_ownership(self):
        mesh = self._make_mesh((2,), ("dp_shard",))
        matrix = torch.arange(12).view(4, 3).float()
        experts = torch.arange(48).view(4, 4, 3).float()
        values = {
            "layers.0.first": matrix,
            "layers.0.second": matrix,
            "layers.0.experts": experts,
        }
        parameters = {
            fqn: torch.nn.Parameter(
                DTensor.from_local(
                    value[:2].clone(),
                    mesh,
                    (Shard(0),),
                    shape=value.shape,
                    stride=value.stride(),
                )
            )
            for fqn, value in values.items()
        }
        optimizer = self._make_optimizer(
            parameters,
            {
                fqn: ComputeLayout(
                    {"dp_shard": Owned() if value.ndim == 2 else Shard(0)}
                )
                for fqn, value in values.items()
            },
        )
        layouts = {
            layout.fqn: layout for layout in optimizer._parameter_compute_layouts
        }
        self._assert_identity_view(optimizer, layouts["layers.0.first"], matrix)
        self._assert_identity_view(optimizer, layouts["layers.0.second"], matrix[:0])
        self._assert_identity_view(optimizer, layouts["layers.0.experts"], experts[:2])
        (bucket,) = optimizer._bucket_plans
        self.assertEqual(len(bucket.unredistributed_items), 1)
        self.assertEqual(len(bucket.redistributed_items), 2)
        for item in (*bucket.unredistributed_items, *bucket.redistributed_items):
            self.assertIs(item, layouts[item.fqn])

    def test_redistributed_expert_view_uses_subgroup_shape(self):
        mesh = self._make_mesh((2, 2), ("efsdp", "ep"), rank=1)
        value = torch.arange(45).view(3, 5, 3).float()
        fqn = "layers.0.experts"
        parameter = torch.nn.Parameter(
            DTensor.from_local(
                value[2:3, :3].contiguous(),
                mesh,
                (Shard(1), Shard(0)),
                shape=value.shape,
                stride=value.stride(),
            )
        )
        optimizer = self._make_optimizer(
            {fqn: parameter},
            {
                fqn: ComputeLayout(
                    {"efsdp": Shard(0), "ep": Shard(0)},
                    shard_order_by_tensor_dim={0: ("ep", "efsdp")},
                )
            },
        )
        (layout,) = optimizer._parameter_compute_layouts
        self.assertFalse(layout.storage_is_compute_ready)
        self._assert_identity_view(optimizer, layout, value[2:3])
        self.assertIs(optimizer._bucket_plans[0].redistributed_items[0], layout)

    def test_loading_state_rebuilds_views_for_changed_owners(self):
        mesh = self._make_mesh((2,), ("dp_shard",))
        values = {
            "layers.0.thin": torch.arange(32).view(2, 16).float(),
            "layers.0.square": torch.arange(16).view(4, 4).float(),
        }
        parameters = {
            name: torch.nn.Parameter(
                DTensor.from_local(
                    value[: value.shape[0] // 2].clone(),
                    mesh,
                    (Shard(0),),
                    shape=value.shape,
                    stride=value.stride(),
                )
            )
            for name, value in values.items()
        }
        optimizer = self._make_optimizer(
            parameters,
            {name: ComputeLayout({"dp_shard": Owned()}) for name in values},
        )
        layouts = {
            layout.fqn: layout for layout in optimizer._parameter_compute_layouts
        }
        thin_name, square_name = values
        self._assert_identity_view(optimizer, layouts[thin_name], values[thin_name][:0])
        self._assert_identity_view(optimizer, layouts[square_name], values[square_name])

        state = optimizer.state_dict()
        # With no Newton-Schulz work, the larger transfer is assigned first.
        state["param_groups"][0]["ns_steps"] = 0
        with patch.object(_BucketedRedistributionRuntime, "reserve_buffers"):
            optimizer.load_state_dict(state)
        self.assertCountEqual(optimizer._matrix_views_by_fqn, values)
        self._assert_identity_view(optimizer, layouts[thin_name], values[thin_name])
        self._assert_identity_view(
            optimizer, layouts[square_name], values[square_name][:0]
        )


if __name__ == "__main__":
    unittest.main()
