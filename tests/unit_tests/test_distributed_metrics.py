# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.metrics import (
    collect_dtensor_metrics,
    distribute_rank_local_metric,
    merge_dtensor_metrics,
)


class TestCollectDTensorMetrics(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @with_comms
    def test_collects_propagated_sum_and_max(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("loss",)
        )
        values = torch.tensor([1.25, 3.5], dtype=torch.float32, device=self.device_type)
        sharded = distribute_tensor(values, mesh, [Shard(0)])
        sum_metric = sharded.sum()
        max_metric = sharded.amax()
        assert isinstance(sum_metric, DTensor)
        assert isinstance(max_metric, DTensor)

        collected = collect_dtensor_metrics(
            {
                "value/sum": sum_metric,
                "value/max": max_metric,
            }
        )

        self.assertEqual(collected, {"value/sum": 4.75, "value/max": 3.5})

    @with_comms
    def test_combines_partial_metrics_without_materializing(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("loss",)
        )
        values = torch.tensor([[1.0, 5.0], [2.0, 3.0]], device=self.device_type)
        sharded = distribute_tensor(values, mesh, [Shard(0)])

        combined = merge_dtensor_metrics(
            {
                "value/sum": sharded[:, 0].sum(),
                "value/max": sharded[:, 0].amax(),
            },
            {
                "value/sum": sharded[:, 1].sum(),
                "value/max": sharded[:, 1].amax(),
            },
        )
        sum_metric = combined["value/sum"]
        max_metric = combined["value/max"]
        assert isinstance(sum_metric, DTensor)
        assert isinstance(max_metric, DTensor)
        self.assertTrue(
            all(placement.is_partial() for placement in sum_metric.placements)
        )
        self.assertTrue(
            all(placement.is_partial() for placement in max_metric.placements)
        )

        self.assertEqual(
            collect_dtensor_metrics({"value/sum": sum_metric, "value/max": max_metric}),
            {"value/sum": 11.0, "value/max": 5.0},
        )

    @with_comms
    def test_rank_local_metric_propagates_sum_mean_and_max(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("loss",)
        )
        local_value = torch.tensor(float(dist.get_rank() + 1), device=self.device_type)
        values_by_rank = distribute_rank_local_metric(local_value, mesh)
        sum_metric = values_by_rank.sum()
        mean_metric = values_by_rank.mean()
        max_metric = values_by_rank.amax()
        assert isinstance(sum_metric, DTensor)
        assert isinstance(mean_metric, DTensor)
        assert isinstance(max_metric, DTensor)

        collected = collect_dtensor_metrics(
            {
                "value/sum": sum_metric,
                "value/mean": mean_metric,
                "value/max": max_metric,
            }
        )

        self.assertEqual(collected["value/sum"], 3.0)
        self.assertEqual(collected["value/mean"], 1.5)
        self.assertEqual(collected["value/max"], 2.0)

    @with_comms
    def test_accepts_replicated_scalar(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("loss",)
        )
        scalar = distribute_tensor(
            torch.tensor(2.5, device=self.device_type), mesh, [Replicate()]
        )

        self.assertEqual(collect_dtensor_metrics({"value": scalar}), {"value": 2.5})

    @with_comms
    def test_rejects_non_scalar_metric(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("loss",)
        )
        vector = distribute_tensor(
            torch.tensor([1.0, 2.0], device=self.device_type), mesh, [Replicate()]
        )

        with self.assertRaisesRegex(ValueError, "must be scalar"):
            collect_dtensor_metrics({"value": vector})

    @with_comms
    def test_rejects_metric_attached_to_autograd(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("loss",)
        )
        scalar = distribute_tensor(
            torch.tensor(2.5, device=self.device_type, requires_grad=True),
            mesh,
            [Replicate()],
        )

        with self.assertRaisesRegex(ValueError, "detached from autograd"):
            collect_dtensor_metrics({"value": scalar})


class TestCollectDTensorMetricsWithoutDistributed(unittest.TestCase):
    def test_empty_metrics(self):
        self.assertEqual(collect_dtensor_metrics({}), {})

    def test_rejects_plain_tensor(self):
        with self.assertRaisesRegex(ValueError, "must be a DTensor"):
            collect_dtensor_metrics({"value": torch.tensor(1.0)})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
