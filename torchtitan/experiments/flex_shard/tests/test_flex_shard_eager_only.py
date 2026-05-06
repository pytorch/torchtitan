# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from contextlib import contextmanager
from datetime import timedelta
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import DataParallelMeshDims

from torchtitan.experiments.flex_shard import (
    BucketSpec,
    flex_shard,
    is_flex_shard_param,
    lift_params_to_global_spmd_mesh,
    per_param_placements,
)


@contextmanager
def _single_rank_cpu_mesh():
    created_pg = False
    with NamedTemporaryFile() as store:
        if not dist.is_initialized():
            dist.init_process_group(
                "gloo",
                init_method=f"file://{store.name}",
                rank=0,
                world_size=1,
                timeout=timedelta(seconds=20),
            )
            created_pg = True
        try:
            yield init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
        finally:
            if created_pg and dist.is_initialized():
                dist.destroy_process_group()


def _flex_shard_tiny_model(mesh):
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    lift_params_to_global_spmd_mesh(model, mesh)
    flex_shard(
        model,
        mesh,
        DataParallelMeshDims(shard="fsdp"),
        shard_placement_fn=per_param_placements,
        buckets=[
            BucketSpec(["0.*"], reshard_after_forward=False),
            BucketSpec(["2.*"], reshard_after_forward=False),
        ],
    )
    return model


class TestFlexShardEagerOnly(unittest.TestCase):
    def test_eager_forward_backward_on_cpu_mesh(self):
        with _single_rank_cpu_mesh() as mesh:
            model = _flex_shard_tiny_model(mesh)

            loss = model(torch.randn(3, 4)).sum()
            loss.backward()

            for param in model.parameters():
                self.assertTrue(is_flex_shard_param(param))
                self.assertIsNotNone(param.grad)

    def test_graph_capture_raises(self):
        with _single_rank_cpu_mesh() as mesh:
            model = _flex_shard_tiny_model(mesh)

            with patch.object(torch.compiler, "is_compiling", return_value=True):
                with self.assertRaisesRegex(ValueError, "eager execution only"):
                    model(torch.randn(3, 4))


if __name__ == "__main__":
    unittest.main()
