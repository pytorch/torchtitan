# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.ops.topk import deterministic_topk


class TestDeterministicTopKDTensor(DTensorTestBase):
    """deterministic_topk is a custom op, so it needs an explicit DTensor
    sharding strategy (register_sharding) to run under TP/EP MoE routing.
    Without it, DTensor dispatch raises "Operator torchtitan.deterministic_topk
    .default does not have a sharding strategy registered."
    """

    @property
    def world_size(self):
        return 2

    def _check(self, placements, *, topk_dim=-1):
        torch.manual_seed(0)
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        # (rows, experts); topk along the expert dim, mirroring MoE routing.
        global_x = torch.randn(8, 16, device=self.device_type)
        ref_values, _ = torch.topk(global_x, k=4, dim=topk_dim, sorted=True)

        x_dt = distribute_tensor(global_x.clone(), mesh, placements)
        values_dt, indices_dt = deterministic_topk(x_dt, k=4, dim=topk_dim, sorted=True)

        # Values and indices share the input's placement (after any redistribute
        # off the topk dim), and the gathered values match plain torch.topk.
        self.assertEqual(values_dt.placements, indices_dt.placements)
        torch.testing.assert_close(values_dt.full_tensor(), ref_values)

    @with_comms
    def test_replicate(self):
        # Input replicated on the topk dim: the MoE router's common case.
        self._check([Replicate()])

    @with_comms
    def test_shard_non_topk_dim(self):
        # Sharding a non-topk dim is allowed; outputs stay sharded the same way.
        self._check([Shard(0)])

    @with_comms
    def test_shard_topk_dim_redistributes(self):
        # Sharding the topk dim is not an acceptable input placement, so DTensor
        # must redistribute it to replicate before the topk runs.
        self._check([Shard(1)])


if __name__ == "__main__":
    unittest.main()
