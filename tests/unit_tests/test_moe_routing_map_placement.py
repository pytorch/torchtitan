# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The routing-map scatter must preserve the router's shard placement.

Under TP/SP with EP the router outputs are DTensors sharded on the token
dim. A plain ``zeros_like(...).scatter_(...)`` cannot keep ``Shard(1)``:
the in-place form has no DTensor strategy, and the out-of-place form
redistributes to ``Replicate``, which turns the per-shard token counts
that ``RoutedExperts`` consumes as ``Partial(sum)`` into full-sequence
counts on every rank. See ``MoE.forward``.
"""

import torch
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


def _routing_map(scores_BLE, topk_expert_ids_BLK):
    """The construction under test, lifted from ``MoE.forward``."""
    if isinstance(scores_BLE, DTensor):
        local_map = torch.zeros_like(
            scores_BLE.to_local(), dtype=torch.bool
        ).scatter_(-1, topk_expert_ids_BLK.to_local(), True)
        return DTensor.from_local(
            local_map, scores_BLE.device_mesh, scores_BLE.placements
        )
    return torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
        -1, topk_expert_ids_BLK, True
    )


class TestRoutingMapPlacement(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_token_sharded_scores_keep_placement(self):
        mesh = self.build_device_mesh()
        B, L, E, K = 2, 8, 4, 2
        torch.manual_seed(0)
        scores = torch.rand(B, L, E, device=self.device_type)
        topk = scores.topk(K, dim=-1).indices

        expected = _routing_map(scores, topk)

        scores_dt = distribute_tensor(scores, mesh, [Shard(1)])
        topk_dt = distribute_tensor(topk, mesh, [Shard(1)])
        got = _routing_map(scores_dt, topk_dt)

        self.assertEqual(got.placements, scores_dt.placements)
        self.assertEqual(got.full_tensor(), expected)
        # RoutedExperts reduces the counts as Partial(sum), so each rank must
        # hold LOCAL counts, not full-sequence counts.
        self.assertEqual(
            got.to_local().sum(dim=(0, 1)).sum().item(),
            B * (L // self.world_size) * K,
        )

    @with_comms
    def test_replicated_scores_unchanged(self):
        mesh = self.build_device_mesh()
        torch.manual_seed(0)
        scores = torch.rand(2, 8, 4, device=self.device_type)
        topk = scores.topk(2, dim=-1).indices
        scores_dt = distribute_tensor(scores, mesh, [Replicate()])
        topk_dt = distribute_tensor(topk, mesh, [Replicate()])
        got = _routing_map(scores_dt, topk_dt)
        self.assertEqual(got.placements, scores_dt.placements)
        self.assertEqual(got.full_tensor(), _routing_map(scores, topk))


if __name__ == "__main__":
    import unittest

    unittest.main()
