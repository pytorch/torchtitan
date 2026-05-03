# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for _dist_reduce on DTensor inputs.

Covers two cases:

1. DTensor whose mesh equals the requested reduction mesh:
   ``full_tensor()`` already reduces over that mesh, so the explicit
   all_reduce must be skipped to avoid double-counting.

2. DTensor whose mesh is *orthogonal* to the requested reduction mesh
   (e.g. a TP-Replicated loss tensor reduced across the batch mesh):
   ``full_tensor()`` is a no-op for Replicate placements, so the
   explicit all_reduce must still run on the requested mesh.

Before the fix, the second case was silently dropped — the function
returned ``float(x.full_tensor().item())`` for every DTensor input,
so reductions on orthogonal meshes never ran.
"""

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Partial, Replicate
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed import utils as dist_utils


class TestDistReduceDTensor(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_dtensor_orthogonal_mesh_reduces(self):
        """A Replicated-on-TP loss tensor reduced across batch_mesh.

        With world=4, build a 2D mesh ('batch', 'tp') of shape (2, 2).
        Place a tensor with the per-rank value `1.0 + batch_rank` on the
        TP mesh as Replicate (so all 4 ranks see their batch_rank's value
        replicated across TP). Reduce-sum across batch_mesh: expected
        sum = 1.0 + 2.0 = 3.0 (each batch_rank contributes once).
        """
        mesh_2d = init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("batch", "tp")
        )
        batch_mesh = mesh_2d["batch"]
        tp_mesh = mesh_2d["tp"]

        # Each rank's local value depends on its batch position.
        batch_rank = batch_mesh.get_local_rank()
        local = torch.tensor(
            [1.0 + float(batch_rank)], device=self.device_type
        )

        # Build a Replicated DTensor on the TP mesh (no reduction across TP).
        replicated_on_tp = distribute_tensor(
            local, tp_mesh, placements=[Replicate()]
        )

        # Reduce across batch_mesh. This is the failure case.
        result = dist_utils.dist_sum(replicated_on_tp, batch_mesh)

        # Expected: 1.0 (batch=0) + 2.0 (batch=1) = 3.0 on every rank.
        self.assertEqual(result, 3.0)

    @with_comms
    def test_dtensor_same_mesh_no_double_count(self):
        """A Partial-on-batch DTensor reduced across batch_mesh.

        When the DTensor's mesh equals the reduction mesh, ``full_tensor()``
        does the reduction — the explicit all_reduce must be skipped.
        Build a Partial-on-batch DTensor where each batch rank holds
        ``1.0`` locally; reduce on batch_mesh gives ``world_batch = 2``.
        With double-counting, we'd see 4.
        """
        mesh_2d = init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("batch", "tp")
        )
        batch_mesh = mesh_2d["batch"]

        local = torch.tensor([1.0], device=self.device_type)
        partial_on_batch = distribute_tensor(
            local, batch_mesh, placements=[Partial()]
        )
        result = dist_utils.dist_sum(partial_on_batch, batch_mesh)
        self.assertEqual(result, 2.0)

    @with_comms
    def test_plain_tensor_unchanged(self):
        """Plain-tensor path still works as before."""
        mesh_2d = init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("batch", "tp")
        )
        batch_mesh = mesh_2d["batch"]

        local = torch.tensor([1.0], device=self.device_type)
        result = dist_utils.dist_sum(local, batch_mesh)
        self.assertEqual(result, 2.0)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
