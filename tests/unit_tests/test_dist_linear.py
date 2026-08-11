# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical parity for the fused TP+SP linear primitives.

These test the two autograd Functions in ``torchtitan/distributed/dist_linear.py``
directly, against a single-device reference built from the unsharded weights. No
model, no DTensor -- just the collective + GEMM math and its gradients.

Start here when adding a new fused primitive: if a shard-vs-replica mismatch or a
transposed gradient slips in, it shows up as a large error on exactly one of the
tensors below, which localizes the bug immediately.
"""

import unittest

import torch
import torch.nn.functional as F
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.dist_linear import AllGatherLinear, LinearReduceScatter


# DTensorTestBase falls back to a CPU/gloo mesh when CUDA is unavailable, so
# without this guard the CPU CI job runs these for real and dies in the
# dispatcher: the symm_mem ops have no CPU kernel.
@unittest.skipUnless(torch.cuda.is_available(), "symmetric memory requires CUDA")
class TestDistLinearPrimitives(DTensorTestBase):
    """Forward and backward parity against an unsharded reference."""

    @property
    def world_size(self) -> int:
        return 2

    # bf16 accumulates over K, so parity is checked at the dtype's noise floor
    # rather than exactly. The terms that involve no cross-rank reduction come
    # out bit-exact and are asserted as such.
    TOL = 2e-2

    def _reference(self, M, N, K, dtype):
        """Unsharded (x, w, dy) plus the single-device forward/backward."""
        torch.manual_seed(0)
        dev = self.device_type
        x = torch.randn(M, K, device=dev, dtype=dtype)
        w = torch.randn(N, K, device=dev, dtype=dtype)
        dy = torch.randn(M, N, device=dev, dtype=dtype)
        xr = x.clone().requires_grad_()
        wr = w.clone().requires_grad_()
        F.linear(xr, wr).backward(dy)
        return x, w, dy, xr.grad, wr.grad

    @with_comms
    def test_all_gather_linear_matches_unsharded(self):
        """Column-parallel: x sharded over tokens, w sharded over out-features."""
        W = self.world_size
        M, N, K = 8 * W, 64, 32
        group = torch.distributed.group.WORLD
        x, w, dy, ref_dx, ref_dw = self._reference(M, N, K, torch.bfloat16)

        xs = x.chunk(W, 0)[self.rank].clone().requires_grad_()
        ws = w.chunk(W, 0)[self.rank].clone().requires_grad_()
        y = AllGatherLinear.apply(xs, ws, None, group, group.group_name)
        y.backward(dy.chunk(W, 1)[self.rank])

        # y holds every token but only this rank's output features
        ref_y = F.linear(x, w).chunk(W, 1)[self.rank]
        torch.testing.assert_close(y, ref_y, atol=self.TOL, rtol=self.TOL)
        # dgrad goes through a reduce-scatter, so only close, not exact
        torch.testing.assert_close(
            xs.grad, ref_dx.chunk(W, 0)[self.rank], atol=self.TOL, rtol=self.TOL
        )
        # wgrad involves no cross-rank reduction -> must be exact
        self.assertEqual(ws.grad, ref_dw.chunk(W, 0)[self.rank], atol=0, rtol=0)

    @with_comms
    def test_linear_reduce_scatter_matches_unsharded(self):
        """Row-parallel: x and w both sharded over in-features (K)."""
        W = self.world_size
        M, N, K = 8 * W, 64, 32
        group = torch.distributed.group.WORLD
        x, w, dy, ref_dx, ref_dw = self._reference(M, N, K, torch.bfloat16)

        xs = x.chunk(W, 1)[self.rank].contiguous().clone().requires_grad_()
        ws = w.chunk(W, 1)[self.rank].contiguous().clone().requires_grad_()
        y = LinearReduceScatter.apply(xs, ws, None, group, group.group_name)
        y.backward(dy.chunk(W, 0)[self.rank])

        # y holds this rank's slice of the sequence but all output features
        ref_y = F.linear(x, w).chunk(W, 0)[self.rank]
        torch.testing.assert_close(y, ref_y, atol=self.TOL, rtol=self.TOL)
        # both grads are local products here -> exact
        self.assertEqual(xs.grad, ref_dx.chunk(W, 1)[self.rank], atol=0, rtol=0)
        self.assertEqual(ws.grad, ref_dw.chunk(W, 1)[self.rank], atol=0, rtol=0)

    @with_comms
    def test_bias_is_applied_once(self):
        """A replicated bias must land once, not once per rank.

        Compared against a reference that includes the bias, rather than by
        differencing the with/without outputs: |y| is much larger than |b| here,
        so one ulp of y in bf16 already exceeds any sensible tolerance on b.
        """
        W = self.world_size
        M, N, K = 8 * W, 64, 32
        group = torch.distributed.group.WORLD
        dev = self.device_type
        torch.manual_seed(1)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        b = torch.randn(N, device=dev, dtype=torch.bfloat16)

        xs = x.chunk(W, 1)[self.rank].contiguous()
        ws = w.chunk(W, 1)[self.rank].contiguous()
        y = LinearReduceScatter.apply(xs, ws, b, group, group.group_name)

        ref = F.linear(x, w, b).chunk(W, 0)[self.rank]
        torch.testing.assert_close(y, ref, atol=self.TOL, rtol=self.TOL)

        # A bias applied W times instead of once would be off by (W-1)*b, which
        # is far outside the tolerance above -- confirm that is really true, so
        # the assertion above cannot pass vacuously.
        double = F.linear(x, w, b * W).chunk(W, 0)[self.rank]
        self.assertGreater((double - ref).abs().max().item(), 10 * self.TOL)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
