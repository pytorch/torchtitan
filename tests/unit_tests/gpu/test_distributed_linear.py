# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical parity for the fused TP+SP linear primitives.

These test the autograd Functions in ``torchtitan/models/common/async_linear.py``
directly, against a single-device reference built from the unsharded weights. No
model, no DTensor -- just the collective + GEMM math and its gradients.

Start here when adding a new fused primitive: if a shard-vs-replica mismatch or a
transposed gradient slips in, it shows up as a large error on exactly one of the
tensors below, which localizes the bug immediately.
"""

import unittest

import spmd_types as spmd
import torch
import torch.nn.functional as F
from spmd_types.checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed.spmd_types import set_current_spmd_mesh

from torchtitan.models.common.async_linear import (
    AsyncAllGatherLinear,
    AsyncLinearReduceScatter,
)


@unittest.skipUnless(
    torch.cuda.device_count() >= 2, "symmetric memory requires two CUDA devices"
)
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
        y = AsyncAllGatherLinear.apply(xs, ws, None, group, group.group_name)
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
    def test_all_gather_linear_preserves_stacked_projection_axis(self):
        """Column-parallel stacked weights shard N without sharding the stack."""
        W = self.world_size
        M, N, K = 8 * W, 32, 32
        group = torch.distributed.group.WORLD
        dev = self.device_type
        torch.manual_seed(0)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        weight = torch.randn(2, N, K, device=dev, dtype=torch.bfloat16)
        bias = torch.randn(2, N, device=dev, dtype=torch.bfloat16)
        grad_y = torch.randn(M, 2, N, device=dev, dtype=torch.bfloat16)

        reference_x = x.clone().requires_grad_()
        reference_weight = weight.clone().requires_grad_()
        reference_bias = bias.clone().requires_grad_()
        reference_y = F.linear(
            reference_x,
            reference_weight.flatten(0, -2),
            reference_bias.flatten(),
        ).unflatten(-1, (2, N))
        reference_y.backward(grad_y)

        x_shard = x.chunk(W, 0)[self.rank].clone().requires_grad_()
        weight_shard = weight.chunk(W, 1)[self.rank].clone().requires_grad_()
        bias_shard = bias.chunk(W, 1)[self.rank].clone().requires_grad_()
        y_shard_flat = AsyncAllGatherLinear.apply(
            x_shard,
            weight_shard.flatten(0, -2),
            bias_shard.flatten(),
            group,
            group.group_name,
        )
        y_shard = y_shard_flat.unflatten(-1, weight_shard.shape[:-1])
        y_shard.backward(grad_y.chunk(W, 2)[self.rank])

        torch.testing.assert_close(
            y_shard,
            reference_y.chunk(W, 2)[self.rank],
            atol=self.TOL,
            rtol=self.TOL,
        )
        torch.testing.assert_close(
            x_shard.grad,
            reference_x.grad.chunk(W, 0)[self.rank],
            atol=2 * self.TOL,
            rtol=2 * self.TOL,
        )
        self.assertEqual(
            weight_shard.grad,
            reference_weight.grad.chunk(W, 1)[self.rank],
            atol=0,
            rtol=0,
        )
        self.assertEqual(
            bias_shard.grad,
            reference_bias.grad.chunk(W, 1)[self.rank],
            atol=0,
            rtol=0,
        )

    @with_comms
    def test_linear_reduce_scatter_matches_unsharded(self):
        """Row-parallel: x and w both sharded over in-features (K)."""
        W = self.world_size
        M, N, K = 8 * W, 64, 32
        group = torch.distributed.group.WORLD
        x, w, dy, ref_dx, ref_dw = self._reference(M, N, K, torch.bfloat16)

        xs = x.chunk(W, 1)[self.rank].contiguous().clone().requires_grad_()
        ws = w.chunk(W, 1)[self.rank].contiguous().clone().requires_grad_()
        y = AsyncLinearReduceScatter.apply(xs, ws, None, group, group.group_name)
        y.backward(dy.chunk(W, 0)[self.rank])

        # y holds this rank's slice of the sequence but all output features
        ref_y = F.linear(x, w).chunk(W, 0)[self.rank]
        torch.testing.assert_close(y, ref_y, atol=self.TOL, rtol=self.TOL)
        # both grads are local products here -> exact
        self.assertEqual(xs.grad, ref_dx.chunk(W, 1)[self.rank], atol=0, rtol=0)
        self.assertEqual(ws.grad, ref_dw.chunk(W, 1)[self.rank], atol=0, rtol=0)

    @with_comms
    def test_bias_is_applied_once(self):
        """An invariant bias must land once, not once per rank.

        Compared against a reference that includes the bias, rather than by
        differencing the with/without outputs: |y| is much larger than |b| here,
        so one ulp of y in bf16 already exceeds any sensible tolerance on b.
        """
        W = self.world_size
        M, N, K = 8 * W, 64, 32
        dev = self.device_type
        mesh = init_device_mesh(dev, (W,), mesh_dim_names=("tp",))
        group = mesh.get_group("tp")
        torch.manual_seed(1)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        b = torch.randn(N, device=dev, dtype=torch.bfloat16, requires_grad=True)
        ref_bias = b.detach().clone().requires_grad_()

        xs = x.chunk(W, 1)[self.rank].contiguous()
        ws = w.chunk(W, 1)[self.rank].contiguous()
        with set_current_spmd_mesh(mesh), typecheck(local=False):
            spmd.assert_type(xs, {group: spmd.S(1)})
            spmd.assert_type(ws, {group: spmd.S(1)})
            spmd.assert_type(b, {group: spmd.I})
            y = AsyncLinearReduceScatter.apply(xs, ws, b, group, group.group_name)
            y.sum().backward()

        ref_full = F.linear(x, w, ref_bias)
        ref_full.sum().backward()
        ref = ref_full.chunk(W, 0)[self.rank]
        torch.testing.assert_close(y, ref, atol=self.TOL, rtol=self.TOL)
        self.assertEqual(b.grad, ref_bias.grad, atol=0, rtol=0)

        # A bias applied W times instead of once would be off by (W-1)*b, which
        # is far outside the tolerance above -- confirm that is really true, so
        # the assertion above cannot pass vacuously.
        double = F.linear(x, w, b.detach() * W).chunk(W, 0)[self.rank]
        self.assertGreater((double - ref).abs().max().item(), 10 * self.TOL)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
