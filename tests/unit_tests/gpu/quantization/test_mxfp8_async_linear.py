# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerics of the async tensor-parallel MXFP8 linears against MXFP8Linear.

Developer-run: needs 2 SM100+ GPUs.
"""

import unittest

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.tools.utils import has_cuda_capability

try:
    from torchtitan.quantization.mxfp8.async_linear import (
        _MXFP8AllGatherLinearFunction,
        _MXFP8LinearReduceScatterFunction,
    )
    from torchtitan.quantization.mxfp8.linear import _MXFP8LinearFunction
    from torchtitan.quantization.mxfp8.tensor import _quantize_mxfp8_weight
except ImportError:
    _MXFP8LinearFunction = None


def _operands(w: torch.Tensor) -> tuple[torch.Tensor, ...]:
    ops = _quantize_mxfp8_weight(w)
    return (
        ops.weight_qdata_fprop_KN,
        ops.weight_scale_fprop_swizzled,
        ops.weight_qdata_dgrad_NK,
        ops.weight_scale_dgrad_swizzled,
    )


@unittest.skipIf(
    _MXFP8LinearFunction is None or not has_cuda_capability(10, 0),
    "MXFP8 requires torchao and SM100+",
)
class TestMXFP8AsyncLinear(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    # The parts with a cross-rank reduction or a differently tiled GEMM are
    # compared at bf16 noise; WGRAD sees identical operands and must be exact.
    TOL = 2e-2

    def _reference(self, x, w, bias, dy):
        """Unsharded MXFP8 forward/backward."""
        xr = x.clone().requires_grad_()
        wr = w.clone().requires_grad_()
        br = None if bias is None else bias.clone().requires_grad_()
        y = _MXFP8LinearFunction.apply(xr, wr, *_operands(w), br, "bf16", False)
        y.backward(dy)
        return y, xr.grad, wr.grad, None if br is None else br.grad

    @with_comms
    @parametrize("leading_dims", [(), (2,)])
    @parametrize("has_bias", [False, True])
    @parametrize("saved_format", ["bf16", "mxfp8"])
    def test_all_gather_linear_matches_unsharded(
        self, leading_dims, has_bias, saved_format
    ):
        """Column-parallel: x sharded over tokens, w sharded over out-features."""
        W, rank = self.world_size, self.rank
        group = torch.distributed.group.WORLD
        # 256 rows per rank: the fused MXFP8 ops need multiples of 128.
        M, N, K = 256 * W, 64 * W, 128
        torch.manual_seed(0)
        dev, dt = self.device_type, torch.bfloat16
        # Scaled so the outputs are O(1), where TOL is a few bf16 ulps.
        x = torch.randn(M, *leading_dims, K, device=dev, dtype=dt) / 4
        w = torch.randn(N, K, device=dev, dtype=dt) / 4
        bias = torch.randn(N, device=dev, dtype=dt) if has_bias else None
        dy = torch.randn(M, *leading_dims, N, device=dev, dtype=dt) / 4
        ref_y, ref_dx, ref_dw, ref_db = self._reference(x, w, bias, dy)

        xs = x.chunk(W, 0)[rank].clone().requires_grad_()
        ws = w.chunk(W, 0)[rank].clone().requires_grad_()
        bs = None if bias is None else bias.chunk(W)[rank].clone().requires_grad_()
        y = _MXFP8AllGatherLinearFunction.apply(
            xs,
            ws,
            *_operands(ws.detach()),
            bs,
            group,
            group.group_name,
            saved_format,
            False,
        )
        y.backward(dy.chunk(W, -1)[rank].contiguous())

        torch.testing.assert_close(
            y, ref_y.chunk(W, -1)[rank], atol=self.TOL, rtol=self.TOL
        )
        torch.testing.assert_close(
            xs.grad, ref_dx.chunk(W, 0)[rank], atol=self.TOL, rtol=self.TOL
        )
        self.assertEqual(ws.grad, ref_dw.chunk(W, 0)[rank], atol=0, rtol=0)
        if has_bias:
            self.assertEqual(bs.grad, ref_db.chunk(W)[rank], atol=0, rtol=0)

    @with_comms
    @parametrize("leading_dims", [(), (2,)])
    @parametrize("has_bias", [False, True])
    @parametrize("saved_format", ["bf16", "mxfp8"])
    def test_linear_reduce_scatter_matches_unsharded(
        self, leading_dims, has_bias, saved_format
    ):
        """Row-parallel: x and w sharded over in-features, y over tokens."""
        W, rank = self.world_size, self.rank
        group = torch.distributed.group.WORLD
        # 256 rows per rank after the scatter: the fused MXFP8 ops need 128.
        M, N, K = 256 * W, 64, 128 * W
        torch.manual_seed(0)
        dev, dt = self.device_type, torch.bfloat16
        # Scaled so the outputs are O(1), where TOL is a few bf16 ulps.
        x = torch.randn(M, *leading_dims, K, device=dev, dtype=dt) / 4
        w = torch.randn(N, K, device=dev, dtype=dt) / 4
        bias = torch.randn(N, device=dev, dtype=dt) if has_bias else None
        dy = torch.randn(M, *leading_dims, N, device=dev, dtype=dt) / 4
        ref_y, ref_dx, ref_dw, ref_db = self._reference(x, w, bias, dy)

        xs = x.chunk(W, -1)[rank].contiguous().requires_grad_()
        ws = w.chunk(W, -1)[rank].contiguous().requires_grad_()
        bs = None if bias is None else bias.clone().requires_grad_()
        y = _MXFP8LinearReduceScatterFunction.apply(
            xs,
            ws,
            *_operands(ws.detach()),
            bs,
            group,
            group.group_name,
            saved_format,
            False,
        )
        y.backward(dy.chunk(W, 0)[rank].contiguous())

        torch.testing.assert_close(
            y, ref_y.chunk(W, 0)[rank], atol=self.TOL, rtol=self.TOL
        )
        torch.testing.assert_close(
            xs.grad, ref_dx.chunk(W, -1)[rank], atol=self.TOL, rtol=self.TOL
        )
        self.assertEqual(ws.grad, ref_dw.chunk(W, -1)[rank], atol=0, rtol=0)
        if has_bias:
            torch.testing.assert_close(bs.grad, ref_db, atol=self.TOL, rtol=self.TOL)


instantiate_parametrized_tests(TestMXFP8AsyncLinear)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
