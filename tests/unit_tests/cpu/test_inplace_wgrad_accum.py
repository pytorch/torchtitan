# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.models.common.inplace_wgrad_accum import (
    running_grad,
    uses_inplace_wgrad_accum,
)


class TestUsesInplaceWgradAccum(unittest.TestCase):
    def setUp(self):
        self.module = nn.Linear(4, 4)
        self.param = nn.Parameter(torch.zeros(4, 4))

    def test_on_for_leaf_parameter_in_eager(self):
        self.assertTrue(uses_inplace_wgrad_accum(True, self.param, self.module))

    def test_off_when_disabled_or_no_wgrad(self):
        self.assertFalse(uses_inplace_wgrad_accum(False, self.param, self.module))
        with torch.no_grad():
            self.assertFalse(uses_inplace_wgrad_accum(True, self.param, self.module))
        frozen = nn.Parameter(torch.zeros(4, 4), requires_grad=False)
        self.assertFalse(uses_inplace_wgrad_accum(True, frozen, self.module))

    def test_raises_for_non_leaf_weight(self):
        stacked = nn.Parameter(torch.zeros(2, 4, 4))
        with self.assertRaisesRegex(RuntimeError, "not the leaf parameter"):
            uses_inplace_wgrad_accum(True, stacked.flatten(0, -2), self.module)

    def test_raises_while_tracing(self):
        def traced(x):
            uses_inplace_wgrad_accum(True, self.param, self.module)
            return x

        def traced_with_flag_off(x):
            self.assertFalse(uses_inplace_wgrad_accum(False, self.param, self.module))
            return x

        with self.assertRaisesRegex(RuntimeError, "inplace_wgrad_accum=False"):
            make_fx(traced)(torch.zeros(1))
        make_fx(traced_with_flag_off)(torch.zeros(1))


class _InplaceAccumLinear(torch.autograd.Function):
    """``x @ w.T`` whose backward adds WGRAD into ``w.grad`` in place when it can."""

    observed: list[torch.Tensor | None] = []

    @staticmethod
    def forward(ctx, x, w, wgrad_dtype):  # pyrefly: ignore[bad-override]
        ctx.save_for_backward(x, w)
        ctx.weight_param = w
        ctx.wgrad_dtype = wgrad_dtype
        return x @ w.t()

    @staticmethod
    def backward(ctx, grad_out):  # pyrefly: ignore[bad-override]
        x, w = ctx.saved_tensors
        grad = running_grad(ctx.weight_param, ctx.wgrad_dtype)
        _InplaceAccumLinear.observed.append(grad)
        if grad is None:
            grad_w = grad_out.t() @ x
        else:
            torch.addmm(grad, grad_out.t(), x, out=grad)
            ctx.weight_param.grad = None
            grad_w = grad
        return grad_out @ w, grad_w, None


class TestRunningGrad(unittest.TestCase):
    def setUp(self):
        _InplaceAccumLinear.observed.clear()

    def _backward(self, w, wgrad_dtype=torch.float32):
        x = torch.randn(3, 4, dtype=w.dtype)
        _InplaceAccumLinear.apply(x, w, wgrad_dtype).sum().backward()
        return _InplaceAccumLinear.observed[-1]

    def test_returns_running_grad_in_backward(self):
        w = nn.Parameter(torch.randn(2, 4))
        self.assertIsNone(self._backward(w))  # first contribution
        running = w.grad
        self.assertIs(self._backward(w), running)
        # Read-only outside of backward: no AccumulateGrad runs there.
        self.assertIsNone(running_grad(w, torch.float32))

    def test_accepts_wider_and_rejects_narrower_grad(self):
        # FP32 into FP32 (HiMidLoLinear), and BF16 into FP32: under FSDP an
        # activation checkpoint recompute builds a BF16 WGRAD while the running
        # gradient is already in the FP32 reduce dtype. Adding an FP32 WGRAD
        # into a BF16 gradient would round it.
        for grad_dtype, wgrad_dtype, accepted in (
            (torch.float32, torch.bfloat16, True),
            (torch.bfloat16, torch.float32, False),
        ):
            w = nn.Parameter(torch.randn(2, 4))
            w.grad_dtype = None
            w.grad = torch.zeros(2, 4, dtype=grad_dtype)
            observed = self._backward(w, wgrad_dtype)
            self.assertEqual(observed is not None, accepted)

    def test_none_without_parameter(self):
        self.assertIsNone(running_grad(None, torch.float32))

    def test_split_backward_keeps_earlier_contributions(self):
        # Pipelining's zero-bubble schedules split backward into an input pass
        # and a weight pass, both under autograd.grad(), where AccumulateGrad
        # never runs. Adding into .grad there would drop the earlier microbatches.
        from torch.distributed.pipelining._backward import (
            stage_backward_input,
            stage_backward_weight,
        )

        torch.manual_seed(0)
        w = nn.Parameter(torch.randn(8, 4, dtype=torch.float64))
        expected = torch.zeros_like(w)
        for _ in range(3):
            x = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
            out = _InplaceAccumLinear.apply(x * 1.0, w, torch.float64)
            grad_out = torch.randn_like(out)
            expected += grad_out.t() @ x.detach()
            _, param_groups = stage_backward_input([out], [grad_out], [x], iter([w]))
            stage_backward_weight(iter([w]), param_groups)
        self.assertTrue(all(grad is None for grad in _InplaceAccumLinear.observed))
        torch.testing.assert_close(w.grad, expected)


if __name__ == "__main__":
    unittest.main()
