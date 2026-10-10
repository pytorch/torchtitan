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
    def forward(ctx, x, w):  # pyrefly: ignore[bad-override]
        ctx.save_for_backward(x, w)
        ctx.weight_param = w
        return x @ w.t()

    @staticmethod
    def backward(ctx, grad_out):  # pyrefly: ignore[bad-override]
        x, w = ctx.saved_tensors
        grad = running_grad(ctx.weight_param)
        _InplaceAccumLinear.observed.append(grad)
        if grad is None:
            grad_w = grad_out.t() @ x
        else:
            torch.addmm(grad, grad_out.t(), x, out=grad)
            ctx.weight_param.grad = None
            grad_w = grad
        return grad_out @ w, grad_w


class TestRunningGrad(unittest.TestCase):
    def setUp(self):
        _InplaceAccumLinear.observed.clear()

    def _backward(self, w):
        x = torch.randn(3, 4, dtype=w.dtype)
        _InplaceAccumLinear.apply(x, w).sum().backward()
        return _InplaceAccumLinear.observed[-1]

    def test_returns_running_grad_in_backward(self):
        w = nn.Parameter(torch.randn(2, 4))
        self.assertIsNone(self._backward(w))  # first contribution
        running = w.grad
        self.assertIs(self._backward(w), running)
        # Read-only outside of backward: no AccumulateGrad runs there.
        self.assertIsNone(running_grad(w))

    def test_none_without_parameter(self):
        self.assertIsNone(running_grad(None))

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
            out = _InplaceAccumLinear.apply(x * 1.0, w)
            grad_out = torch.randn_like(out)
            expected += grad_out.t() @ x.detach()
            _, param_groups = stage_backward_input([out], [grad_out], [x], iter([w]))
            stage_backward_weight(iter([w]), param_groups)
        self.assertTrue(all(grad is None for grad in _InplaceAccumLinear.observed))
        torch.testing.assert_close(w.grad, expected)


if __name__ == "__main__":
    unittest.main()
