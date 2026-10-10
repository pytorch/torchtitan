# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.distributed.pipeline_parallel import (
    _disable_inplace_wgrad_accum,
    _splits_backward,
)
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
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
        # Read-only: it does not clear .grad.
        self.assertIs(running_grad(w), w.grad)

    def test_none_without_parameter(self):
        self.assertIsNone(running_grad(None))


class TestSplitBackwardSchedules(unittest.TestCase):
    def _schedule(self, *computation_types):
        from torch.distributed.pipelining.schedules import _Action

        actions = [_Action(0, ct, i) for i, ct in enumerate(computation_types)]
        return SimpleNamespace(pipeline_order={0: [*actions, None]})

    def test_detects_weight_pass_actions(self):
        from torch.distributed.pipelining.schedules import (
            BACKWARD_INPUT,
            BACKWARD_WEIGHT,
            FORWARD,
            FULL_BACKWARD,
        )

        self.assertTrue(
            _splits_backward(self._schedule(FORWARD, BACKWARD_INPUT, BACKWARD_WEIGHT))
        )
        self.assertFalse(_splits_backward(self._schedule(FORWARD, FULL_BACKWARD)))
        # Single-stage schedules such as 1F1B and GPipe have no pipeline_order.
        self.assertFalse(_splits_backward(SimpleNamespace(pipeline_order=None)))

    def test_disables_inplace_wgrad_accum(self):
        on = HiMidLoLinear.Config(in_features=8, out_features=4).build()
        off = HiMidLoLinear.Config(
            in_features=8, out_features=4, inplace_wgrad_accum=False
        ).build()
        _disable_inplace_wgrad_accum([nn.Sequential(on, off), nn.Linear(4, 4)])
        self.assertFalse(on.inplace_wgrad_accum)
        self.assertFalse(off.inplace_wgrad_accum)


if __name__ == "__main__":
    unittest.main()
