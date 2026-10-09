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


class TestRunningGrad(unittest.TestCase):
    def _param_with_grad(self, grad_dtype: torch.dtype) -> nn.Parameter:
        param = nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
        param.grad_dtype = None
        param.grad = torch.ones(4, 4, dtype=grad_dtype)
        return param

    def test_returns_grad_at_least_as_wide_as_wgrad(self):
        # FP32 into FP32 (HiMidLoLinear), and BF16 into FP32: under FSDP an
        # activation checkpoint recompute builds a BF16 WGRAD while the running
        # gradient is already in the FP32 reduce dtype.
        for wgrad_dtype in (torch.float32, torch.bfloat16):
            param = self._param_with_grad(torch.float32)
            self.assertIs(running_grad(param, wgrad_dtype), param.grad)
            # Read-only: the caller clears .grad after adding into it.
            self.assertIsNotNone(param.grad)

    def test_none_for_narrower_or_missing_grad(self):
        # Adding an FP32 WGRAD into a BF16 gradient would round it.
        param = self._param_with_grad(torch.bfloat16)
        self.assertIsNone(running_grad(param, torch.float32))

        param.grad = None
        self.assertIsNone(running_grad(param, torch.float32))
        self.assertIsNone(running_grad(None, torch.float32))


if __name__ == "__main__":
    unittest.main()
