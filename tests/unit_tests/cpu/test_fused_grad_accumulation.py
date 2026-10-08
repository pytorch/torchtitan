# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.models.common.fused_grad_accumulation import (
    can_fuse_grad_accumulation,
    take_grad_for_fused_accumulation,
)


class TestCanFuseGradAccumulation(unittest.TestCase):
    def test_requires_leaf_parameter(self):
        param = nn.Parameter(torch.zeros(2, 4, 4))
        self.assertTrue(can_fuse_grad_accumulation(param))
        # A stacked weight's flattened view is not a leaf.
        self.assertFalse(can_fuse_grad_accumulation(param.flatten(0, -2)))

    def test_disabled_while_tracing(self):
        param = nn.Parameter(torch.zeros(4, 4))
        results = []

        def fn(x):
            results.append(can_fuse_grad_accumulation(param))
            return x

        make_fx(fn)(torch.zeros(1))
        self.assertEqual(results, [False])


class TestTakeGradForFusedAccumulation(unittest.TestCase):
    def _param_with_grad(self, grad_dtype: torch.dtype) -> nn.Parameter:
        param = nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
        param.grad_dtype = None
        param.grad = torch.ones(4, 4, dtype=grad_dtype)
        return param

    def test_takes_grad_at_least_as_wide_as_wgrad(self):
        # FP32 into FP32 (HiMidLoLinear), and BF16 into FP32: under FSDP an
        # activation checkpoint recompute builds a BF16 WGRAD while the running
        # gradient is already in the FP32 reduce dtype.
        for wgrad_dtype in (torch.float32, torch.bfloat16):
            param = self._param_with_grad(torch.float32)
            running_grad = param.grad

            taken = take_grad_for_fused_accumulation(param, wgrad_dtype)

            self.assertIs(taken, running_grad)
            self.assertIsNone(param.grad)

    def test_leaves_narrower_or_missing_grad(self):
        # Adding an FP32 WGRAD into a BF16 gradient would round it.
        param = self._param_with_grad(torch.bfloat16)
        self.assertIsNone(take_grad_for_fused_accumulation(param, torch.float32))
        self.assertIsNotNone(param.grad)

        param.grad = None
        self.assertIsNone(take_grad_for_fused_accumulation(param, torch.float32))
        self.assertIsNone(take_grad_for_fused_accumulation(None, torch.float32))


if __name__ == "__main__":
    unittest.main()
