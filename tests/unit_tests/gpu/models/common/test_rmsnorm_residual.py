# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch._inductor.utils import run_and_get_code

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.nn_modules import RMSNorm


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRMSNormResidualCompile(unittest.TestCase):
    def tearDown(self):
        LocalCompileConfig().apply()
        torch._dynamo.reset()

    def test_add_and_norm_run_as_one_kernel(self):
        torch.manual_seed(0)
        norm = RMSNorm.Config(normalized_shape=1024, eps=1e-6).build().cuda()
        norm = norm.to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.normal_(1.0, 0.1)
        x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
        residual = 4 * torch.randn_like(x)
        grad_outputs = (torch.randn_like(x), torch.randn_like(x))

        def forward_backward():
            inputs = (x.clone().requires_grad_(), residual.clone().requires_grad_())
            outputs = norm(inputs[0], residual=inputs[1])
            grads = torch.autograd.grad(outputs, (*inputs, norm.weight), grad_outputs)
            return outputs, grads

        LocalCompileConfig().apply()
        (eager_out, eager_sum), eager_grads = forward_backward()
        LocalCompileConfig(regions=["add_rmsnorm"]).apply()
        ((out, new_residual), grads), codes = run_and_get_code(forward_backward)

        with torch.no_grad():
            no_grad_out, no_grad_sum = norm(x, residual=residual)

        forward_code = codes[0]
        self.assertEqual(forward_code.count("@triton.jit"), 1)
        self.assertIn("rsqrt", forward_code)
        # The trainer (grad) and the generator (no_grad) get the same bits.
        torch.testing.assert_close(no_grad_out, out, rtol=0, atol=0)
        torch.testing.assert_close(no_grad_sum, new_residual, rtol=0, atol=0)
        # bf16 add: the compiled sum rounds exactly as eager's; the norm sums in another order.
        torch.testing.assert_close(new_residual, eager_sum, rtol=0, atol=0)
        torch.testing.assert_close(out, eager_out)
        for grad, eager_grad in zip(grads, eager_grads, strict=True):
            torch.testing.assert_close(grad, eager_grad, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
