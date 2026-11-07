# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.qwen3 import build_model_config
from torchtitan.models.qwen3.model import Qwen3TransformerBlock


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestQwen3AddRMSNorm(unittest.TestCase):
    def tearDown(self):
        LocalCompileConfig().apply()
        torch._dynamo.reset()

    def test_compiled_add_and_norm(self):
        config = build_model_config("8B", parallelism=ParallelismConfig())
        torch.manual_seed(0)
        norm = config.layers[0].attention_norm.build().cuda().to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.normal_(1.0, 0.1)
        # The method reads only attention_norm; a full 8B block would build ~200M params.
        block = torch.nn.Module()
        block.attention_norm = norm
        add_attention_norm = Qwen3TransformerBlock._add_attention_norm
        x = torch.randn(64, config.dim, device="cuda", dtype=torch.bfloat16)
        residual = 4 * torch.randn_like(x)
        grad_outputs = (torch.randn_like(x), torch.randn_like(x))

        def forward_backward():
            inputs = (x.clone().requires_grad_(), residual.clone().requires_grad_())
            outputs = add_attention_norm(block, *inputs)
            grads = torch.autograd.grad(outputs, (*inputs, norm.weight), grad_outputs)
            return outputs, grads

        LocalCompileConfig().apply()
        (eager_out, eager_sum), eager_grads = forward_backward()
        LocalCompileConfig(regions=["qwen3_add_rmsnorm"]).apply()
        (out, new_residual), grads = forward_backward()
        with torch.no_grad():
            no_grad_out, no_grad_sum = add_attention_norm(block, x, residual)

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
