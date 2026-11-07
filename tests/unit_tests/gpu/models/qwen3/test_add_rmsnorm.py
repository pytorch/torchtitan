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

    def test_compiled_add_and_norm_matches_eager(self):
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

        # The region runs only without grad: under grad, each block returns the sum.
        with torch.no_grad():
            LocalCompileConfig().apply()
            eager_out, eager_sum = add_attention_norm(block, x, residual)
            LocalCompileConfig(regions=["qwen3_add_rmsnorm"]).apply()
            out, new_residual = add_attention_norm(block, x, residual)

        # bf16 add: the compiled sum rounds exactly as eager's.
        torch.testing.assert_close(new_residual, eager_sum, rtol=0, atol=0)
        # emulate_precision_casts: the compiled norm reads that bf16 sum, as eager does,
        # so all but a few outputs match eager bit for bit (0.7% differ without it).
        fraction_differing = (out != eager_out).float().mean().item()
        self.assertLess(fraction_differing, 1e-3)


if __name__ == "__main__":
    unittest.main()
