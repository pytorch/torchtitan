# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.config import CompileConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module, ModuleDict


class TransformerBlock(Module):
    def __init__(self, dim=512):
        super().__init__()
        linear_config = Linear.Config(in_features=dim, out_features=dim, bias=False)
        self.attention = linear_config.build()
        self.mlp = linear_config.build()
        self.moe_enabled = False

    def forward(self, x):
        x = self.attention(x)
        x = self.mlp(x)
        return x


class TinyModel(Module):
    def __init__(self, num_layers=2, dim=512):
        super().__init__()
        self.layers = ModuleDict(
            {str(i): TransformerBlock(dim) for i in range(num_layers)}
        )

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x


class TestApplyCompile(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_grouped_mm_compiles_and_runs(self):
        model = TinyModel(num_layers=2, dim=128).cuda()
        compile_config = CompileConfig(backend="inductor")

        apply_compile(model, compile_config)

        from torchtitan.models.common import moe as moe_module

        num_experts = 8
        dim = 128
        hidden_dim = 256
        w1 = torch.randn(num_experts, hidden_dim, dim, device="cuda")
        w2 = torch.randn(num_experts, dim, hidden_dim, device="cuda")
        w3 = torch.randn(num_experts, hidden_dim, dim, device="cuda")
        num_tokens_per_expert = torch.tensor(
            [10, 8, 12, 9, 11, 7, 10, 13], dtype=torch.int32, device="cuda"
        )
        total_tokens = num_tokens_per_expert.sum().item()
        x = torch.randn(total_tokens, dim, device="cuda")

        output = moe_module._run_experts_grouped_mm(
            w1, w2, w3, x, num_tokens_per_expert
        )

        self.assertEqual(output.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
