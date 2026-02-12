# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.config import CompileConfig
from torchtitan.models.llama4.parallelize import apply_compile


class TransformerBlock(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attention = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Linear(dim, dim, bias=False)
        self.moe_enabled = False

    def forward(self, x):
        x = self.attention(x)
        x = self.mlp(x)
        return x


class TinyModel(nn.Module):
    def __init__(self, num_layers=2, dim=512):
        super().__init__()
        self.layers = nn.ModuleDict(
            {str(i): TransformerBlock(dim) for i in range(num_layers)}
        )

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x


class TestApplyCompile(unittest.TestCase):
    def test_patched_once(self):
        """
        Calls apply_compile multiple times, as in the case with PP.
        But patches should only happen once
        """
        unused_model1 = TinyModel(num_layers=2, dim=128)
        unused_model2 = TinyModel(num_layers=2, dim=128)
        compile_config = CompileConfig(backend="eager")

        apply_compile(unused_model1, compile_config, ep_enabled=True)
        apply_compile(unused_model2, compile_config, ep_enabled=True)

        from torchtitan.models.common.moe import moe as moe_module

        # Generate sample inputs for _run_experts_grouped_mm
        num_experts = 8
        dim = 128
        hidden_dim = 256
        w1 = torch.randn(num_experts, hidden_dim, dim)
        w2 = torch.randn(num_experts, dim, hidden_dim)
        w3 = torch.randn(num_experts, hidden_dim, dim)
        num_tokens_per_expert = torch.tensor(
            [10, 8, 12, 9, 11, 7, 10, 13], dtype=torch.int32
        )
        total_tokens = num_tokens_per_expert.sum().item()
        x = torch.randn(total_tokens, dim)

        # Call the function, should not error
        output = moe_module._run_experts_grouped_mm(
            w1, w2, w3, x, num_tokens_per_expert
        )

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Num tokens per expert: {num_tokens_per_expert}")


if __name__ == "__main__":
    unittest.main()
