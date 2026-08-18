# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

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
    def test_async_tp_requires_model_compile(self):
        invalid_configs = (
            {"enable_async_tensor_parallel": True},
            {
                "enable": True,
                "enable_async_tensor_parallel": True,
                "components": ["loss"],
            },
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(
                    ValueError,
                    "Async TP requires 'model' in --compile.components and "
                    "--compile.enable",
                ):
                    CompileConfig(**kwargs)

    def test_apply_compile_configures_async_tp(self):
        model = TinyModel(num_layers=2, dim=128)
        compile_config = CompileConfig(
            enable=True,
            enable_async_tensor_parallel=True,
        )
        parallel_dims = MagicMock(tp_enabled=True)
        tp_mesh = parallel_dims.get_dense_tp_mesh.return_value
        tp_mesh.get_group.return_value.group_name = "tp_group"
        previous_micro_pipeline_tp = torch._inductor.config._micro_pipeline_tp

        try:
            with (
                patch.object(torch.nn.Module, "compile"),
                patch(
                    "torch.distributed._symmetric_memory.enable_symm_mem_for_group"
                ) as enable_symm_mem,
            ):
                apply_compile(
                    model,
                    compile_config=compile_config,
                    parallel_dims=parallel_dims,
                )

            enable_symm_mem.assert_called_once_with("tp_group")
            self.assertTrue(torch._inductor.config._micro_pipeline_tp)
        finally:
            torch._inductor.config._micro_pipeline_tp = previous_micro_pipeline_tp

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_grouped_mm_compiles_and_runs(self):
        model = TinyModel(num_layers=2, dim=128).cuda()
        compile_config = CompileConfig(backend="inductor")

        apply_compile(
            model,
            compile_config=compile_config,
            parallel_dims=MagicMock(tp_enabled=False),
        )

        from torchtitan.models.common.moe import GroupedExperts

        num_experts = 8
        dim = 128
        hidden_dim = 256
        experts = GroupedExperts(
            GroupedExperts.Config(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
            )
        ).cuda()
        num_tokens_per_expert = torch.tensor(
            [10, 8, 12, 9, 11, 7, 10, 13], dtype=torch.int32, device="cuda"
        )
        total_tokens = num_tokens_per_expert.sum().item()
        x = torch.randn(total_tokens, dim, device="cuda")

        output = experts(x, num_tokens_per_expert)

        self.assertEqual(output.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
