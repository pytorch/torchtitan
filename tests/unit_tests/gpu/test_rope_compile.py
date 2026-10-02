# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.qwen3_5.rope import MRoPE


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRoPELocalCompile(unittest.TestCase):
    def setUp(self):
        LocalCompileConfig(regions=["cos_sin_rope"]).apply_local_compile()

    def tearDown(self):
        LocalCompileConfig(regions=[]).apply_local_compile()
        torch._dynamo.reset()

    def test_forward_and_backward_emit_triton(self):
        from torch._inductor.utils import run_fw_bw_and_get_code

        rope = CosSinRoPE.Config(dim=128, max_context_length=2048).build().cuda()
        query = torch.randn(
            256,
            8,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn(
            256,
            1,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        positions = torch.arange(256, device="cuda")

        def run_rope():
            query_out, key_out = rope(query, key, positions)
            return query_out.sum() + key_out.sum()

        _, codes = run_fw_bw_and_get_code(run_rope)

        self.assertGreaterEqual(sum("triton" in code for code in codes), 2)

    def test_forward_and_backward_match_eager(self):
        torch.manual_seed(42)
        compiled_rope = (
            CosSinRoPE.Config(
                dim=128,
                max_context_length=128,
            )
            .build()
            .cuda()
        )
        query = torch.randn(
            64,
            8,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn(
            64,
            1,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        positions = torch.randperm(128, device="cuda")[:64]
        grad_query_out = torch.randn_like(query)
        grad_key_out = torch.randn_like(key)

        compiled_outputs = compiled_rope(query, key, positions)
        compiled_grads = torch.autograd.grad(
            compiled_outputs,
            (query, key),
            (grad_query_out, grad_key_out),
        )

        LocalCompileConfig(regions=[]).apply_local_compile()
        eager_rope = (
            CosSinRoPE.Config(
                dim=128,
                max_context_length=128,
            )
            .build()
            .cuda()
        )
        eager_query = query.detach().clone().requires_grad_()
        eager_key = key.detach().clone().requires_grad_()
        eager_outputs = eager_rope(eager_query, eager_key, positions)
        eager_grads = torch.autograd.grad(
            eager_outputs,
            (eager_query, eager_key),
            (grad_query_out, grad_key_out),
        )

        for compiled, eager in zip(
            (*compiled_outputs, *compiled_grads),
            (*eager_outputs, *eager_grads),
            strict=True,
        ):
            torch.testing.assert_close(compiled, eager, rtol=0, atol=0)

    def test_forward_and_backward_are_batch_invariant(self):
        torch.manual_seed(42)
        rope = CosSinRoPE.Config(dim=128, max_context_length=128).build().cuda()
        query = torch.randn(
            64,
            8,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn(
            64,
            1,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        positions = torch.randperm(128, device="cuda")[:64]
        grad_query_out = torch.randn_like(query)
        grad_key_out = torch.randn_like(key)

        def run_rope(query, key, positions, grad_query_out, grad_key_out):
            query_out, key_out = rope(query, key, positions)
            grad_query, grad_key = torch.autograd.grad(
                (query_out, key_out),
                (query, key),
                (grad_query_out, grad_key_out),
            )
            return query_out, key_out, grad_query, grad_key

        full_outputs = run_rope(
            query,
            key,
            positions,
            grad_query_out,
            grad_key_out,
        )

        split_outputs = [[], [], [], []]
        for inputs in zip(
            query.detach().chunk(2),
            key.detach().chunk(2),
            positions.chunk(2),
            grad_query_out.chunk(2),
            grad_key_out.chunk(2),
            strict=True,
        ):
            (
                query_part,
                key_part,
                positions_part,
                grad_query_part,
                grad_key_part,
            ) = inputs
            outputs = run_rope(
                query_part.requires_grad_(),
                key_part.requires_grad_(),
                positions_part,
                grad_query_part,
                grad_key_part,
            )
            for output_parts, output in zip(split_outputs, outputs, strict=True):
                output_parts.append(output)

        for full_output, output_parts in zip(
            full_outputs,
            split_outputs,
            strict=True,
        ):
            self.assertTrue(torch.equal(full_output, torch.cat(output_parts)))

    def test_mrope_three_axis_positions_match_eager(self):
        torch.manual_seed(42)
        config = MRoPE.Config(
            dim=12,
            max_context_length=32,
            mrope_section=[2, 2, 2],
        )
        compiled_rope = config.build().cuda()
        query = torch.randn(
            16,
            4,
            12,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn(
            16,
            1,
            12,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        positions = torch.randint(0, 32, (16, 3), device="cuda")
        grad_query_out = torch.randn_like(query)
        grad_key_out = torch.randn_like(key)

        compiled_outputs = compiled_rope(query, key, positions)
        compiled_grads = torch.autograd.grad(
            compiled_outputs,
            (query, key),
            (grad_query_out, grad_key_out),
        )

        LocalCompileConfig(regions=[]).apply_local_compile()
        eager_rope = config.build().cuda()
        eager_query = query.detach().clone().requires_grad_()
        eager_key = key.detach().clone().requires_grad_()
        eager_outputs = eager_rope(eager_query, eager_key, positions)
        eager_grads = torch.autograd.grad(
            eager_outputs,
            (eager_query, eager_key),
            (grad_query_out, grad_key_out),
        )

        for compiled, eager in zip(
            (*compiled_outputs, *compiled_grads),
            (*eager_outputs, *eager_grads),
            strict=True,
        ):
            torch.testing.assert_close(compiled, eager, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
