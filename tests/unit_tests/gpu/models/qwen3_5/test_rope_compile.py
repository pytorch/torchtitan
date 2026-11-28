# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE
from torchtitan.models.qwen3_5.rope import MRoPE


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRoPELocalCompile(unittest.TestCase):
    def setUp(self):
        LocalCompileConfig(regions=["rope"]).apply()

    def tearDown(self):
        LocalCompileConfig().apply()
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

        LocalCompileConfig().apply()
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

        LocalCompileConfig().apply()
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestComplexRoPELocalCompile(unittest.TestCase):
    def tearDown(self):
        LocalCompileConfig().apply()
        torch._dynamo.reset()

    def _run(self, rope, query_base, key, positions, inverse):
        # RoPE reads the last 64 dims of query_base: all of it, or a strided slice.
        query_base = query_base.detach().clone().requires_grad_()
        query = query_base[..., -64:]
        key = key.detach().clone().requires_grad_()
        if inverse:
            inputs = (query_base,)
            outputs = (rope(query, positions=positions, inverse=True),)
        else:
            inputs = (query_base, key)
            outputs = rope(query, key, positions)
        generator = torch.Generator(device="cuda").manual_seed(1)
        grad_outputs = [
            torch.randn(out.shape, dtype=out.dtype, device="cuda", generator=generator)
            for out in outputs
        ]
        grads = torch.autograd.grad(outputs, inputs, grad_outputs)
        return outputs, grads

    def test_compiled_matches_eager_complex(self):
        torch.manual_seed(42)
        rope = ComplexRoPE.Config(dim=64, max_context_length=128).build().cuda()
        positions = torch.randperm(128, device="cuda")[:64]

        # bf16 and fp32; a contiguous query and a strided slice of a wider one (DeepSeek-V3's q_pe).
        # Compiled real arithmetic can differ from eager complex by 1 bf16 ulp in rare elements.
        for dtype, query_dim, inverse in itertools.product(
            (torch.bfloat16, torch.float32), (64, 192), (False, True)
        ):
            query_base = torch.randn(64, 8, query_dim, device="cuda", dtype=dtype)
            key = torch.randn(64, 1, 64, device="cuda", dtype=dtype)
            with self.subTest(dtype=dtype, query_dim=query_dim, inverse=inverse):
                torch._dynamo.reset()  # 8 cases: stay clear of the 8-graph limit
                LocalCompileConfig().apply()
                eager_outputs, eager_grads = self._run(
                    rope, query_base, key, positions, inverse
                )
                LocalCompileConfig(regions=["rope"]).apply()
                outputs, grads = self._run(rope, query_base, key, positions, inverse)
                for compiled, eager in zip(
                    (*outputs, *grads), (*eager_outputs, *eager_grads), strict=True
                ):
                    torch.testing.assert_close(compiled, eager)

    def test_compiled_matches_eager_complex_non_contiguous(self):
        # A split key with an inserted head dim, and a query with transposed head strides:
        # the compiled path rotates these without the contiguous int32 shortcut.
        torch.manual_seed(42)
        rope = ComplexRoPE.Config(dim=64, max_context_length=128).build().cuda()
        positions = torch.randperm(128, device="cuda")[:64]
        query_storage = torch.randn(8, 64, 64, device="cuda", dtype=torch.bfloat16)
        key_full = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16)

        def run():
            query_leaf = query_storage.detach().clone().requires_grad_()
            key_leaf = key_full.detach().clone().requires_grad_()
            query = query_leaf.transpose(0, 1)
            key = key_leaf[..., -64:].unsqueeze(1)
            outputs = rope(query, key, positions)
            grads = torch.autograd.grad(
                outputs, (query_leaf, key_leaf), [torch.ones_like(o) for o in outputs]
            )
            return (*outputs, *grads)

        LocalCompileConfig().apply()
        eager = run()
        LocalCompileConfig(regions=["rope"]).apply()
        for compiled, expected in zip(run(), eager, strict=True):
            torch.testing.assert_close(compiled, expected)

    def test_compiled_is_batch_invariant(self):
        torch.manual_seed(42)
        LocalCompileConfig(regions=["rope"]).apply()
        rope = ComplexRoPE.Config(dim=64, max_context_length=128).build().cuda()
        query = torch.randn(64, 8, 64, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(64, 1, 64, device="cuda", dtype=torch.bfloat16)
        positions = torch.randperm(128, device="cuda")[:64]
        grad_query = torch.randn_like(query)
        grad_key = torch.randn_like(key)

        def run(query, key, positions, grad_query, grad_key):
            query = query.detach().clone().requires_grad_()
            key = key.detach().clone().requires_grad_()
            outputs = rope(query, key, positions)
            grads = torch.autograd.grad(outputs, (query, key), (grad_query, grad_key))
            return (*outputs, *grads)

        full = run(query, key, positions, grad_query, grad_key)
        first, second = (
            run(*inputs)
            for inputs in zip(
                query.chunk(2),
                key.chunk(2),
                positions.chunk(2),
                grad_query.chunk(2),
                grad_key.chunk(2),
                strict=True,
            )
        )
        for whole, *parts in zip(full, first, second, strict=True):
            self.assertTrue(torch.equal(whole, torch.cat(parts)))

    def test_compiled_accepts_complex_cache(self):
        # Direct callers (the vision encoders) pass the complex cache, not the real view.
        torch.manual_seed(42)
        rope = ComplexRoPE.Config(dim=64, max_context_length=128).build().cuda()
        query, key = (
            torch.randn(64, 8, 64, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        )
        cache = rope.cache[torch.randperm(128, device="cuda")[:64]].unsqueeze(1)
        compiled = torch.compile(ComplexRoPE.apply_rotary_emb, fullgraph=True)
        for out, ref in zip(
            compiled(query, key, cache),
            ComplexRoPE.apply_rotary_emb(query, key, cache),
            strict=True,
        ):
            torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    unittest.main()
