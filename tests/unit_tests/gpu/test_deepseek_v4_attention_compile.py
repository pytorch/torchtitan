# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.deepseek_v4 import build_model_config

TOKENS = 64


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestAttentionRoPELocalCompile(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        attention_config = build_model_config("debugmodel").layers[2].attention
        with torch.device("cuda"):
            self.attention = attention_config.build()
            self.attention.init_states(buffer_device=torch.device("cuda"))
        self.positions = torch.randperm(TOKENS, device="cuda")
        heads, head_dim = self.attention.n_heads, self.attention.head_dim
        self.q = torch.randn(TOKENS, heads, head_dim, device="cuda").bfloat16()
        self.kv = torch.randn(TOKENS, head_dim, device="cuda").bfloat16()

    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def _run(self, fn, *inputs):
        inputs = [x.detach().clone().requires_grad_() for x in inputs]
        outputs = fn(*inputs, self.positions)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        generator = torch.Generator(device="cuda").manual_seed(1)
        grad_outputs = [
            torch.randn(out.shape, dtype=out.dtype, device="cuda", generator=generator)
            for out in outputs
        ]
        grads = torch.autograd.grad(outputs, inputs, grad_outputs)
        return (*outputs, *grads)

    def _compiled_and_eager(self, region, fn, *inputs):
        apply_local_compile([])
        eager = self._run(fn, *inputs)
        apply_local_compile([region])
        return self._run(fn, *inputs), eager

    def test_q_norm_rope_matches_eager(self):
        compiled, eager = self._compiled_and_eager(
            "q_norm_rope", self.attention._q_norm_rope, self.q, self.kv
        )
        # The compiled per-head RMS keeps fp32 intermediates; eager squares in bf16.
        for actual, expected in zip(compiled, eager, strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_inverse_partial_rope_matches_eager(self):
        compiled, eager = self._compiled_and_eager(
            "partial_rope", self.attention._inverse_partial_rope, self.q
        )
        for actual, expected in zip(compiled, eager, strict=True):
            torch.testing.assert_close(actual, expected)
        # o is stored group-major so wo_a's einsum runs a plain bmm.
        self.assertTrue(compiled[0].transpose(0, 1).is_contiguous())
        self.assertTrue(eager[0].transpose(0, 1).is_contiguous())


if __name__ == "__main__":
    unittest.main()
