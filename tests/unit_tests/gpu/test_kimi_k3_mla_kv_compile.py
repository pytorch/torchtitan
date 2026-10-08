# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.kimi_k3 import build_model_config

TOKENS = 64


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMLAKVLocalCompile(unittest.TestCase):
    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def test_compiled_matches_eager_bitwise(self):
        torch.manual_seed(0)
        layers = build_model_config("debugmodel", attn_backend="flex").layers
        attention_config = next(
            layer.attention for layer in layers if layer.attention is not None
        )
        with torch.device("cuda"):
            attention = attention_config.build()
        kv_width = attention_config.qk_nope_head_dim + attention_config.v_head_dim
        kv = torch.randn(
            TOKENS, attention_config.n_heads, kv_width, device="cuda"
        ).bfloat16()
        k_rope = torch.randn(
            TOKENS, attention_config.qk_rope_head_dim, device="cuda"
        ).bfloat16()

        def run():
            inputs = [kv.clone().requires_grad_(), k_rope.clone().requires_grad_()]
            k, v = attention._assemble_kv(*inputs)
            generator = torch.Generator(device="cuda").manual_seed(1)
            grad_k = torch.randn(
                k.shape, dtype=k.dtype, device="cuda", generator=generator
            )
            grad_v = torch.randn(
                v.shape, dtype=v.dtype, device="cuda", generator=generator
            )
            grads = torch.autograd.grad((k, v), inputs, (grad_k, grad_v))
            return (k, v, *grads)

        apply_local_compile([])
        eager = run()
        apply_local_compile(["mla_kv"])
        compiled = run()
        # Pure data movement plus a sum over heads for k_rope's gradient.
        for actual, expected in zip(compiled, eager, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
