# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.rope import apply_rotary_emb_cos_sin


class TestApplyRotaryEmbCosSin(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.bsz = 2
        self.seqlen = 16
        self.n_heads = 4
        self.head_dim = 64
        self.xq = torch.randn(
            self.bsz, self.seqlen, self.n_heads, self.head_dim, dtype=torch.bfloat16
        )
        self.xk = torch.randn(
            self.bsz, self.seqlen, self.n_heads, self.head_dim, dtype=torch.bfloat16
        )
        self.rope_cache = torch.randn(
            self.seqlen, self.head_dim * 2, dtype=torch.float32
        )

    def test_output_dtype_matches_input(self):
        xq_out, xk_out = apply_rotary_emb_cos_sin(self.xq, self.xk, self.rope_cache)
        self.assertEqual(xq_out.dtype, self.xq.dtype)
        self.assertEqual(xk_out.dtype, self.xk.dtype)

    def test_output_shape_matches_input(self):
        xq_out, xk_out = apply_rotary_emb_cos_sin(self.xq, self.xk, self.rope_cache)
        self.assertEqual(xq_out.shape, self.xq.shape)
        self.assertEqual(xk_out.shape, self.xk.shape)

    def test_computes_in_fp32(self):
        """Output must match a reference computed entirely in float32.

        Ensures inductor cannot fuse away the fp32 upcast when compiling
        adjacent ops (e.g. q_norm/k_norm) with the RoPE computation.
        """
        xq_out, xk_out = apply_rotary_emb_cos_sin(self.xq, self.xk, self.rope_cache)

        cos = self.rope_cache[..., : self.head_dim].unsqueeze(0).unsqueeze(2)
        sin = self.rope_cache[..., self.head_dim :].unsqueeze(0).unsqueeze(2)

        def rotate_half(x):
            half = x.shape[-1] // 2
            return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

        xq_ref = (
            (self.xq.float() * cos) + (rotate_half(self.xq.float()) * sin)
        ).bfloat16()
        xk_ref = (
            (self.xk.float() * cos) + (rotate_half(self.xk.float()) * sin)
        ).bfloat16()

        self.assertEqual((xq_out - xq_ref).abs().max().item(), 0.0)
        self.assertEqual((xk_out - xk_ref).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
