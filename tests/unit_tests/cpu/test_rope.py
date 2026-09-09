# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import math
import unittest
from unittest.mock import patch

import torch
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    GQAttention,
    QKVLinear,
    VarlenAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rope import (
    _maybe_check_max_pos,
    _yarn_inv_freq,
    ComplexRoPE,
    CosSinRoPE,
    RoPE,
)
from torchtitan.models.qwen3_5.rope import MRoPE


class TestApplyRotaryEmbCosSin(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.num_tokens = 16
        self.n_heads = 4
        self.head_dim = 64
        self.xq = torch.randn(
            self.num_tokens, self.n_heads, self.head_dim, dtype=torch.bfloat16
        )
        self.xk = torch.randn(
            self.num_tokens, self.n_heads, self.head_dim, dtype=torch.bfloat16
        )
        self.rope_cache = torch.randn(
            self.num_tokens, 1, self.head_dim * 2, dtype=torch.float32
        )
        self.rope = CosSinRoPE(
            CosSinRoPE.Config(dim=self.head_dim, max_context_length=self.num_tokens)
        )

    def test_output_dtype_matches_input(self):
        xq_out, xk_out = self.rope.apply_rotary_emb(
            self.xq,
            self.xk,
            self.rope_cache,
        )
        self.assertEqual(xq_out.dtype, self.xq.dtype)
        self.assertEqual(xk_out.dtype, self.xk.dtype)

    def test_output_shape_matches_input(self):
        xq_out, xk_out = self.rope.apply_rotary_emb(
            self.xq,
            self.xk,
            self.rope_cache,
        )
        self.assertEqual(xq_out.shape, self.xq.shape)
        self.assertEqual(xk_out.shape, self.xk.shape)

    def test_computes_in_fp32(self):
        """Output must match a reference computed entirely in float32.

        Ensures inductor cannot fuse away the fp32 upcast when compiling
        adjacent ops (e.g. q_norm/k_norm) with the RoPE computation.
        """
        xq_out, xk_out = self.rope.apply_rotary_emb(
            self.xq,
            self.xk,
            self.rope_cache,
        )

        cos = self.rope_cache[..., : self.head_dim]
        sin = self.rope_cache[..., self.head_dim :]

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


class TestMaybeCheckMaxPos(unittest.TestCase):
    """Tests for the _maybe_check_max_pos bounds check."""

    def test_positions_within_bounds(self):
        positions = torch.tensor([0, 1, 2, 3])
        _maybe_check_max_pos(positions, max_valid_pos=3)

    def test_positions_at_boundary(self):
        positions = torch.tensor([0, 5, 10, 15])
        _maybe_check_max_pos(positions, max_valid_pos=15)

    def test_positions_out_of_bounds_raises(self):
        positions = torch.tensor([0, 1, 2, 16])
        with self.assertRaises(RuntimeError):
            _maybe_check_max_pos(positions, max_valid_pos=15)
            torch.cuda.synchronize() if torch.cuda.is_available() else None


class TestRoPEPositionBoundsComplex(unittest.TestCase):
    """RoPE complex-format apply must reject out-of-range positions."""

    def setUp(self):
        torch.manual_seed(42)
        self.head_dim = 64
        self.max_context_length = 32
        rope_cfg = ComplexRoPE.Config(
            dim=self.head_dim, max_context_length=self.max_context_length
        )
        self.rope = rope_cfg.build()
        self.assertIsInstance(self.rope, ComplexRoPE)

    def test_valid_positions(self):
        num_tokens = 16
        xq = torch.randn(num_tokens, 4, self.head_dim)
        xk = torch.randn(num_tokens, 4, self.head_dim)
        positions = torch.arange(num_tokens) % 8
        self.rope(xq, xk, positions)

    def test_out_of_range_positions_raises(self):
        num_tokens = 4
        xq = torch.randn(num_tokens, 4, self.head_dim)
        xk = torch.randn(num_tokens, 4, self.head_dim)
        positions = torch.tensor(
            [0, 1, self.max_context_length, self.max_context_length + 1]
        )
        with self.assertRaises(RuntimeError):
            self.rope(xq, xk, positions)


class TestRoPEPositionBoundsCosSin(unittest.TestCase):
    """RoPE cos/sin-format apply must reject out-of-range positions."""

    def setUp(self):
        torch.manual_seed(42)
        self.head_dim = 64
        self.max_context_length = 32
        rope_cfg = CosSinRoPE.Config(
            dim=self.head_dim, max_context_length=self.max_context_length
        )
        self.rope = rope_cfg.build()
        self.assertIsInstance(self.rope, CosSinRoPE)

    def test_valid_positions(self):
        num_tokens = 16
        xq = torch.randn(num_tokens, 4, self.head_dim)
        xk = torch.randn(num_tokens, 4, self.head_dim)
        positions = torch.arange(num_tokens) % 8
        self.rope(xq, xk, positions)

    def test_out_of_range_positions_raises(self):
        num_tokens = 4
        xq = torch.randn(num_tokens, 4, self.head_dim)
        xk = torch.randn(num_tokens, 4, self.head_dim)
        positions = torch.tensor(
            [0, 1, self.max_context_length, self.max_context_length + 1]
        )
        with self.assertRaises(RuntimeError):
            self.rope(xq, xk, positions)


class TestMRoPECache(unittest.TestCase):
    def test_rejects_invalid_sections(self):
        for sections, error in (
            ([2, 1], "must have 3 entries"),
            ([4, 3, -1], "must be non-negative"),
            ([1, 1, 1], "must sum to dim // 2"),
        ):
            with self.subTest(sections=sections):
                with self.assertRaisesRegex(ValueError, error):
                    MRoPE.Config(
                        dim=12,
                        max_context_length=8,
                        mrope_section=sections,
                    ).build()

    def test_rejects_invalid_position_width(self):
        num_tokens, head_dim = 2, 12
        rope = MRoPE.Config(
            dim=head_dim,
            max_context_length=8,
            mrope_section=[2, 2, 2],
        ).build()
        x = torch.randn(num_tokens, 1, head_dim)

        for width in (2, 4):
            with self.subTest(width=width):
                with self.assertRaisesRegex(ValueError, "must have shape"):
                    rope(x, x, torch.zeros(num_tokens, width, dtype=torch.long))

    def test_forward_accepts_three_axis_positions(self):
        torch.manual_seed(42)
        num_tokens, n_heads = 6, 4
        head_dim = 12
        rope = MRoPE.Config(
            dim=head_dim,
            max_context_length=8,
            mrope_section=[2, 2, 2],
        ).build()
        # (tokens, 3): per-token [temporal, height, width] positions.
        position_ids = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6],
                [5, 6, 7],
            ]
        )
        xq = torch.randn(num_tokens, n_heads, head_dim)
        xk = torch.randn(num_tokens, n_heads, head_dim)

        xq_out, xk_out = rope(xq, xk, position_ids)

        self.assertEqual(xq_out.shape, xq.shape)
        self.assertEqual(xk_out.shape, xk.shape)


class TestYaRNScaling(unittest.TestCase):
    """YaRN follows the explicit scaling policy, not the cache length.

    The cache is sized to the training sequence length, which can be shorter
    than ``original_seq_len`` (e.g. fine-tuning a YaRN checkpoint on short
    sequences), so it must not decide whether YaRN applies.
    """

    def test_zero_lower_correction_boundary(self):
        dim = 128
        rope_factor = 40.0
        inv_freq = _yarn_inv_freq(
            dim=dim,
            base=10000.0,
            rope_factor=rope_factor,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=64,
            truncate=True,
        )
        unscaled_inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        self.assertEqual(inv_freq.shape, (dim // 2,))
        torch.testing.assert_close(inv_freq[0], unscaled_inv_freq[0])
        torch.testing.assert_close(inv_freq[17], unscaled_inv_freq[17] / rope_factor)

    def test_complex_rope_applies_below_original_sequence_length(self):
        yarn = ComplexRoPE.Config(
            dim=64,
            max_context_length=2048,
            scaling="yarn",
            rope_factor=40.0,
            original_seq_len=4096,
        ).build()
        unscaled = ComplexRoPE.Config(dim=64, max_context_length=2048).build()

        self.assertFalse(torch.equal(yarn.cache[1], unscaled.cache[1]))

    def test_deepseek_mscale_applies_below_original_sequence_length(self):
        from torchtitan.models.deepseek_v3 import deepseekv3_configs
        from torchtitan.models.deepseek_v3.model import Attention

        build_config, max_context_length = deepseekv3_configs["debugmodel"]
        model_config = build_config("flex", "standard", seq_len=max_context_length)
        attention_config = model_config.layers[0].attention
        assert isinstance(attention_config, Attention.Config)
        attention_config.rope = dataclasses.replace(
            attention_config.rope,
            max_context_length=attention_config.rope.original_seq_len // 2,
        )
        attention = attention_config.build()

        expected_mscale = (
            0.1 * attention_config.mscale * math.log(attention_config.rope.rope_factor)
            + 1.0
        )
        expected_softmax_scale = attention.qk_head_dim**-0.5 * expected_mscale**2
        self.assertAlmostEqual(attention.softmax_scale, expected_softmax_scale)


class TestPerLayerRoPECache(unittest.TestCase):
    def test_gqa_attention_uses_layer_rope_cache(self):
        torch.manual_seed(42)
        dim = 8
        head_dim = 4
        attention = GQAttention.Config(
            n_heads=2,
            n_kv_heads=2,
            head_dim=head_dim,
            dim=dim,
            qkv_linear=QKVLinear.Config(
                head_dim=head_dim,
                wq=Linear.Config(in_features=dim, out_features=dim),
                wkv=Linear.Config(in_features=dim, out_features=dim),
            ),
            wo=Linear.Config(in_features=dim, out_features=dim),
            inner_attention=VarlenAttention.Config(),
            rope=ComplexRoPE.Config(dim=head_dim, max_context_length=16),
        ).build()

        x = torch.randn(8, dim)
        positions = torch.arange(8)
        attention_masks = create_varlen_metadata_for_document(positions)

        with patch(
            "torchtitan.models.common.attention._varlen_attn",
            side_effect=lambda q, k, v, *args, **kwargs: q,
        ):
            out = attention(x, attention_masks, positions)

        self.assertIsNotNone(attention.rope)
        self.assertEqual(out.shape, x.shape)

    def test_decoder_builds_distinct_rope_modules_per_attention_layer(self):
        from torchtitan.models.llama3 import llama3_configs

        build_config, max_context_length = llama3_configs["debugmodel"]
        model = build_config("flex", seq_len=max_context_length).build()
        layer_ropes = [layer.attention.rope for layer in model.layers.values()]

        self.assertTrue(all(isinstance(rope, RoPE) for rope in layer_ropes))
        self.assertEqual(len({id(rope) for rope in layer_ropes}), len(layer_ropes))

    def test_decoder_builds_distinct_rope_configs_per_attention_layer(self):
        from torchtitan.models.llama3 import llama3_configs

        build_config, max_context_length = llama3_configs["debugmodel"]
        cfg = build_config("flex", seq_len=max_context_length)
        layer_rope_cfgs = [layer.attention.rope for layer in cfg.layers]

        self.assertEqual(
            len({id(rope_cfg) for rope_cfg in layer_rope_cfgs}),
            len(layer_rope_cfgs),
        )


if __name__ == "__main__":
    unittest.main()
