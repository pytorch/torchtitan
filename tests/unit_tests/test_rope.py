# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest

import torch
from torchtitan.models.common.attention import (
    GQAttention,
    QKVLinear,
    ScaledDotProductAttention,
)
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.common.rope import (
    _maybe_check_max_pos,
    ComplexRoPE,
    CosSinRoPE,
    RoPE,
)
from torchtitan.models.qwen3_5.rope import MRoPE


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
        ).view(1, self.seqlen, 1, self.head_dim * 2)
        self.rope = CosSinRoPE(
            CosSinRoPE.Config(dim=self.head_dim, max_seq_len=self.seqlen)
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
        positions = torch.tensor([[0, 1, 2, 3]])
        _maybe_check_max_pos(positions, max_valid_pos=3)

    def test_positions_at_boundary(self):
        positions = torch.tensor([[0, 5, 10, 15]])
        _maybe_check_max_pos(positions, max_valid_pos=15)

    def test_positions_out_of_bounds_raises(self):
        positions = torch.tensor([[0, 1, 2, 16]])
        with self.assertRaises(RuntimeError):
            _maybe_check_max_pos(positions, max_valid_pos=15)
            torch.cuda.synchronize() if torch.cuda.is_available() else None


class TestRoPEPositionBoundsComplex(unittest.TestCase):
    """RoPE complex-format apply must reject out-of-range positions."""

    def setUp(self):
        torch.manual_seed(42)
        self.head_dim = 64
        self.max_seq_len = 32
        rope_cfg = ComplexRoPE.Config(dim=self.head_dim, max_seq_len=self.max_seq_len)
        self.rope = rope_cfg.build()
        self.assertIsInstance(self.rope, ComplexRoPE)

    def test_valid_positions(self):
        bsz, seqlen = 2, 8
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.arange(seqlen).unsqueeze(0).expand(bsz, -1)
        self.rope(xq, xk, positions)

    def test_out_of_range_positions_raises(self):
        bsz, seqlen = 1, 4
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.tensor([[0, 1, self.max_seq_len, self.max_seq_len + 1]])
        with self.assertRaises(RuntimeError):
            self.rope(xq, xk, positions)


class TestRoPEPositionBoundsCosSin(unittest.TestCase):
    """RoPE cos/sin-format apply must reject out-of-range positions."""

    def setUp(self):
        torch.manual_seed(42)
        self.head_dim = 64
        self.max_seq_len = 32
        rope_cfg = CosSinRoPE.Config(dim=self.head_dim, max_seq_len=self.max_seq_len)
        self.rope = rope_cfg.build()
        self.assertIsInstance(self.rope, CosSinRoPE)

    def test_valid_positions(self):
        bsz, seqlen = 2, 8
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.arange(seqlen).unsqueeze(0).expand(bsz, -1)
        self.rope(xq, xk, positions)

    def test_out_of_range_positions_raises(self):
        bsz, seqlen = 1, 4
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.tensor([[0, 1, self.max_seq_len, self.max_seq_len + 1]])
        with self.assertRaises(RuntimeError):
            self.rope(xq, xk, positions)


class TestMRoPECache(unittest.TestCase):
    def test_forward_accepts_three_axis_positions(self):
        torch.manual_seed(42)
        bsz, seqlen, n_heads = 2, 3, 4
        head_dim = 12
        rope = MRoPE.Config(
            dim=head_dim,
            max_seq_len=8,
            mrope_section=[2, 1, 1],
        ).build()
        # (batch, seq, 3): per-token [temporal, height, width] positions.
        position_ids = torch.tensor(
            [
                [[0, 1, 2], [1, 2, 3], [2, 3, 4]],  # batch 0
                [[3, 4, 5], [4, 5, 6], [5, 6, 7]],  # batch 1
            ]
        )
        xq = torch.randn(bsz, seqlen, n_heads, head_dim)
        xk = torch.randn(bsz, seqlen, n_heads, head_dim)

        xq_out, xk_out = rope(xq, xk, position_ids)

        self.assertEqual(xq_out.shape, xq.shape)
        self.assertEqual(xk_out.shape, xk.shape)


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
            inner_attention=ScaledDotProductAttention.Config(),
            rope=ComplexRoPE.Config(dim=head_dim, max_seq_len=16),
        ).build()

        x = torch.randn(2, 4, dim)
        out = attention(x, None)

        self.assertIsNotNone(attention.rope)
        self.assertEqual(out.shape, x.shape)

    def test_decoder_builds_distinct_rope_modules_per_attention_layer(self):
        from torchtitan.models.llama3 import llama3_configs

        model = llama3_configs["debugmodel"]("flex").build()
        layer_ropes = [layer.attention.rope for layer in model.layers.values()]

        self.assertTrue(all(isinstance(rope, RoPE) for rope in layer_ropes))
        self.assertEqual(len({id(rope) for rope in layer_ropes}), len(layer_ropes))

    def test_decoder_builds_distinct_rope_configs_per_attention_layer(self):
        from torchtitan.models.llama3 import llama3_configs

        cfg = llama3_configs["debugmodel"]("flex")
        layer_rope_cfgs = [layer.attention.rope for layer in cfg.layers]

        self.assertEqual(
            len({id(rope_cfg) for rope_cfg in layer_rope_cfgs}),
            len(layer_rope_cfgs),
        )


class TestUpdateFromConfigSeqLenValidation(unittest.TestCase):
    """update_from_config must reject seq_len > rope.max_seq_len."""

    def _make_trainer_config(self, seq_len):
        from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
        from torchtitan.trainer import Trainer

        return Trainer.Config(
            training=dataclasses.replace(TrainingConfig(), seq_len=seq_len),
            parallelism=ParallelismConfig(),
            debug=DebugConfig(),
        )

    def _make_config(self):
        """Build a minimal Llama3 debug config."""
        from torchtitan.models.llama3 import llama3_configs

        return llama3_configs["debugmodel"]("flex")

    def test_rejects_oversized_seq_len(self):
        cfg = self._make_config()
        rope_max = cfg.max_seq_len
        with self.assertRaises(ValueError):
            cfg.update_from_config(config=self._make_trainer_config(rope_max + 1))

    def test_accepts_valid_seq_len(self):
        cfg = self._make_config()
        rope_max = cfg.max_seq_len
        cfg.update_from_config(config=self._make_trainer_config(rope_max))
        self.assertEqual(cfg.max_seq_len, rope_max)

    def test_vllm_max_model_len_as_seq_len(self):
        """vLLM wrapper translates max_model_len to TrainingConfig.seq_len.

        When seq_len equals rope.max_seq_len, the RoPE cache stays at
        the model's intrinsic maximum.
        """
        cfg = self._make_config()
        original_max = cfg.max_seq_len
        cfg.update_from_config(config=self._make_trainer_config(original_max))
        self.assertEqual(cfg.max_seq_len, original_max)


if __name__ == "__main__":
    unittest.main()
