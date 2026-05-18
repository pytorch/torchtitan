# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest

import torch

from torchtitan.models.common.rope import (
    _maybe_check_max_pos,
    apply_rotary_emb_complex,
    apply_rotary_emb_cos_sin,
    RoPE,
)


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
        rope_cfg = RoPE.Config(
            dim=self.head_dim, max_seq_len=self.max_seq_len, backend="complex"
        )
        rope = rope_cfg.build()
        self.freqs_cis = rope.cache

    def test_valid_positions(self):
        bsz, seqlen = 2, 8
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.arange(seqlen).unsqueeze(0).expand(bsz, -1)
        apply_rotary_emb_complex(xq, xk, self.freqs_cis, positions)

    def test_out_of_range_positions_raises(self):
        bsz, seqlen = 1, 4
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.tensor([[0, 1, self.max_seq_len, self.max_seq_len + 1]])
        with self.assertRaises(RuntimeError):
            apply_rotary_emb_complex(xq, xk, self.freqs_cis, positions)


class TestRoPEPositionBoundsCosSin(unittest.TestCase):
    """RoPE cos/sin-format apply must reject out-of-range positions."""

    def setUp(self):
        torch.manual_seed(42)
        self.head_dim = 64
        self.max_seq_len = 32
        rope_cfg = RoPE.Config(
            dim=self.head_dim, max_seq_len=self.max_seq_len, backend="cos_sin"
        )
        rope = rope_cfg.build()
        self.rope_cache = rope.cache

    def test_valid_positions(self):
        bsz, seqlen = 2, 8
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.arange(seqlen).unsqueeze(0).expand(bsz, -1)
        apply_rotary_emb_cos_sin(xq, xk, self.rope_cache, positions)

    def test_out_of_range_positions_raises(self):
        bsz, seqlen = 1, 4
        xq = torch.randn(bsz, seqlen, 4, self.head_dim)
        xk = torch.randn(bsz, seqlen, 4, self.head_dim)
        positions = torch.tensor([[0, 1, self.max_seq_len, self.max_seq_len + 1]])
        with self.assertRaises(RuntimeError):
            apply_rotary_emb_cos_sin(xq, xk, self.rope_cache, positions)


class TestUpdateFromConfigSeqLenValidation(unittest.TestCase):
    """update_from_config must reject seq_len > rope.max_seq_len."""

    def _make_trainer_config(self, seq_len):
        from types import SimpleNamespace

        from torchtitan.config import ParallelismConfig, TrainingConfig

        return SimpleNamespace(
            training=dataclasses.replace(TrainingConfig(), seq_len=seq_len),
            parallelism=ParallelismConfig(),
        )

    def _make_config(self):
        """Build a minimal Llama3 debug config."""
        from torchtitan.models.llama3 import llama3_configs

        return llama3_configs["debugmodel"]("sdpa")

    def test_rejects_oversized_seq_len(self):
        cfg = self._make_config()
        rope_max = cfg.rope.max_seq_len
        with self.assertRaises(ValueError):
            cfg.update_from_config(
                trainer_config=self._make_trainer_config(rope_max + 1)
            )

    def test_accepts_valid_seq_len(self):
        cfg = self._make_config()
        rope_max = cfg.rope.max_seq_len
        cfg.update_from_config(trainer_config=self._make_trainer_config(rope_max))
        self.assertEqual(cfg.rope.max_seq_len, rope_max)

    def test_training_none_preserves_intrinsic_max(self):
        """When training is None, rope.max_seq_len stays at the model default."""
        from types import SimpleNamespace

        from torchtitan.config import ParallelismConfig

        cfg = self._make_config()
        original_max = cfg.rope.max_seq_len
        cfg.update_from_config(
            trainer_config=SimpleNamespace(parallelism=ParallelismConfig())
        )
        self.assertEqual(cfg.rope.max_seq_len, original_max)


if __name__ == "__main__":
    unittest.main()
