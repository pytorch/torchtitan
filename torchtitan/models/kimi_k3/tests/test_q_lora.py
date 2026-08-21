# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MLA Q-compression (q_lora_rank), K3's official config ships 1536.

Before the official release this port asserted ``q_lora_rank is None`` (the
48B-A3B path), so a K3 config could not even be constructed. The compression
pair mirrors DSv3's wq_a/wq_b and this class's own KV pair:
``q_a_proj -> q_a_layernorm -> q_b_proj``.
"""

import unittest

import torch

from torchtitan.models.kimi_k3.model import KimiK3Config, KimiMLAAttention


def _cfg(q_lora_rank):
    return KimiK3Config(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
    )


class TestMLAQLoRA(unittest.TestCase):
    def test_none_path_unchanged(self):
        attn = KimiMLAAttention.make_config(_cfg(None), layer_idx=0).build()
        self.assertTrue(hasattr(attn, "q_proj"))
        self.assertFalse(hasattr(attn, "q_a_proj"))
        self.assertEqual(attn.q_proj.weight.shape, (4 * 24, 64))

    def test_compression_pair_shapes(self):
        attn = KimiMLAAttention.make_config(_cfg(48), layer_idx=0).build()
        self.assertFalse(hasattr(attn, "q_proj"))
        self.assertEqual(attn.q_a_proj.weight.shape, (48, 64))
        self.assertEqual(attn.q_b_proj.weight.shape, (4 * 24, 48))
        self.assertEqual(attn.q_a_layernorm.normalized_shape, (48,))

    def test_project_q_matches_hand_composition(self):
        torch.manual_seed(0)
        attn = KimiMLAAttention.make_config(_cfg(48), layer_idx=0).build()
        x = torch.randn(2, 5, 64)
        expect = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(x)))
        torch.testing.assert_close(attn._project_q(x), expect)

    def test_project_q_none_path(self):
        torch.manual_seed(0)
        attn = KimiMLAAttention.make_config(_cfg(None), layer_idx=0).build()
        x = torch.randn(2, 5, 64)
        torch.testing.assert_close(attn._project_q(x), attn.q_proj(x))

    def test_forward_runs_with_compression(self):
        torch.manual_seed(0)
        attn = KimiMLAAttention.make_config(_cfg(48), layer_idx=0).build()
        x = torch.randn(2, 6, 64)
        out = attn(x)
        out = out[0] if isinstance(out, tuple) else out
        self.assertEqual(out.shape, (2, 6, 64))
        self.assertTrue(torch.isfinite(out).all())

    def test_both_paths_same_output_shape(self):
        torch.manual_seed(0)
        x = torch.randn(2, 6, 64)
        a = KimiMLAAttention.make_config(_cfg(None), layer_idx=0).build()(x)
        b = KimiMLAAttention.make_config(_cfg(48), layer_idx=0).build()(x)
        a = a[0] if isinstance(a, tuple) else a
        b = b[0] if isinstance(b, tuple) else b
        self.assertEqual(a.shape, b.shape)

    def test_official_k3_rank_builds(self):
        # the exact value in the official config.json
        attn = KimiMLAAttention.make_config(_cfg(1536), layer_idx=0).build()
        self.assertEqual(attn.q_a_proj.weight.shape, (1536, 64))


if __name__ == "__main__":
    unittest.main()
