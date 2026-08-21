# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated MLA output gate -- K3 tech report Eq. 7.

    y_t = W_o [ Sigmoid(W_g x_t) (.) o~_t ]

W_g is FULL RANK: one gate value per output channel of the ungated attention
output (num_heads * v_head_dim), applied before W_o. The per-head variant is
this repo's graft-preserving alternative (near-identity at step 0).
"""

import unittest

import torch

from torchtitan.models.kimi_k3.model import KimiK3Config, KimiMLAAttention

H, DV, D = 4, 16, 64


def _cfg(param):
    return KimiK3Config(
        vocab_size=128,
        hidden_size=D,
        num_hidden_layers=2,
        num_attention_heads=H,
        num_key_value_heads=H,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=DV,
        mla_gated=True,
        attn_gate_param=param,
    )


class TestAttnGate(unittest.TestCase):
    def test_full_rank_shape_is_per_channel_no_bias(self):
        attn = KimiMLAAttention.make_config(_cfg("full_rank"), layer_idx=0).build()
        self.assertEqual(attn.attn_gate_proj.weight.shape, (H * DV, D))
        self.assertIsNone(attn.attn_gate_proj.bias)

    def test_per_head_shape_has_bias(self):
        attn = KimiMLAAttention.make_config(_cfg("per_head_graft"), layer_idx=0).build()
        self.assertEqual(attn.attn_gate_proj.weight.shape, (H, D))
        self.assertIsNotNone(attn.attn_gate_proj.bias)

    def test_full_rank_gate_equals_sigmoid_projection(self):
        torch.manual_seed(0)
        attn = KimiMLAAttention.make_config(_cfg("full_rank"), layer_idx=0).build()
        x = torch.randn(2, 3, D)
        torch.testing.assert_close(
            attn._attn_gate(x, H * DV), torch.sigmoid(attn.attn_gate_proj(x))
        )

    def test_per_head_gate_expands_across_v_head_dim(self):
        torch.manual_seed(0)
        attn = KimiMLAAttention.make_config(_cfg("per_head_graft"), layer_idx=0).build()
        x = torch.randn(2, 3, D)
        g = attn._attn_gate(x, H * DV)
        self.assertEqual(g.shape, (2, 3, H * DV))
        per_head = torch.sigmoid(attn.attn_gate_proj(x))
        # every DV-wide slice repeats that head's single value
        for h in range(H):
            sl = g[..., h * DV : (h + 1) * DV]
            torch.testing.assert_close(sl, per_head[..., h : h + 1].expand_as(sl))

    def test_forward_runs_both_params(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, D)
        for param in ("full_rank", "per_head_graft"):
            attn = KimiMLAAttention.make_config(_cfg(param), layer_idx=0).build()
            out = attn(x)
            out = out[0] if isinstance(out, tuple) else out
            self.assertEqual(out.shape, (2, 5, D), param)
            self.assertTrue(torch.isfinite(out).all(), param)

    def test_gate_actually_modulates(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, D)
        gated = KimiMLAAttention.make_config(_cfg("full_rank"), layer_idx=0).build()
        plain_cfg = _cfg("full_rank")
        plain_cfg.mla_gated = False
        plain = KimiMLAAttention.make_config(plain_cfg, layer_idx=0).build()
        plain.load_state_dict(
            {k: v for k, v in gated.state_dict().items() if "attn_gate" not in k}
        )
        a = gated(x)
        b = plain(x)
        a = a[0] if isinstance(a, tuple) else a
        b = b[0] if isinstance(b, tuple) else b
        self.assertFalse(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
