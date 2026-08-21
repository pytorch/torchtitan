# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3's NoPE makes MLA exactly absorbable into latent-space MQA.

The decompressed form we train with expands the 512-dim KV latent to 96 full
heads through ``kv_b_proj``. The score for head h is

    q_nope_h . (W_UK[h] c) + q_rot_h . k_rot
  = (W_UK[h]^T q_nope_h) . c + q_rot_h . k_rot

so ``W_UK`` can move onto the query and the attention runs directly over the
shared latent ``c`` -- one MQA head of width kv_lora_rank + qk_rope_head_dim,
with ``W_UV`` folded out of the context afterwards. This holds ONLY because
mla_use_nope skips RoPE: a rotation between q and W_UK blocks the transpose.
That is the property that sets the KV cache at 576 values per token per layer
instead of 96 x 192, and it is what an inference backend must implement.

This test asserts the algebra on our actual module, so a change to the MLA
forward that silently breaks absorbability fails here rather than in a serving
stack months later.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.models.kimi_k3.model import KimiK3Config, KimiMLAAttention


def _cfg(**kw) -> KimiK3Config:
    base = dict(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        mla_use_nope=True,
        num_experts=None,
        num_experts_per_token=1,
        num_shared_experts=0,
        first_k_dense_replace=2,
        rms_norm_eps=1e-5,
        hidden_act="silu",
    )
    base.update(kw)
    return KimiK3Config(**base)


def _absorbed_forward(attn: KimiMLAAttention, x: torch.Tensor) -> torch.Tensor:
    """MLA computed as MQA over the latent -- no per-head K/V ever built."""
    B, T, _ = x.shape
    H, Dn, Dr, Dv = (
        attn.num_heads,
        attn.qk_nope_head_dim,
        attn.qk_rope_head_dim,
        attn.v_head_dim,
    )
    R = attn.kv_lora_rank

    q = attn._project_q(x).view(B, T, H, Dn + Dr)
    q_nope_BTHN, q_rot_BTHR = torch.split(q, [Dn, Dr], dim=-1)

    # the entire per-token cache: latent + the head-shared rot channels
    compressed = attn.kv_a_proj_with_mqa(x)
    c_BTR, k_rot_BTR2 = torch.split(compressed, [R, Dr], dim=-1)
    c_BTR = attn.kv_a_layernorm(c_BTR)

    w = attn.kv_b_proj.weight.view(H, Dn + Dv, R)
    w_uk_HNR, w_uv_HVR = torch.split(w, [Dn, Dv], dim=1)

    # absorb W_UK onto the query: q_tilde lives in the latent space
    q_tilde_BTHR = torch.einsum("bthn,hnr->bthr", q_nope_BTHN, w_uk_HNR)
    scores = (
        torch.einsum("bthr,bsr->bhts", q_tilde_BTHR, c_BTR)
        + torch.einsum("bthr,bsr->bhts", q_rot_BTHR, k_rot_BTR2)
    ) * attn.scaling
    causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    probs = scores.softmax(dim=-1)

    # attend in the latent, then decompress the CONTEXT (not the keys)
    ctx_BHTR = torch.einsum("bhts,bsr->bhtr", probs, c_BTR)
    out_BTHV = torch.einsum("bhtr,hvr->bthv", ctx_BHTR, w_uv_HVR)
    out_BTE = out_BTHV.reshape(B, T, H * Dv)
    if attn.mla_gated:
        out_BTE = out_BTE * attn._attn_gate(x, out_BTE.shape[-1])
    return attn.o_proj(out_BTE)


class TestMLALatentAbsorption(unittest.TestCase):
    def _check(self, cfg: KimiK3Config) -> float:
        torch.manual_seed(0)
        attn = KimiMLAAttention.make_config(cfg, layer_idx=0).build().double()
        for p in attn.parameters():
            torch.nn.init.normal_(p, std=0.02)
        x = torch.randn(2, 12, cfg.hidden_size, dtype=torch.float64)
        with torch.no_grad():
            ref = attn(x)
            got = _absorbed_forward(attn, x)
        rel = ((got - ref).norm() / ref.norm()).item()
        self.assertLess(rel, 1e-10, f"absorption not exact: rel {rel:.3e}")
        return rel

    def test_absorption_is_exact_gated(self):
        self._check(_cfg(mla_gated=True, attn_gate_param="full_rank"))

    def test_absorption_is_exact_ungated(self):
        self._check(_cfg(mla_gated=False))

    def test_absorption_is_exact_without_q_compression(self):
        # the 48B-A3B path: q_proj direct, no q_a/q_b pair
        self._check(_cfg(q_lora_rank=None, mla_gated=True, attn_gate_param="full_rank"))

    def test_official_kv_cache_width_is_kv_lora_plus_rot(self):
        from torchtitan.models.kimi_k3.model_configs import build_kimi_linear_config

        c = build_kimi_linear_config("2p8t")
        per_token_per_layer = c.kv_lora_rank + c.qk_rope_head_dim
        self.assertEqual(per_token_per_layer, 576)
        # vs the decompressed keys+values a non-absorbed cache would hold
        naive = c.num_attention_heads * (
            c.qk_nope_head_dim + c.qk_rope_head_dim + c.v_head_dim
        )
        self.assertEqual(naive, 96 * 320)
        self.assertGreater(naive / per_token_per_layer, 50)

    def test_gate_is_channel_wise_on_x_not_per_head(self):
        cfg = _cfg(mla_gated=True, attn_gate_param="full_rank")
        attn = KimiMLAAttention.make_config(cfg, layer_idx=0).build()
        # Eq. 7's W_g is full rank: one gate per ungated-output channel
        self.assertEqual(
            attn.attn_gate_proj.weight.shape,
            (cfg.num_attention_heads * cfg.v_head_dim, cfg.hidden_size),
        )
        self.assertFalse(attn.attn_gate_proj.bias is not None)


if __name__ == "__main__":
    unittest.main()
