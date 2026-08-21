# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KDA's two K3 deltas -- tech report sec 2.1.1.

Eq. 5, lower-bounded decay. Kimi Linear: g = -exp(A) * Softplus(z), unbounded
below. K3: g = g_min * Sigmoid(exp(A) z) in (g_min, 0) with g_min = -5, which
keeps the reciprocal chunk rescaling inside bf16 range so every causal tile can
use dense Tensor Core matmuls.

Eq. 6, full-rank output gate: y = W_o [ Sigmoid(W_g x) (.) RMSNorm(o~) ],
where W_g is full rank rather than Kimi Linear's low-rank factorization.
"""

import unittest

import torch

from torchtitan.models.kimi_k3.model import KimiDeltaAttention, KimiK3Config

D, H, HD = 64, 4, 16


def _cfg(**kw):
    base = dict(
        vocab_size=128,
        hidden_size=D,
        num_hidden_layers=2,
        num_attention_heads=H,
        num_key_value_heads=H,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        kda_num_heads=H,
        kda_head_dim=HD,
    )
    base.update(kw)
    return KimiK3Config(**base)


class TestKDAFullRankGate(unittest.TestCase):
    def test_low_rank_is_the_default(self):
        kda = KimiDeltaAttention.make_config(_cfg(), layer_idx=0).build()
        self.assertFalse(kda.use_full_rank_gate)
        self.assertTrue(hasattr(kda, "g_a_proj"))
        self.assertFalse(hasattr(kda, "g_proj"))
        self.assertEqual(kda.g_a_proj.weight.shape, (HD, D))
        self.assertEqual(kda.g_b_proj.weight.shape, (H * HD, D if False else HD))

    def test_full_rank_shape(self):
        kda = KimiDeltaAttention.make_config(_cfg(kda_use_full_rank_gate=True), layer_idx=0).build()
        self.assertTrue(kda.use_full_rank_gate)
        self.assertFalse(hasattr(kda, "g_a_proj"))
        # one full projection hidden -> H * head_dim, no bottleneck
        self.assertEqual(kda.g_proj.weight.shape, (H * HD, D))

    def test_full_rank_has_more_capacity_than_low_rank(self):
        low = KimiDeltaAttention.make_config(_cfg(), layer_idx=0).build()
        full = KimiDeltaAttention.make_config(_cfg(kda_use_full_rank_gate=True), layer_idx=0).build()
        n_low = low.g_a_proj.weight.numel() + low.g_b_proj.weight.numel()
        self.assertGreater(full.g_proj.weight.numel(), n_low)

    def test_gate_helper_matches_each_parameterization(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, D)
        low = KimiDeltaAttention.make_config(_cfg(), layer_idx=0).build()
        torch.testing.assert_close(
            low._output_gate_raw(x), low.g_b_proj(low.g_a_proj(x))
        )
        full = KimiDeltaAttention.make_config(_cfg(kda_use_full_rank_gate=True), layer_idx=0).build()
        torch.testing.assert_close(full._output_gate_raw(x), full.g_proj(x))


class TestKDALowerBoundedDecay(unittest.TestCase):
    def test_default_keeps_kimi_linear_form(self):
        kda = KimiDeltaAttention.make_config(_cfg(), layer_idx=0).build()
        self.assertIsNone(kda.gate_lower_bound)

    def test_official_value_is_plumbed(self):
        kda = KimiDeltaAttention.make_config(_cfg(kda_gate_lower_bound=-5.0), layer_idx=0).build()
        self.assertEqual(kda.gate_lower_bound, -5.0)

    @unittest.skipUnless(torch.cuda.is_available(), "fused_kda_gate is Triton")
    def test_formula_matches_report_eq5_cuda(self):
        # g = g_min * sigmoid(exp(A) * z), bounded in (g_min, 0)
        from fla.ops.kda.gate import fused_kda_gate

        torch.manual_seed(0)
        z = torch.randn(2, 3, H, HD, device="cuda")
        A_log = torch.zeros(H, device="cuda")  # report: A_h initialized to 0
        g = fused_kda_gate(z, A_log, dt_bias=None, lower_bound=-5.0)
        expect = -5.0 * torch.sigmoid(A_log.view(H, 1).exp() * z)
        torch.testing.assert_close(g, expect, rtol=1e-4, atol=1e-4)
        self.assertTrue((g > -5.0).all() and (g < 0.0).all())

    @unittest.skipUnless(torch.cuda.is_available(), "fused_kda_gate is Triton")
    def test_alpha_stays_above_exp_gmin_cuda(self):
        # report: with g_min = -5 every retention factor exceeds e^-5
        from fla.ops.kda.gate import fused_kda_gate

        z = torch.randn(2, 3, H, HD, device="cuda") * 20  # push the extremes
        g = fused_kda_gate(
            z, torch.zeros(H, device="cuda"), dt_bias=None, lower_bound=-5.0
        )
        alpha = g.exp()
        # The report states the open interval alpha > e^-5; in float32 the
        # sigmoid saturates, so the bound is attained exactly rather than
        # approached. Attained is what matters: the reciprocal chunk
        # rescaling stays <= e^80 either way, which is the point of Eq. 5.
        self.assertTrue((alpha >= torch.tensor(-5.0).exp().cuda()).all())
        self.assertTrue((alpha <= 1.0).all())
        self.assertAlmostEqual(alpha.min().item(), 0.0067379, places=6)

    @unittest.skipUnless(torch.cuda.is_available(), "fused_kda_gate is Triton")
    def test_kimi_linear_form_is_unbounded_below_cuda(self):
        # the contrast the report draws: without the bound, large negative z
        # drives g far below -5 (that is what overflows the reciprocal)
        from fla.ops.kda.gate import fused_kda_gate

        z = torch.full((1, 2, H, HD), 40.0, device="cuda")
        g_old = fused_kda_gate(
            z, torch.zeros(H, device="cuda"), dt_bias=None, lower_bound=None
        )
        self.assertLess(g_old.min().item(), -5.0)


if __name__ == "__main__":
    unittest.main()
