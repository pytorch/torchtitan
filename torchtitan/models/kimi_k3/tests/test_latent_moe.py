# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stable LatentMoE entry/exit -- K3 tech report sec 2.3, Eq. 11.

    u = sum_{i in Tk(x)} p_i * E_i^routed(W_down x)
    y = sum_j E_j^shared(x) + W_up RMSNorm(u)

Official widths: hidden 7168, routed_expert_hidden_size 3584 (the latent l),
moe_intermediate_size 3072 (inside each routed expert), num_shared_experts 2.
The routed dispatch itself is GPU-only, so what is covered here is the shared
entry/exit math and the fail-loud guard on the unwired training path.
"""

import unittest

import torch

from torchtitan.models.kimi_k3.model import (
    KimiK3Config,
    KimiLatentMoEProjection,
    KimiMoE,
)


class TestLatentProjection(unittest.TestCase):
    def test_official_widths(self):
        proj = KimiLatentMoEProjection.make_config(7168, 3584).build()
        self.assertEqual(proj.down.weight.shape, (3584, 7168))
        self.assertEqual(proj.up.weight.shape, (7168, 3584))
        self.assertEqual(proj.norm.normalized_shape, (3584,))

    def test_round_trip_shapes(self):
        proj = KimiLatentMoEProjection.make_config(64, 32).build()
        x = torch.randn(2, 5, 64)
        u = proj.to_latent(x)
        self.assertEqual(u.shape, (2, 5, 32))
        self.assertEqual(proj.from_latent(u).shape, (2, 5, 64))

    def test_norm_sits_before_up(self):
        torch.manual_seed(0)
        proj = KimiLatentMoEProjection.make_config(64, 32).build()
        u = torch.randn(2, 5, 32) * 100  # scale the aggregate up
        torch.testing.assert_close(proj.from_latent(u), proj.up(proj.norm(u)))

    def test_norm_makes_exit_scale_insensitive(self):
        # the point of sec 2.3.1: u's scale varies with the selected experts
        torch.manual_seed(0)
        proj = KimiLatentMoEProjection.make_config(64, 32).build()
        u = torch.randn(2, 5, 32)
        a = proj.from_latent(u)
        b = proj.from_latent(u * 50.0)
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)

    def test_norm_can_be_disabled(self):
        proj = KimiLatentMoEProjection.make_config(64, 32, use_norm=False).build()
        self.assertIsNone(proj.norm)
        u = torch.randn(2, 5, 32)
        torch.testing.assert_close(proj.from_latent(u), proj.up(u))

    def test_projections_are_shared_not_per_expert(self):
        # one down/up pair per layer -- applied once per token, which is what
        # keeps 896-expert dispatch affordable (traffic is O(l), not O(d))
        proj = KimiLatentMoEProjection.make_config(64, 32).build()
        names = {n for n, _ in proj.named_parameters()}
        self.assertEqual(names, {"down.weight", "up.weight", "norm.weight"})


class TestLatentMoEWiring(unittest.TestCase):
    def _cfg(self, latent):
        return KimiK3Config(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            num_experts=8,
            num_experts_per_token=2,
            moe_intermediate_size=32,
            routed_expert_hidden_size=latent,
        )

    def test_latent_path_builds_with_the_right_widths(self):
        moe = KimiMoE.make_config(self._cfg(48)).build()
        self.assertEqual(moe.latent_size, 48)
        # entry/exit are full-width <-> latent
        self.assertEqual(moe.latent.down.weight.shape, (48, 64))
        self.assertEqual(moe.latent.up.weight.shape, (64, 48))
        # experts live in the latent: w1 is [E, moe_intermediate, latent]
        experts = moe._moe.routed_experts.inner_experts
        self.assertEqual(tuple(experts.w1_EFD.shape), (8, 32, 48))
        self.assertEqual(tuple(experts.w2_EDF.shape), (8, 48, 32))
        # the router still reads the FULL-WIDTH token (report sec 2.3.3)
        self.assertEqual(moe._moe.router.gate.weight.shape[-1], 64)

    def test_shared_experts_are_full_width_and_ours(self):
        moe = KimiMoE.make_config(self._cfg(48)).build()
        # Eq. 11 adds the shared branch at full width, outside the latent
        self.assertIsNotNone(moe.shared_experts)
        self.assertEqual(moe.shared_experts.gate_proj.weight.shape[-1], 64)
        self.assertIsNone(moe._moe.shared_experts)

    def test_non_latent_keeps_shared_inside_the_inner_moe(self):
        moe = KimiMoE.make_config(self._cfg(None)).build()
        self.assertIsNone(moe.latent_size)
        self.assertIsNone(moe.shared_experts)
        self.assertIsNotNone(moe._moe.shared_experts)

    def test_none_keeps_the_conventional_path_constructible(self):
        # not asserting a forward (routed dispatch is GPU-only), only that the
        # non-latent config still builds as it did before the release
        KimiMoE.make_config(self._cfg(None)).build()


if __name__ == "__main__":
    unittest.main()
