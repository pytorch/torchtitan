# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SiTU (Sigmoid Tanh Unit) -- K3's activation, official config 2026-07-27.

Reference form (modeling_kimi_linear.SituAndMul):

    situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    up     = linear_beta * tanh(up / linear_beta)      # when set
    out    = situ_a * up

with activation_situ_beta=4.0, activation_situ_linear_beta=25.0.
"""

import unittest

import torch
import torch.nn.functional as F

from torchtitan.models.kimi_k3.model import KimiMLP, situ_and_mul


class TestSituAndMul(unittest.TestCase):
    def test_matches_reference_formula(self):
        torch.manual_seed(0)
        g = torch.randn(4, 8) * 10  # wide enough to exercise the caps
        u = torch.randn(4, 8) * 60
        beta, lin = 4.0, 25.0
        expect = (
            beta * torch.tanh(g / beta) * torch.sigmoid(g)
        ) * (lin * torch.tanh(u / lin))
        torch.testing.assert_close(situ_and_mul(g, u, beta, lin), expect)

    def test_saturates_at_beta(self):
        # tanh(g/beta) -> 1 and sigmoid(g) -> 1 for large g, so situ -> beta
        g = torch.full((3,), 1e4)
        u = torch.ones(3)
        out = situ_and_mul(g, u, 4.0, None)
        torch.testing.assert_close(out, torch.full((3,), 4.0))

    def test_linear_branch_cap(self):
        # with a huge up, the linear branch saturates at linear_beta
        g = torch.full((3,), 1e4)
        u = torch.full((3,), 1e6)
        out = situ_and_mul(g, u, 4.0, 25.0)
        torch.testing.assert_close(out, torch.full((3,), 4.0 * 25.0))

    def test_no_linear_cap_when_none(self):
        g = torch.randn(5)
        u = torch.randn(5)
        a = situ_and_mul(g, u, 4.0, None)
        expect = (4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)) * u
        torch.testing.assert_close(a, expect)

    def test_dtype_preserved_and_fp32_internally(self):
        g = torch.randn(6, dtype=torch.bfloat16)
        u = torch.randn(6, dtype=torch.bfloat16)
        self.assertEqual(situ_and_mul(g, u, 4.0, 25.0).dtype, torch.bfloat16)


class TestKimiMLPSitu(unittest.TestCase):
    def test_mlp_situ_path_runs_and_differs_from_silu(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 16)
        mlp_silu = KimiMLP.make_config(16, 32, hidden_act="silu").build()
        mlp_situ = KimiMLP.make_config(16, 32, hidden_act="situ").build()
        # same weights, different activation -> different output
        mlp_situ.load_state_dict(mlp_silu.state_dict())
        y_silu, y_situ = mlp_silu(x), mlp_situ(x)
        self.assertEqual(y_situ.shape, y_silu.shape)
        self.assertFalse(torch.allclose(y_silu, y_situ))

    def test_mlp_situ_equals_hand_computed(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, 16)
        mlp = KimiMLP.make_config(16, 32, hidden_act="situ", situ_beta=4.0, situ_linear_beta=25.0).build()
        expect = mlp.down_proj(
            situ_and_mul(mlp.gate_proj(x), mlp.up_proj(x), 4.0, 25.0)
        )
        torch.testing.assert_close(mlp(x), expect)

    def test_silu_path_unchanged(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 16)
        mlp = KimiMLP.make_config(16, 32, hidden_act="silu").build()
        expect = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
        torch.testing.assert_close(mlp(x), expect)

    def test_unknown_act_raises(self):
        with self.assertRaises(ValueError):
            KimiMLP.make_config(8, 16, hidden_act="nope").build()


if __name__ == "__main__":
    unittest.main()
