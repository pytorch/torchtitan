# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3's routed experts must use SiTU-GLU, not the core SwiGLU.

hidden_act="situ" in the released config applies globally, and the routed
experts hold the overwhelming majority of the parameters -- silently running
them as SwiGLU would be the single largest fidelity error in the stack while
still training to a plausible-looking loss.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.models.common.moe import GroupedExperts

from torchtitan.models.kimi_k3.model import KimiMoE, situ_and_mul
from torchtitan.models.kimi_k3.model_configs import build_kimi_linear_config
from torchtitan.models.kimi_k3.moe import KimiSiTUGroupedExperts


class TestSiTUGroupedExperts(unittest.TestCase):
    def test_k3_flavors_build_situ_experts(self):
        for size in ("2p8t", "k3mini"):
            cfg = build_kimi_linear_config(size, vocab_size=256)
            self.assertEqual(cfg.hidden_act, "situ", size)
            with torch.device("meta"):
                moe = KimiMoE.make_config(cfg).build()
            experts = moe._moe.routed_experts.inner_experts
            self.assertIsInstance(experts, KimiSiTUGroupedExperts, size)
            self.assertEqual(experts.situ_beta, 4.0)
            self.assertEqual(experts.situ_linear_beta, 25.0)

    def test_non_k3_flavors_keep_core_swiglu_experts(self):
        cfg = build_kimi_linear_config("48b", vocab_size=256)
        self.assertEqual(cfg.hidden_act, "silu")
        with torch.device("meta"):
            moe = KimiMoE.make_config(cfg).build()
        experts = moe._moe.routed_experts.inner_experts
        self.assertIsInstance(experts, GroupedExperts)
        self.assertNotIsInstance(experts, KimiSiTUGroupedExperts)

    def test_param_names_unchanged_so_adapters_keep_working(self):
        # the whole point of subclassing rather than a new module: the
        # state-dict adapter, expert TP/EP layout, and torchao expert
        # converters all key off these names.
        cfg = build_kimi_linear_config("k3mini", vocab_size=256)
        with torch.device("meta"):
            moe = KimiMoE.make_config(cfg).build()
        names = {n for n, _ in moe._moe.routed_experts.inner_experts.named_parameters()}
        self.assertEqual(names, {"w1_EFD", "w2_EDF", "w3_EFD"})

    @unittest.skipUnless(torch.cuda.is_available(), "grouped_mm needs CUDA")
    def test_forward_matches_hand_computed_situ(self):
        torch.manual_seed(0)
        cfg = KimiSiTUGroupedExperts.Config(dim=32, hidden_dim=64, num_experts=2)
        experts = KimiSiTUGroupedExperts(cfg).cuda()
        # std chosen so pre-activations reach O(8) > situ_beta=4: SiTU only
        # differs from SiLU once the tanh clip engages (see
        # test_situ_matches_silu_below_the_clip), so a small init would make
        # the SwiGLU control below vacuous.
        for p in experts.parameters():
            torch.nn.init.normal_(p, std=1.5)
        x_RD = torch.randn(6, 32, device="cuda", dtype=torch.bfloat16)
        counts = torch.tensor([4, 2], device="cuda", dtype=torch.int32)

        got = experts(x_RD, counts)

        # reference: per-expert dense SiTU-GLU over that expert's token slice
        ref = torch.empty_like(got)
        start = 0
        for e, n in enumerate(counts.tolist()):
            xs = x_RD[start : start + n]
            # the module casts weights to bf16 for grouped_mm; mirror that
            gate = xs @ experts.w1_EFD[e].bfloat16().transpose(0, 1)
            up = xs @ experts.w3_EFD[e].bfloat16().transpose(0, 1)
            h = situ_and_mul(gate, up, 4.0, 25.0)
            ref[start : start + n] = h @ experts.w2_EDF[e].bfloat16().transpose(0, 1)
            start += n
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        self.assertLess(rel, 2e-2, f"SiTU expert forward mismatch: {rel:.3e}")

        # and it must NOT equal the SwiGLU the core class would compute
        swiglu_cfg = GroupedExperts.Config(dim=32, hidden_dim=64, num_experts=2)
        core = GroupedExperts(swiglu_cfg).cuda()
        with torch.no_grad():
            for n in ("w1_EFD", "w2_EDF", "w3_EFD"):
                getattr(core, n).copy_(getattr(experts, n))
        swiglu_out = core(x_RD, counts)
        diff = ((got.float() - swiglu_out.float()).norm() / got.float().norm()).item()
        self.assertGreater(
            diff, 0.1, "SiTU and SwiGLU experts agree -- test has no power"
        )

    def test_situ_matches_silu_below_the_clip(self):
        """SiTU is a soft-clipped SiLU: they coincide for |g| << beta.

        This is why the SwiGLU control above needs a large init, and it is
        also the reason SiTU can replace SiLU without retuning init scale --
        at K3's initializer_range the two are numerically close, and the
        clip only matters for the outliers it exists to bound.
        """
        g = torch.linspace(-1.0, 1.0, 64)
        u = torch.ones_like(g)
        situ = situ_and_mul(g, u, 4.0, None)
        silu = torch.nn.functional.silu(g)
        self.assertLess(((situ - silu).abs() / (silu.abs() + 1e-6)).max(), 0.03)

        # and the bound the report claims: |f| <= beta1 * beta2 = 100
        big = torch.linspace(-500.0, 500.0, 1024)
        out = situ_and_mul(big, big, 4.0, 25.0)
        self.assertLessEqual(out.abs().max().item(), 100.0 + 1e-3)

    def test_situ_without_latent_shared_experts_is_rejected(self):
        cfg = build_kimi_linear_config("k3mini", vocab_size=256)
        cfg.routed_expert_hidden_size = None  # non-latent + situ + shared
        with self.assertRaisesRegex(ValueError, "latent MoE path"):
            with torch.device("meta"):
                KimiMoE.make_config(cfg).build()


if __name__ == "__main__":
    unittest.main()
