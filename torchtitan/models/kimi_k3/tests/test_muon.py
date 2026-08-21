# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""(Per-Head) Muon optimizer tests.

Locks the base Muon algorithm (published; K3's exact per-head variant
reconciles at 7.27): Newton-Schulz equalizes singular values, Muon
optimizes matrices while non-2-D params take the AdamW fallback, and
the per-head path orthogonalizes head blocks independently.
"""

import re
import unittest

import torch

from torchtitan.models.kimi_k3.attn_res_model import KimiK3AttnResModel
from torchtitan.models.kimi_k3.muon import _newton_schulz, default_muon, Muon


@unittest.skipIf(not torch.cuda.is_available(), "bf16 NS on CUDA")
class TestMuon(unittest.TestCase):
    def test_newton_schulz_equalizes_singular_values(self):
        torch.manual_seed(0)
        G = torch.randn(128, 64, device="cuda")
        Q = _newton_schulz(G, steps=5)
        # Muon's NS pushes singular values toward 1 -> condition number
        # (max/min sigma) drops sharply vs the raw Gaussian matrix.
        cond_in = torch.linalg.svdvals(G.float())
        cond_out = torch.linalg.svdvals(Q.float())
        r_in = (cond_in.max() / cond_in.min()).item()
        r_out = (cond_out.max() / cond_out.min()).item()
        self.assertLess(r_out, r_in)
        self.assertLess(r_out, 3.0)  # near-orthogonal

    def test_muon_matrix_adamw_fallback(self):
        torch.manual_seed(0)
        W = torch.nn.Parameter(torch.randn(64, 32, device="cuda"))
        b = torch.nn.Parameter(torch.ones(64, device="cuda"))
        target = torch.randn(64, 32, device="cuda")
        opt = Muon([W, b], lr=0.05, adamw_lr=0.02)
        first = last = None
        for i in range(60):
            loss = (W - target).pow(2).mean() + b.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i == 0:
                first = loss.item()
            last = loss.item()
        self.assertLess(last, first)
        # AdamW fallback drove the bias vector toward 0.
        self.assertLess(b.abs().mean().item(), 1.0)

    def test_per_head_path(self):
        torch.manual_seed(0)
        W = torch.nn.Parameter(torch.randn(128, 32, device="cuda"))
        W._muon_heads = 4
        opt = Muon([W], lr=0.05, per_head=True)
        first = W.pow(2).mean().item()
        for _ in range(20):
            loss = W.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(W.pow(2).mean().item(), first)


class TestWeightDecayScope(unittest.TestCase):
    """Weight decay must not reach 1-D parameters.

    The container assigns each parameter to the first pattern that ``search``es
    its FQN, so this replicates that walk over a real model's names rather than
    asserting on the pattern strings, which would pass even if a pattern never
    matched anything.
    """

    def _assign(self, name: str):
        """The group a parameter lands in, by the container's own rule."""
        for group in default_muon().param_groups:
            if re.compile(group.pattern).search(name):
                return group
        self.fail(f"no group matched {name}")

    def _named_parameters(self):
        from torchtitan.models.kimi_k3.tests.test_kimi_attn_res_model import (
            _dense_mla_only_config,
        )

        with torch.device("meta"):
            model = KimiK3AttnResModel(
                _dense_mla_only_config(num_hidden_layers=4), num_blocks=2
            )
        return list(model.named_parameters())

    def test_no_one_dimensional_parameter_is_decayed(self):
        seen_1d = 0
        for name, param in self._named_parameters():
            if param.ndim != 1:
                continue
            seen_1d += 1
            group = self._assign(name)
            with self.subTest(name=name):
                self.assertEqual(
                    group.optimizer_kwargs.get("weight_decay", 0.0),
                    0.0,
                    f"{name} is 1-D and must not be decayed",
                )
        # Guards the walk itself: a config that produced no 1-D parameters would
        # make the assertion above vacuous.
        self.assertGreater(seen_1d, 0)

    def test_matrix_parameters_still_reach_muon(self):
        muon_named = [
            name
            for name, param in self._named_parameters()
            if self._assign(name).optimizer_name == "Muon"
        ]
        self.assertTrue(muon_named)
        # Every projection matrix, and nothing that Muon would fall back on.
        for name in muon_named:
            self.assertTrue(name.endswith(".weight"), name)

    def test_decaying_group_keeps_the_released_value(self):
        decayed = [
            g
            for g in default_muon().param_groups
            if g.optimizer_kwargs.get("weight_decay")
        ]
        self.assertEqual(len(decayed), 1)
        self.assertEqual(decayed[0].optimizer_kwargs["weight_decay"], 0.1)


if __name__ == "__main__":
    unittest.main()


class TestPerHeadMuonTagging(unittest.TestCase):
    """Per-Head Muon was inert: ``_muon_heads`` was set only inside tests, so a
    real run fell back to full-matrix orthogonalization with nothing to show it.
    These tests make the tagging and the fallback both observable."""

    def _model(self):
        import torch

        from torchtitan.models.kimi_k3.model import KimiK3Model
        from torchtitan.models.kimi_k3.model_configs import build_kimi_linear_config

        with torch.device("meta"):
            return KimiK3Model.make_config(build_kimi_linear_config("k3mini", vocab_size=256)).build()

    def _tagged(self, model):
        out = {}
        for fqn, mod in model.named_modules():
            w = getattr(mod, "weight", None)
            if w is not None and getattr(w, "_muon_heads", None):
                out[fqn] = w._muon_heads
        return out

    def test_tags_qkv_on_both_attention_types(self):
        from torchtitan.models.kimi_k3.muon import tag_per_head_muon

        model = self._model()
        self.assertEqual(self._tagged(model), {}, "nothing tagged before the call")
        n = tag_per_head_muon(model)
        tagged = self._tagged(model)
        self.assertEqual(n, len(tagged))
        leaves = {fqn.rsplit(".", 1)[1] for fqn in tagged}
        # MLA compressed-Q path and fused KV, plus KDA's q/k/v
        self.assertEqual(
            leaves, {"q_b_proj", "kv_b_proj", "q_proj", "k_proj", "v_proj"}
        )

    def test_o_proj_is_deliberately_not_tagged(self):
        # report sec 2.5 names Q, K and V. o_proj carries the head axis on its
        # INPUT side, so a row partition would not be a head partition.
        from torchtitan.models.kimi_k3.muon import tag_per_head_muon

        model = self._model()
        tag_per_head_muon(model)
        for fqn in self._tagged(model):
            self.assertFalse(fqn.endswith("o_proj"), fqn)

    def test_head_counts_divide_the_row_dimension(self):
        from torchtitan.models.kimi_k3.muon import tag_per_head_muon

        model = self._model()
        tag_per_head_muon(model)
        for fqn, heads in self._tagged(model).items():
            w = model.get_submodule(fqn).weight
            self.assertEqual(
                w.size(0) % heads, 0, f"{fqn}: {tuple(w.shape)} rows vs {heads}"
            )

    def test_tagging_is_idempotent(self):
        from torchtitan.models.kimi_k3.muon import tag_per_head_muon

        model = self._model()
        first = tag_per_head_muon(model)
        self.assertEqual(tag_per_head_muon(model), first)

    def test_untagged_group_warns_that_per_head_is_inert(self):
        import torch

        from torchtitan.models.kimi_k3.muon import Muon

        w = torch.nn.Parameter(torch.randn(8, 4))
        w.grad = torch.randn(8, 4)
        opt = Muon([w], lr=1e-3, per_head=True)
        with self.assertLogs(level="WARNING") as cm:
            opt.step()
        self.assertTrue(
            any("tag_per_head_muon" in line for line in cm.output),
            f"no actionable warning: {cm.output}",
        )

    def test_tagged_group_does_not_warn(self):
        import logging

        import torch

        from torchtitan.models.kimi_k3.muon import Muon

        w = torch.nn.Parameter(torch.randn(8, 4))
        w._muon_heads = 2
        w.grad = torch.randn(8, 4)
        opt = Muon([w], lr=1e-3, per_head=True)
        with self.assertNoLogs(level=logging.WARNING):
            opt.step()

    def test_per_head_update_differs_from_full_matrix(self):
        """If these agreed, the tagging would be cosmetic."""
        import torch

        from torchtitan.models.kimi_k3.muon import Muon

        torch.manual_seed(0)
        base = torch.randn(8, 4)
        grad = torch.randn(8, 4)

        a = torch.nn.Parameter(base.clone())
        a._muon_heads = 4
        a.grad = grad.clone()
        Muon([a], lr=0.1, per_head=True).step()

        b = torch.nn.Parameter(base.clone())
        b.grad = grad.clone()
        Muon([b], lr=0.1, per_head=False).step()

        self.assertGreater((a.data - b.data).abs().max().item(), 1e-4)
