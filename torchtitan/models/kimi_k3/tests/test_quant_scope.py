# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3 quantizes routed experts and nothing else.

Before the release our QAT and QLoRA target lists were name-based sets of MLA
and dense/shared-FFN Linears -- very nearly the COMPLEMENT of the official
scope, which ignores self_attn, shared_experts, the dense FFN, lm_head, the
latent projections and the router, and quantizes only the MoE experts. Getting
this backwards costs quality silently (quantizing layers K3 keeps in bf16) while
saving nothing where the memory actually is.
"""

from __future__ import annotations

import json
import pathlib
import unittest

import torch

from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.models.kimi_k3.model_configs import build_kimi_linear_config
from torchtitan.models.kimi_k3.moe import KimiSiTUGroupedExperts
from torchtitan.models.kimi_k3.mxfp4_qat import apply_mxfp4_qat
from torchtitan.models.kimi_k3.quant_scope import (
    is_ignored,
    is_quantizable,
    MXFP4_GROUP_SIZE,
    OFFICIAL_IGNORE_PATTERNS,
    quantizable_modules,
)

_ARTIFACT = (
    pathlib.Path(__file__).resolve().parents[5]
    / "phase13_k3like_48b_posttrain"
    / "official_k3"
    / "config.json"
)


def _k3mini_model() -> KimiK3Model:
    with torch.device("meta"):
        return KimiK3Model.make_config(build_kimi_linear_config("k3mini", vocab_size=256)).build()


class TestQuantScope(unittest.TestCase):
    def test_ignore_patterns_match_the_released_config(self):
        if not _ARTIFACT.exists():
            self.skipTest("official artifact not present")
        q = json.loads(_ARTIFACT.read_text())["text_config"]["quantization_config"]
        official = tuple(p.removeprefix("re:") for p in q["ignore"])
        self.assertEqual(official, OFFICIAL_IGNORE_PATTERNS)
        self.assertEqual(
            q["config_groups"]["group_0"]["weights"]["group_size"],
            MXFP4_GROUP_SIZE,
        )

    def test_scope_is_exactly_the_routed_experts(self):
        model = _k3mini_model()
        scoped = {fqn for fqn, _ in quantizable_modules(model)}
        self.assertTrue(scoped, "k3mini must have routed experts in scope")
        for fqn in scoped:
            self.assertTrue(fqn.endswith("routed_experts.inner_experts"), fqn)
        # every MoE layer contributes exactly one
        moe_layers = sum(
            1 for _, m in model.named_modules() if isinstance(m, KimiSiTUGroupedExperts)
        )
        self.assertEqual(len(scoped), moe_layers)

    def test_components_k3_keeps_in_high_precision_are_out_of_scope(self):
        model = _k3mini_model()
        scoped = {fqn for fqn, _ in quantizable_modules(model)}
        offenders = [
            fqn
            for fqn, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and fqn in scoped
        ]
        self.assertEqual(offenders, [], "no nn.Linear may be in K3's MXFP4 scope")
        # and the named non-expert components are explicitly ignored
        for fqn in (
            "layers.1.attention.o_proj",
            "layers.2.delta_attention.o_proj",
            "layers.1.moe.shared_experts.gate_proj",
            "layers.1.moe.latent.down",
            "layers.1.moe._moe.router.gate",
            "lm_head",
        ):
            self.assertTrue(is_ignored(fqn), f"{fqn} should be ignored")

    def test_dense_ffn_ignored_under_both_naming_conventions(self):
        # HF calls it mlp; ours is feed_forward. Both must be ignored.
        self.assertTrue(is_ignored("model.layers.0.mlp.gate_proj"))
        self.assertTrue(is_ignored("layers.0.feed_forward.gate_proj"))
        self.assertTrue(is_ignored("layers.0.feed_forward.down_proj"))

    def test_unclassified_modules_default_to_high_precision(self):
        # positive predicate: something we have never seen must NOT be
        # quantized just because no ignore pattern happens to name it.
        self.assertFalse(is_quantizable("some.new.module", torch.nn.Linear(8, 8)))

    def test_qat_default_scope_wraps_experts_only_and_is_idempotent(self):
        model = _k3mini_model()
        n = apply_mxfp4_qat(model)
        self.assertEqual(n, len(quantizable_modules(model)))
        self.assertEqual(apply_mxfp4_qat(model), 0)
        experts = model.layers["1"].moe._moe.routed_experts.inner_experts
        self.assertTrue(type(experts).__name__.startswith("MXFP4QAT"))
        # masters stay registered under their original names, so the
        # state-dict adapter and expert sharding are unaffected
        self.assertEqual(
            {n for n, _ in experts.named_parameters()},
            {"w1_EFD", "w2_EDF", "w3_EFD"},
        )
        # no Linear got wrapped
        from torchtitan.models.kimi_k3.mxfp4_qat import MXFP4QATLinear

        self.assertEqual(
            [fqn for fqn, m in model.named_modules() if isinstance(m, MXFP4QATLinear)],
            [],
        )

    def test_unknown_scope_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown scope"):
            apply_mxfp4_qat(_k3mini_model(), scope="everything")

    def test_all_linear_scope_still_available_as_ablation(self):
        from torchtitan.models.kimi_k3.mxfp4_qat import MXFP4QATLinear

        model = _k3mini_model()
        n = apply_mxfp4_qat(model, scope="all_linear")
        self.assertGreater(n, 0)
        wrapped = [
            fqn for fqn, m in model.named_modules() if isinstance(m, MXFP4QATLinear)
        ]
        self.assertEqual(len(wrapped), n)
        # the ablation scope must leave the experts alone -- it is the
        # complement of the faithful scope, not a superset of it
        for _fqn, experts in quantizable_modules(model):
            self.assertFalse(getattr(experts, "_mxfp4_qat", False))

    @unittest.skipUnless(torch.cuda.is_available(), "grouped_mm needs CUDA")
    def test_qat_changes_expert_output_and_passes_gradient(self):
        torch.manual_seed(0)
        cfg = KimiSiTUGroupedExperts.Config(dim=64, hidden_dim=128, num_experts=2)
        experts = KimiSiTUGroupedExperts(cfg).cuda()
        for p in experts.parameters():
            torch.nn.init.normal_(p, std=0.1)
        x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
        counts = torch.tensor([5, 3], device="cuda", dtype=torch.int32)

        ref = experts(x, counts).clone()
        apply_mxfp4_qat(_wrap_in_holder(experts), quantize_act=True)
        got = experts(x, counts)

        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        # MXFP4 is 4-bit: the output must move, or the fake-quant is a no-op
        self.assertGreater(rel, 1e-3, "MXFP4 fake-quant had no effect")
        self.assertLess(rel, 0.5, f"fake-quant destroyed the output: {rel:.3e}")

        # STE: the bf16 masters must still receive finite gradients
        got.float().sum().backward()
        for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
            g = experts._parameters[name].grad
            self.assertIsNotNone(g, name)
            self.assertTrue(torch.isfinite(g).all(), name)
            self.assertGreater(g.abs().sum().item(), 0.0, name)


def _wrap_in_holder(experts: torch.nn.Module) -> torch.nn.Module:
    """quantizable_modules walks named_modules, so give it a parent whose
    child fqn is not caught by the ignore list."""
    holder = torch.nn.Module()
    holder.routed_experts = torch.nn.Module()
    holder.routed_experts.inner_experts = experts
    return holder


if __name__ == "__main__":
    unittest.main()


class TestGroupedExpertMXFP4Packing(unittest.TestCase):
    """Real MXFP4 packing of routed experts (the QLoRA counterpart of QAT).

    This combination -- MXFP4 on GroupedExperts -- used to be the one the code
    explicitly deferred ("nf4 experts remain the validated path"), and it is
    exactly the scope the released checkpoint uses.
    """

    def _experts(self, dim=64, hidden=128, num_experts=4):
        torch.manual_seed(0)
        cfg = KimiSiTUGroupedExperts.Config(
            dim=dim, hidden_dim=hidden, num_experts=num_experts
        )
        e = KimiSiTUGroupedExperts(cfg)
        for p in e.parameters():
            torch.nn.init.normal_(p, std=0.1)
        return e

    def test_packing_shrinks_experts_and_restores_logical_shape(self):
        from torchtitan.models.kimi_k3.lora import quantize_grouped_experts_mxfp4

        e = self._experts()
        ref = e.w1_EFD.clone()
        holder = _wrap_in_holder(e)
        before = sum(p.numel() * p.element_size() for p in e.parameters())

        self.assertEqual(quantize_grouped_experts_mxfp4(holder), 1)
        self.assertEqual(quantize_grouped_experts_mxfp4(holder), 0)  # idempotent

        after = sum(p.numel() * p.element_size() for p in e.parameters())
        self.assertLess(after, before / 4)  # 4-bit + block scales vs fp32
        got = e.w1_EFD
        self.assertEqual(got.shape, ref.shape)
        # MXFP4 is 4 bits with 2 mantissa bits, so a Gaussian master
        # round-trips at ~10% relative error. That is the whole reason K3
        # does QAT rather than post-training quantization -- it is not a bug,
        # but it does mean packing a bf16 master without QAT loses quality.
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        self.assertGreater(rel, 0.01)
        self.assertLess(rel, 0.25)

    def test_packed_params_are_contiguous_uint8_for_fsdp(self):
        from torchtitan.models.kimi_k3.lora import quantize_grouped_experts_mxfp4

        e = self._experts()
        quantize_grouped_experts_mxfp4(_wrap_in_holder(e))
        names = {n for n, _ in e.named_parameters()}
        self.assertEqual(
            names,
            {
                f"{w}_{part}"
                for w in ("w1_EFD", "w2_EDF", "w3_EFD")
                for part in ("qdata", "scale")
            },
        )
        for n, p in e.named_parameters():
            self.assertEqual(p.dtype, torch.uint8, n)
            self.assertTrue(p.is_contiguous(), n)
            self.assertFalse(p.requires_grad, n)

    def test_non_blockable_last_dim_stays_bf16(self):
        from torchtitan.models.kimi_k3.lora import quantize_grouped_experts_mxfp4

        e = self._experts(dim=48, hidden=96)  # 48 % 32 != 0
        holder = _wrap_in_holder(e)
        # w1/w3 are [E, 96, 48] -> last dim 48, not packable; w2 is
        # [E, 48, 96] -> last dim 96, packable.
        self.assertEqual(quantize_grouped_experts_mxfp4(holder), 1)
        self.assertIn("w1_EFD", e._parameters)
        self.assertNotIn("w2_EDF", e._parameters)

    @unittest.skipUnless(torch.cuda.is_available(), "grouped_mm needs CUDA")
    def test_packed_experts_still_forward(self):
        from torchtitan.models.kimi_k3.lora import quantize_grouped_experts_mxfp4

        e = self._experts().cuda()
        x = torch.randn(6, 64, device="cuda", dtype=torch.bfloat16)
        counts = torch.tensor([4, 1, 1, 0], device="cuda", dtype=torch.int32)
        ref = e(x, counts).clone()
        quantize_grouped_experts_mxfp4(_wrap_in_holder(e))
        got = e(x, counts)
        self.assertTrue(torch.isfinite(got).all())
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        self.assertGreater(rel, 1e-3, "packing had no effect on the forward")
