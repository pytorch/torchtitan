# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The backbone's precision and LoRA are independent concerns.

K3 trains the BACKBONE in MXFP4 weights with MXFP8 activations (report sec
4.1.4). That is a property of the model, not of any adaptation method: LoRA
attaches on top and neither implies nor requires it. These tests pin the
independence, and pin that the two act on disjoint module sets so composing
them cannot double-quantize anything.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.models.kimi_k3.lora import KimiLoRALinear
from torchtitan.models.kimi_k3.model import KimiK3Spec
from torchtitan.models.kimi_k3.model_configs import build_kimi_linear_config
from torchtitan.models.kimi_k3.quant_scope import quantizable_modules


def _spec(**kw) -> KimiK3Spec:
    return KimiK3Spec(
        kimi_config=build_kimi_linear_config("k3mini", vocab_size=256),
        num_blocks=2,
        **kw,
    )


class TestQATAndLoRACompose(unittest.TestCase):
    def _build(self, **kw):
        with torch.device("meta"):
            return _spec(**kw).build()

    def test_backbone_qat_needs_no_lora(self):
        m = self._build(mxfp4_qat=True)
        self.assertEqual(
            [fqn for fqn, x in m.named_modules() if isinstance(x, KimiLoRALinear)],
            [],
        )
        self.assertTrue(
            all(getattr(e, "_mxfp4_qat", False) for _, e in quantizable_modules(m))
        )

    def test_lora_needs_no_qat(self):
        m = self._build(lora_rank=8)
        self.assertTrue(
            any(isinstance(x, KimiLoRALinear) for _, x in m.named_modules())
        )
        self.assertFalse(
            any(getattr(e, "_mxfp4_qat", False) for _, e in quantizable_modules(m))
        )

    def test_composed_and_disjoint(self):
        """QAT lands on GroupedExperts (3-D params); LoRA on nn.Linear. The sets
        must not intersect, or a module would be quantized twice by two
        mechanisms with different semantics."""
        m = self._build(lora_rank=8, mxfp4_qat=True)
        lora = {fqn for fqn, x in m.named_modules() if isinstance(x, KimiLoRALinear)}
        qat = {
            fqn for fqn, e in quantizable_modules(m) if getattr(e, "_mxfp4_qat", False)
        }
        self.assertTrue(lora)
        self.assertTrue(qat)
        self.assertEqual(lora & qat, set())

    def test_activation_quant_is_off_unless_asked(self):
        """The released checkpoint is weights-only (input_activations: null), so
        a frozen-base load without QAT semantics is legitimate and must stay the
        default."""
        base = torch.nn.Linear(64, 32, bias=False)
        w = KimiLoRALinear(base, rank=4, alpha=8.0)
        self.assertFalse(w._quantize_act)

    def test_activation_quant_only_applies_to_a_packed_base(self):
        """Asking for MXFP8 activations on an unpacked bf16 base would change
        the numerics of a configuration nobody described; it is gated on the
        base actually being packed MXFP4."""
        base = torch.nn.Linear(64, 32, bias=False)
        w = KimiLoRALinear(base, rank=4, alpha=8.0, quantize_act=True)
        x = torch.randn(4, 64)
        self.assertIsNone(w._quantize_base)
        self.assertTrue(torch.equal(w._maybe_quantize_act(x), x))

    @unittest.skipUnless(torch.cuda.is_available(), "MX primitives need CUDA")
    def test_activation_quant_changes_the_forward_on_a_packed_base(self):
        torch.manual_seed(0)
        w0 = torch.empty(32, 64, device="cuda")
        torch.nn.init.normal_(w0, std=0.2)

        def packed(quantize_act: bool):
            lin = torch.nn.Linear(64, 32, bias=False).cuda()
            with torch.no_grad():
                lin.weight.copy_(w0)
            mod = KimiLoRALinear(
                lin, rank=4, alpha=8.0, quantize_act=quantize_act
            ).cuda()
            # packing DELETES base.weight, so both arms must be seeded first
            mod.quantize_base_mxfp4()
            with torch.no_grad():
                mod.lora_b.zero_()  # isolate the base path
            return mod

        plain, quant = packed(False), packed(True)

        x = torch.randn(8, 64, device="cuda") * 4.0
        with torch.no_grad():
            a, b = plain(x), quant(x)
        rel = ((a - b).norm() / a.norm()).item()
        self.assertGreater(rel, 1e-4, "MXFP8 activation quant had no effect")
        self.assertLess(rel, 0.5)


if __name__ == "__main__":
    unittest.main()
