# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP4/MXFP8 fake-quant QAT wrapper tests (K3-faithful quant path).

CUDA-only (torchao MX primitives). Scoped to the WRAPPER on a single
controlled Linear (deterministic, in-range) -- deep random-init MXFP4
models are numerically unstable by nature (real QAT weights train
in-range), which is a property of emulated 4-bit, not the wrapper.
"""

import unittest

import torch


@unittest.skipIf(not torch.cuda.is_available(), "torchao MX needs CUDA")
class TestMXFP4QAT(unittest.TestCase):
    def _linear(self):
        torch.manual_seed(0)
        # in-range weights (real QAT weights are trained in-range); dims
        # divisible by the MX block (32).
        lin = torch.nn.Linear(128, 64, bias=False).cuda().to(torch.bfloat16)
        with torch.no_grad():
            lin.weight.mul_(0.1)
        return lin

    def test_wrap_forward_and_ste_grad(self):
        from torchtitan.models.kimi_k3.mxfp4_qat import MXFP4QATLinear

        lin = self._linear()
        wrapped = MXFP4QATLinear(lin, quantize_act=True)
        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16) * 0.1
        x.requires_grad_(True)
        out = wrapped(x)
        self.assertTrue(torch.isfinite(out).all())
        out.float().sum().backward()
        # STE: the frozen-master weight receives a finite grad.
        self.assertIsNotNone(wrapped.base.weight.grad)
        self.assertTrue(torch.isfinite(wrapped.base.weight.grad).all())

    def test_quantization_actually_perturbs(self):
        from torchtitan.models.kimi_k3.mxfp4_qat import MXFP4QATLinear

        lin = self._linear()
        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16) * 0.1
        with torch.no_grad():
            ref = torch.nn.functional.linear(x, lin.weight).float()
            q = MXFP4QATLinear(lin, quantize_act=True)(x).float()
        # MXFP4 weights + MXFP8 acts must measurably change the output;
        # a silent no-op (wrong elem dtype) would make these equal.
        self.assertGreater((ref - q).abs().max().item(), 1e-3)

    def test_apply_wraps_model_targets(self):
        from torchtitan.models.kimi_k3 import config_registry
        from torchtitan.models.kimi_k3.model import KimiK3Spec
        from torchtitan.models.kimi_k3.mxfp4_qat import apply_mxfp4_qat

        kc = config_registry.kimi_k3_debugmodel().model_spec.model.kimi_config
        spec = KimiK3Spec(kimi_config=kc, num_blocks=None)
        with torch.device("cuda"):
            m = spec.build()
            m.init_weights()
        n = apply_mxfp4_qat(m, quantize_act=True)
        self.assertGreater(n, 0)  # MLA + FFN targets wrapped


class TestWrapperLooksLikeLinear(unittest.TestCase):
    def test_weight_and_bias_are_the_base_parameters(self):
        from torch import nn

        from torchtitan.models.kimi_k3.mxfp4_qat import MXFP4QATLinear

        lin = nn.Linear(8, 16, bias=True)
        wrapped = MXFP4QATLinear(lin, quantize_act=True)
        # Identity, not equality: tagging an attribute on the returned tensor
        # has to land on the parameter the optimizer will actually see.
        self.assertIs(wrapped.weight, lin.weight)
        self.assertIs(wrapped.bias, lin.bias)

    def test_weight_is_the_master_not_the_fake_quantized_value(self):
        # The passthrough exposes the trainable bf16 master. Forward quantizes a
        # local copy, so .weight deliberately does not reflect what forward uses.
        from torch import nn

        from torchtitan.models.kimi_k3.mxfp4_qat import MXFP4QATLinear

        lin = nn.Linear(64, 64, bias=False)
        wrapped = MXFP4QATLinear(lin, quantize_act=False)
        self.assertIs(wrapped.weight, lin.weight)
        x = torch.randn(4, 64)
        self.assertFalse(
            torch.equal(wrapped(x), torch.nn.functional.linear(x, lin.weight))
        )

    def test_per_head_muon_still_tags_wrapped_projections(self):
        from torchtitan.models.kimi_k3.attn_res_model import KimiK3AttnResModel
        from torchtitan.models.kimi_k3.muon import tag_per_head_muon
        from torchtitan.models.kimi_k3.mxfp4_qat import apply_mxfp4_qat
        from torchtitan.models.kimi_k3.tests.test_kimi_attn_res_model import (
            _dense_mla_only_config,
        )

        def build():
            with torch.device("meta"):
                return KimiK3AttnResModel(
                    _dense_mla_only_config(num_hidden_layers=4), num_blocks=2
                )

        baseline = tag_per_head_muon(build())
        self.assertGreater(baseline, 0)
        quantized = build()
        # all_linear is the scope that reaches the MLA projections; the default
        # k3_official scope wraps routed experts, which are not per-head targets.
        wrapped = apply_mxfp4_qat(quantized, scope="all_linear")
        self.assertGreater(wrapped, 0)
        self.assertEqual(tag_per_head_muon(quantized), baseline)


if __name__ == "__main__":
    unittest.main()
