# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capstone: all K3 deltas compose in one training step.

Gated MLA + alpha-graft AttnRes + MXFP4/MXFP8 QAT + Per-Head Muon
together on a debug model. Proves the overnight components interoperate
(forward + backward + optimizer step), not model quality.
"""

import dataclasses
import unittest

import torch


@unittest.skipIf(not torch.cuda.is_available(), "KDA/MX/NS need CUDA")
class TestDeltasCompose(unittest.TestCase):
    def test_all_deltas_one_step(self):
        from torchtitan.models.kimi_k3 import config_registry
        from torchtitan.models.kimi_k3.model import KimiK3Spec
        from torchtitan.models.kimi_k3.muon import Muon
        from torchtitan.models.kimi_k3.mxfp4_qat import apply_mxfp4_qat

        torch.manual_seed(0)
        kc = config_registry.kimi_k3_debugmodel().model_spec.model.kimi_config
        kc = dataclasses.replace(kc, mla_gated=True)  # Gated MLA
        spec = KimiK3Spec(
            kimi_config=kc, num_blocks=4, attn_res_gated=True  # alpha graft
        )
        with torch.device("cuda"):
            model = spec.build()
            model.init_weights()
        n_qat = apply_mxfp4_qat(model, quantize_act=True)  # MXFP4 QAT
        self.assertGreater(n_qat, 0)
        model = model.to(torch.bfloat16)
        for name, p in model.named_parameters():
            if name.endswith("q_proj.base.weight"):
                p._muon_heads = kc.num_attention_heads

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = Muon(trainable, lr=1e-3, adamw_lr=2e-4)  # Per-Head Muon
        tok = torch.randint(0, 2016, (1, 128), device="cuda")
        first = last = None
        for i in range(8):
            out = model(tok)
            loss = out.float().pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if i == 0:
                first = loss.item()
            last = loss.item()
        self.assertTrue(torch.isfinite(torch.tensor(last)))
        self.assertLessEqual(last, first + 1e-3)  # not diverging


if __name__ == "__main__":
    unittest.main()
