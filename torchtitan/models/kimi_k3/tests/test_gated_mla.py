# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated MLA tests.

Two parameterizations, two different promises (see
KimiK3Config.attn_gate_param):

* ``per_head_graft`` -- this repo's graft-viable variant. A checkpoint
  pretrained WITHOUT the gate is ~preserved at step 0 (near-identity, not
  bit-exact: the sigmoid(6)=0.9975 leak distinguishes it from the alpha graft
  gate, which IS bit-exact). That is what this test locks.
* ``full_rank`` -- K3's form, tech report Eq. 7. Channel-wise, no bias, no
  near-identity claim; covered by tests/test_attn_gate.py.
"""

import dataclasses
import unittest

import torch

from torchtitan.models.kimi_k3.model import KimiK3Config, KimiK3Model


def _cfg():
    return KimiK3Config(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=2016,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_experts=8,
        kv_lora_rank=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        kda_head_dim=64,
        kda_num_heads=4,
        # MLA-ONLY. This file tests the MLA output gate (attn_gate_proj); a KDA
        # layer contributes nothing to that and drags in fla's triton kernels,
        # which under triton 3.8 request ~106 KB of dynamic shared memory -- more
        # than consumer Blackwell (RTX 50-series) provides, so the test failed on
        # a hardware limit unrelated to what it checks. KDA's own kernels are
        # covered by test_layers.py and the KCP probes.
        kda_layers=[],
        full_attn_layers=[1, 2],
    )


@unittest.skipIf(not torch.cuda.is_available(), "KDA needs CUDA (fla triton)")
class TestGatedMLA(unittest.TestCase):
    def test_near_identity_at_init_and_grad(self):
        torch.manual_seed(0)
        cfg = _cfg()
        with torch.device("cuda"):
            plain = KimiK3Model.make_config(cfg).build()
            plain.init_weights()
            # near-identity is the per_head_graft promise, not K3's
            gated = KimiK3Model.make_config(
                dataclasses.replace(
                    cfg, mla_gated=True, attn_gate_param="per_head_graft"
                )
            ).build()
            gated.init_weights()
        gated.load_state_dict(plain.state_dict(), strict=False)
        tok = torch.randint(0, 2016, (2, 96), device="cuda")
        plain.eval()
        gated.eval()
        with torch.no_grad():
            lp = plain(tok).float()
            lg = gated(tok).float()
        # near-identity: relative (scale-invariant) is robust to
        # random-init amplification; the sigmoid(6) gate leak keeps it
        # small but NON-zero (not bit-exact, unlike the alpha gate).
        rel = ((lp - lg).norm() / lp.norm()).item()
        self.assertLess(rel, 2e-2)
        self.assertGreater(rel, 0.0)
        # gate trains
        gated.train()
        gated(tok).float().sum().backward()
        gp = dict(gated.named_parameters())
        gk = [k for k in gp if k.endswith("attn_gate_proj.weight")]
        self.assertTrue(gk)
        for k in gk:
            self.assertIsNotNone(gp[k].grad, k)


if __name__ == "__main__":
    unittest.main()
