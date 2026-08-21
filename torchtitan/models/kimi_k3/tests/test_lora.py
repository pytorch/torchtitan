# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA P0 trio tests at debug scale.

Locks: (1) step-0 identity of the full graft stack (gated AttnRes +
LoRA) vs the plain backbone; (2) gradient routing -- adapters and
AttnRes graft params train, frozen base gets no grads; (3) the
LoRA-only checkpoint payload is exactly the trainable set.
"""

import unittest

import torch

from torchtitan.models.kimi_k3 import config_registry
from torchtitan.models.kimi_k3.lora import trainable_state_dict
from torchtitan.models.kimi_k3.model import KimiK3Spec


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build(spec):
    with torch.device(_device()):
        m = spec.build()
        m.init_weights()
    return m


class TestKimiLoRA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.kimi_config = (
            config_registry.kimi_k3_debugmodel().model_spec.model.kimi_config
        )

    def test_full_graft_stack_step0_identity(self):
        lora = _build(
            KimiK3Spec(
                kimi_config=self.kimi_config,
                num_blocks=4,
                attn_res_gated=True,
                lora_rank=8,
            )
        )
        base = _build(KimiK3Spec(kimi_config=self.kimi_config, num_blocks=None))
        bsd = base.state_dict()
        # base weights live under .base after wrapping; strip for sharing
        shared = {}
        for k, v in lora.state_dict().items():
            k2 = k.replace(".base.weight", ".weight").replace(".base.bias", ".bias")
            if k2 in bsd:
                shared[k2] = v
        self.assertEqual(set(shared), set(bsd))
        base.load_state_dict(shared, strict=True)
        # apply_lora keeps the frozen base bf16-resident; outside the
        # trainer there is no mp_policy to unify compute dtype, so cast
        # both models to bf16 to compare the actual compute graph.
        lora.to(torch.bfloat16)
        base.to(torch.bfloat16)
        g = torch.Generator().manual_seed(0)
        tokens = torch.randint(0, 2016, (2, 128), generator=g).to(_device())
        lora.eval()
        base.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(lora(tokens).float(), base(tokens).float()))

    def test_grad_routing_and_freeze(self):
        model = _build(
            KimiK3Spec(
                kimi_config=self.kimi_config,
                num_blocks=4,
                attn_res_gated=True,
                lora_rank=8,
            )
        )
        # Unify compute dtype (the trainer's mp_policy does this in the
        # real path; unwrapped bf16 frozen base vs fp32 adapters would
        # dtype-clash otherwise).
        model.to(torch.bfloat16)
        g = torch.Generator().manual_seed(0)
        tokens = torch.randint(0, 2016, (2, 128), generator=g).to(_device())
        model(tokens).sum().backward()

        named = dict(model.named_parameters())
        # Adapters train.
        lora_keys = [k for k in named if "lora_a" in k or "lora_b" in k]
        self.assertTrue(lora_keys)
        for k in lora_keys:
            self.assertTrue(named[k].requires_grad, k)
            self.assertIsNotNone(named[k].grad, k)
        # AttnRes graft params train full-param (alpha exception).
        graft_keys = [
            k
            for k in named
            if "attention_res" in k or "ffn_res" in k or "output_res" in k
        ]
        self.assertTrue(graft_keys)
        for k in graft_keys:
            self.assertTrue(named[k].requires_grad, k)
        # Frozen base: no requires_grad, no grads.
        frozen = [k for k in named if k not in lora_keys and k not in graft_keys]
        self.assertTrue(frozen)
        for k in frozen:
            self.assertFalse(named[k].requires_grad, k)
            self.assertIsNone(named[k].grad, k)

    def test_trainable_state_dict_is_lora_plus_graft(self):
        model = _build(
            KimiK3Spec(
                kimi_config=self.kimi_config,
                num_blocks=4,
                attn_res_gated=True,
                lora_rank=8,
            )
        )
        payload = trainable_state_dict(model)
        self.assertTrue(payload)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in payload.values())
        # Frozen-base training: payload is a small fraction of the model.
        self.assertLess(trainable / total, 0.2)
        for k in payload:
            self.assertTrue(
                "lora_a" in k
                or "lora_b" in k
                or "attention_res" in k
                or "ffn_res" in k
                or "output_res" in k,
                k,
            )


if __name__ == "__main__":
    unittest.main()
