# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch

from torchtitan.config import apply_overrides, OverrideConfig
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import MicrobatchWiseLoadBalanceLoss
from torchtitan.models.deepseek_v3 import build_model_config
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router
from torchtitan_recipes.overrides.fused_dsv3_seqwise_loss import (
    fused_dsv3_seqwise_loss,
    FusedDSv3SeqwiseLoss,
)


class TestFusedDSv3SeqwiseLossConfig(unittest.TestCase):
    def test_override_applies_to_model_config(self):
        model = build_model_config("671B", seq_len=4096)
        apply_overrides(
            OverrideConfig(
                imports=[
                    "torchtitan_recipes.overrides.fused_dsv3_seqwise_loss.fused_dsv3_seqwise_loss"
                ]
            ),
            model,
        )
        router = model.layers[3].moe.router
        self.assertIs(type(router), DeepSeekV3Router.Config)
        self.assertIs(type(router.aux_loss), FusedDSv3SeqwiseLoss.Config)

    def test_override_preserves_router_and_loss_settings(self):
        cfg = DeepSeekV3Router.Config(
            num_experts=8,
            gate=HiMidLoLinear.Config(in_features=16, out_features=8),
            score_func=Sigmoid.Config(),
            top_k=2,
            aux_loss=MicrobatchWiseLoadBalanceLoss.Config(coeff=0.125),
        )
        replacement = fused_dsv3_seqwise_loss(cfg)
        self.assertIs(type(replacement), DeepSeekV3Router.Config)
        self.assertIs(type(replacement.aux_loss), FusedDSv3SeqwiseLoss.Config)
        self.assertEqual(replacement.aux_loss.coeff, 0.125)
        self.assertEqual(replacement.gate, cfg.gate)

    def test_no_aux_loss_is_unchanged(self):
        cfg = DeepSeekV3Router.Config(
            num_experts=8,
            gate=HiMidLoLinear.Config(in_features=16, out_features=8),
            score_func=Sigmoid.Config(),
        )
        self.assertIs(fused_dsv3_seqwise_loss(cfg), cfg)

    def test_cpu_fallback_preserves_forward_backward_metric_and_state_dict(self):
        torch.manual_seed(0)
        native = MicrobatchWiseLoadBalanceLoss.Config(coeff=0.125).build()
        fused = FusedDSv3SeqwiseLoss.Config(coeff=0.125).build()
        self.assertEqual(native.metric_name, fused.metric_name)
        self.assertEqual(native.state_dict().keys(), fused.state_dict().keys())
        scores = torch.rand(15, 8)
        routing_map = torch.rand(15, 8) > 0.5
        padding_mask = torch.arange(15) > 11
        carrier = torch.rand(15, 2)
        results = []
        with patch(
            "torchtitan_recipes.overrides.fused_dsv3_seqwise_loss.seqwise_loss_forward_op",
            side_effect=AssertionError("unsupported input must use native fallback"),
        ):
            for module in (native, fused):
                x = scores.clone().requires_grad_()
                c = carrier.clone().requires_grad_()
                out = module(
                    x,
                    routing_map,
                    carrier=c,
                    padding_mask_T=padding_mask,
                    denominator=torch.tensor(13.0),
                )
                grads = torch.autograd.grad(out, (x, c), torch.ones_like(out))
                results.append((out, *grads, module.instance_acc))
        for expected, actual in zip(*results):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
