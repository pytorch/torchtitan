# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch

from torchtitan.config import apply_overrides, OverrideConfig

from torchtitan.config.override import _REGISTRY
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import MicrobatchWiseLoadBalanceLoss
from torchtitan.models.deepseek_v3 import build_model_config
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router
from torchtitan_recipes.overrides.fused_dsv3_router import (
    fused_dsv3_router,
    FusedDSv3Router,
)


_REGISTERED_OVERRIDES = {
    key: value
    for key, value in _REGISTRY.items()
    if key.startswith("torchtitan_recipes.overrides.fused_dsv3_")
}


class TestFusedDSv3RouterConfig(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(_REGISTRY, _REGISTERED_OVERRIDES))

    def test_override_applies_to_model_config(self):
        model = build_model_config("671B", seq_len=4096)
        apply_overrides(
            OverrideConfig(
                imports=[
                    "torchtitan_recipes.overrides.fused_dsv3_router.fused_dsv3_router"
                ]
            ),
            model,
        )
        router = model.layers[3].moe.router
        self.assertIs(type(router), FusedDSv3Router.Config)
        self.assertIs(type(router.aux_loss), MicrobatchWiseLoadBalanceLoss.Config)
        self.assertEqual(router.gate.in_features, 7168)
        self.assertEqual(router.gate.out_features, 256)
        self.assertEqual(router.top_k, 8)

    def test_cpu_fallback_preserves_model_contract(self):
        torch.manual_seed(42)
        config = DeepSeekV3Router.Config(
            num_experts=8,
            gate=HiMidLoLinear.Config(in_features=16, out_features=8),
            score_func=Sigmoid.Config(),
            top_k=2,
            num_expert_groups=2,
            num_limited_groups=1,
            route_norm=True,
            route_scale=2.5,
            aux_loss=MicrobatchWiseLoadBalanceLoss.Config(coeff=0.125),
        )
        native = config.build()
        fused = fused_dsv3_router(config).build()
        with torch.no_grad():
            native.gate.weight.normal_(0, 0.1)
        fused.load_state_dict(native.state_dict())
        self.assertEqual(native.state_dict().keys(), fused.state_dict().keys())
        inputs = torch.randn(15, 16)
        bias = torch.randn(8)
        mask = torch.arange(15) > 11
        results = []
        with patch(
            "torchtitan_recipes.overrides.fused_dsv3_router.router_forward_op",
            side_effect=AssertionError("unsupported input must use native fallback"),
        ):
            for module in (native, fused):
                x = inputs.clone().requires_grad_()
                outputs = module(
                    x,
                    bias,
                    padding_mask_T=mask,
                    aux_loss_denominator=torch.tensor(12.0),
                )
                grads = torch.autograd.grad(outputs[0].sum(), (x, module.gate.weight))
                results.append(
                    (
                        *outputs,
                        *grads,
                        module.tokens_per_expert_E,
                        module.aux_loss.instance_acc,
                    )
                )
        for expected, actual in zip(*results):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
