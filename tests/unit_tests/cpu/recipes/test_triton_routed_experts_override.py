# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-level tests for the Triton routed-experts override (the kernels run in the GPU tests)."""

import unittest
from functools import partial

from torch.nn import init

from torchtitan.config.override import apply_overrides, OverrideConfig
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.moe import RoutedExperts
from torchtitan_recipes.overrides.triton_routed_experts import TritonRoutedExperts

_TRITON_ROUTED_EXPERTS = (
    "torchtitan_recipes.overrides.triton_routed_experts.triton_routed_experts"
)


def _routed_experts_config() -> RoutedExperts.Config:
    return make_routed_experts_config(
        dim=16,
        hidden_dim=32,
        num_experts=4,
        top_k=2,
        param_init={
            name: partial(init.trunc_normal_, std=0.02)
            for name in ("w1_EFD", "w2_EDF", "w3_EFD")
        },
        enable_ep=False,
    )


class TestTritonRoutedExpertsOverride(unittest.TestCase):
    def test_override_claims_routed_experts(self):
        moe = make_moe_config(
            num_experts=4,
            router=make_router_config(
                dim=16,
                num_experts=4,
                gate_param_init={"weight": partial(init.trunc_normal_, std=0.02)},
                score_func=Sigmoid.Config(),
                top_k=2,
            ),
            routed_experts=_routed_experts_config(),
            load_balance_coeff=1e-3,
            enable_sp=False,
        )
        apply_overrides(OverrideConfig(imports=[_TRITON_ROUTED_EXPERTS]), moe)
        self.assertIsInstance(moe.routed_experts, TritonRoutedExperts.Config)


if __name__ == "__main__":
    unittest.main()
