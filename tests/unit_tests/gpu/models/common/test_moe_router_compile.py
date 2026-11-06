# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import TokenChoiceTopKRouter


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRouterPostprocessLocalCompile(unittest.TestCase):
    def setUp(self):
        previous = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, previous)
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        LocalCompileConfig().apply()
        torch._dynamo.reset()

    def test_compiled_matches_eager(self):
        torch.manual_seed(42)
        config = TokenChoiceTopKRouter.Config(
            num_experts=64,
            top_k=8,
            gate=HiMidLoLinear.Config(
                in_features=32,
                out_features=64,
                backward_mode="hi_mid_lo",
                bias=False,
            ),
            score_func=Sigmoid.Config(),
            route_norm=True,
            route_scale=2.5,
        )
        with torch.device("cuda"):
            router = config.build()

        generator = torch.Generator(device="cuda").manual_seed(42)
        x_TD = torch.randn(
            256,
            config.gate.in_features,
            device="cuda",
            generator=generator,
            requires_grad=True,
        )
        grad_TK = torch.randn(
            256,
            config.top_k,
            device="cuda",
            generator=generator,
        )

        def run():
            router.tokens_per_expert_E.zero_()
            inputs_TD = x_TD.detach().clone().requires_grad_(True)
            weights_TK, ids_TK, routing_map_TE = router(inputs_TD)
            gradients = torch.autograd.grad(
                weights_TK, (inputs_TD, router.gate.weight), grad_TK
            )
            return (
                weights_TK,
                ids_TK,
                routing_map_TE,
                router.tokens_per_expert_E.clone(),
                *gradients,
            )

        LocalCompileConfig().apply()
        eager = run()
        LocalCompileConfig(regions=["router_postprocess"]).apply()
        compiled = run()

        for actual, expected in zip(compiled, eager, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
