# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import MicrobatchWiseLoadBalanceLoss
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router

TOKENS, DIM, NUM_EXPERTS = 256, 512, 64


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDeepSeekV3RouterLocalCompile(unittest.TestCase):
    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def _run(self, router, x, expert_bias_E):
        router.tokens_per_expert_E.zero_()
        router.aux_loss.instance_acc.zero_()
        x = x.detach().requires_grad_()
        topk_scores_TK, topk_ids_TK, routing_map_TE = router(
            x,
            expert_bias_E,
            aux_loss_denominator=torch.tensor(float(TOKENS), device="cuda"),
        )
        # Compare per token in expert-id order: topk's id order is unspecified.
        order_TK = topk_ids_TK.argsort(dim=-1)
        topk_scores_TK = topk_scores_TK.gather(-1, order_TK)
        generator = torch.Generator(device="cuda").manual_seed(1)
        grad_TK = torch.randn(topk_scores_TK.shape, device="cuda", generator=generator)
        topk_scores_TK.backward(grad_TK)
        values = [
            topk_scores_TK.detach(),
            topk_ids_TK.gather(-1, order_TK),
            routing_map_TE,
            router.tokens_per_expert_E.clone(),
            router.aux_loss.instance_acc.clone(),
            x.grad,
            router.gate.weight.grad.clone(),
        ]
        router.zero_grad(set_to_none=True)
        return values

    def test_router_matches_eager(self):
        torch.manual_seed(0)
        config = DeepSeekV3Router.Config(
            num_experts=NUM_EXPERTS,
            gate=HiMidLoLinear.Config(
                backward_mode="hi_mid_lo", in_features=DIM, out_features=NUM_EXPERTS
            ),
            score_func=Sigmoid.Config(),
            num_expert_groups=8,
            num_limited_groups=4,
            top_k=8,
            route_norm=True,
            route_scale=2.5,
            aux_loss=MicrobatchWiseLoadBalanceLoss.Config(coeff=1e-3),
        )
        with torch.device("cuda"):
            router = config.build()
        router.init_states(buffer_device=torch.device("cuda"))
        with torch.no_grad():
            router.gate.weight.normal_(std=0.02)
        # bf16 gate weight as under FSDP mixed precision; buffers stay fp32.
        router.gate.to(torch.bfloat16)
        x = torch.randn(TOKENS, DIM, device="cuda", dtype=torch.bfloat16)
        expert_bias_E = torch.randn(NUM_EXPERTS, device="cuda") * 1e-2

        apply_local_compile([])
        eager = self._run(router, x, expert_bias_E)
        apply_local_compile(["router"])
        compiled = self._run(router, x, expert_bias_E)
        # Default dtype-aware tolerances: ids, maps and counts must match exactly, and the
        # compiled kernels may round bf16 grads one ulp differently.
        for actual, expected in zip(compiled, eager, strict=True):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
