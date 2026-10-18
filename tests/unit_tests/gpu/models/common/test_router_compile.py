# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.activation import Softmax
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import BatchWiseLoadBalanceLoss, TokenChoiceTopKRouter

TOKENS, DIM, NUM_EXPERTS = 256, 512, 64


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTokenChoiceRouterLocalCompile(unittest.TestCase):
    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def _run(self, router, x):
        router.tokens_per_expert_E.zero_()
        router.aux_loss.instance_acc.zero_()
        router.aux_loss._cumulative_expert_counts_E.zero_()
        x = x.detach().requires_grad_()
        topk_scores_TK, topk_ids_TK, routing_map_TE = router(
            x, aux_loss_denominator=torch.tensor(float(TOKENS), device="cuda")
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
            router.aux_loss._cumulative_expert_counts_E.clone(),
            x.grad,
            router.gate.weight.grad.clone(),
        ]
        router.zero_grad(set_to_none=True)
        return values

    def test_router_matches_eager(self):
        torch.manual_seed(0)
        config = TokenChoiceTopKRouter.Config(
            num_experts=NUM_EXPERTS,
            gate=HiMidLoLinear.Config(
                backward_mode="hi_mid_lo", in_features=DIM, out_features=NUM_EXPERTS
            ),
            score_func=Softmax.Config(),
            top_k=8,
            route_norm=True,
            aux_loss=BatchWiseLoadBalanceLoss.Config(
                coeff=1e-3, num_experts=NUM_EXPERTS
            ),
        )
        with torch.device("cuda"):
            router = config.build()
        router.init_states(buffer_device=torch.device("cuda"))
        with torch.no_grad():
            router.gate.weight.normal_(std=0.02)
        # bf16 gate weight as under FSDP mixed precision; buffers stay fp32.
        router.gate.to(torch.bfloat16)
        x = torch.randn(TOKENS, DIM, device="cuda", dtype=torch.bfloat16)

        apply_local_compile([])
        eager = self._run(router, x)
        apply_local_compile(["moe_router_topk", "moe_aux_loss"])
        compiled = self._run(router, x)
        # Default dtype-aware tolerances: ids, maps and counts must match exactly, and the
        # compiled kernels may round bf16 grads one ulp differently.
        for actual, expected in zip(compiled, eager, strict=True):
            torch.testing.assert_close(actual, expected)

    def test_scores_do_not_depend_on_batch_size(self):
        """A token's router outputs at 1 and 34 tokens equal its outputs in a 16k batch, bitwise."""
        torch.manual_seed(0)
        config = TokenChoiceTopKRouter.Config(
            num_experts=256,
            gate=HiMidLoLinear.Config(in_features=DIM, out_features=256),
            score_func=Softmax.Config(),
            top_k=8,
            route_norm=True,
        )
        with torch.device("cuda"):
            router = config.build().eval()
        apply_local_compile(["moe_router_topk"])
        gate_TE = torch.randn(16384, 256, device="cuda")
        with torch.no_grad():
            # The generator's compile order: 16384 (static), 14336 (T turns dynamic), then
            # decode; Dynamo gives T=1 its own graph.
            outputs = {
                n: router._score_and_select_experts(
                    gate_TE[:n].contiguous(), None, padding_mask_T=None
                )
                for n in (16384, 14336, 34, 1)
            }
        for n in (34, 1):
            for actual, expected in zip(outputs[n], outputs[16384], strict=True):
                self.assertTrue(torch.equal(actual, expected[:n]))


if __name__ == "__main__":
    unittest.main()
