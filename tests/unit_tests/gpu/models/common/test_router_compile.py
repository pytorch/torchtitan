# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models import deepseek_v3, kimi_k2_7, muse_spark
from torchtitan.models.common.activation import Softmax
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import BatchWiseLoadBalanceLoss, TokenChoiceTopKRouter

TOKENS, DIM, NUM_EXPERTS = 256, 512, 64


def _flavor_router(model_config) -> TokenChoiceTopKRouter:
    """The last layer's router of a full-size flavor, on CUDA, with a bf16 gate."""
    config = model_config.layers[-1].feed_forward.router
    with torch.device("cuda"):
        router = config.build()
    router.init_states(buffer_device=torch.device("cuda"))
    with torch.no_grad():
        router.gate.weight.normal_(std=0.02)
    router.gate.to(torch.bfloat16)
    return router


# Sigmoid scores and route_norm; the tests pass an expert bias, as the MoE does.
# DeepSeek-V3 671B also runs its group-limited top-k (8 expert groups, 4 kept) inside the region.
FLAVOR_ROUTERS = {
    "DeepSeek-V3 671B": lambda: _flavor_router(
        deepseek_v3.build_model_config("671B", parallelism=ParallelismConfig())
    ),
    "Kimi-K2.5": lambda: _flavor_router(
        kimi_k2_7.build_model_config("Kimi-K2.5", parallelism=ParallelismConfig())
    ),
    "Muse Spark 816B-A42B": lambda: _flavor_router(
        muse_spark.build_model_config("816B-A42B")
    ),
}


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTokenChoiceRouterLocalCompile(unittest.TestCase):
    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def _run(self, router, x, expert_bias_E=None):
        router.tokens_per_expert_E.zero_()
        if router.aux_loss is not None:
            router.aux_loss._reset_accumulators()
        x = x.detach().requires_grad_()
        topk_scores_TK, topk_ids_TK, routing_map_TE = router(
            x,
            expert_bias_E=expert_bias_E,
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
            x.grad,
            router.gate.weight.grad.clone(),
        ]
        if router.aux_loss is not None:
            values += [buffer.clone() for buffer in router.aux_loss.buffers()]
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

    def test_flavor_routers_match_eager(self):
        """Full-size flavor routers: compiled fwd + bwd matches eager."""
        for flavor, build_router in FLAVOR_ROUTERS.items():
            with self.subTest(flavor):
                torch.manual_seed(0)
                router = build_router()
                dim = router.gate.weight.shape[1]
                x = torch.randn(TOKENS, dim, device="cuda", dtype=torch.bfloat16)
                expert_bias_E = 1e-3 * torch.randn(router.num_experts, device="cuda")
                apply_local_compile([])
                eager = self._run(router, x, expert_bias_E=expert_bias_E)
                apply_local_compile(["moe_router_topk"])
                compiled = self._run(router, x, expert_bias_E=expert_bias_E)
                torch._dynamo.reset()
                for actual, expected in zip(compiled, eager, strict=True):
                    torch.testing.assert_close(actual, expected)
                # Values 4 and 5 are the input and gate weight grads. bf16's
                # rtol (1.6e-2) lets a 1% error pass, so bound their rel L2
                # (measured 2e-5 to 7e-5).
                for actual, expected in zip(compiled[4:6], eager[4:6], strict=True):
                    error_l2 = (actual - expected).float().norm()
                    self.assertLess((error_l2 / expected.float().norm()).item(), 1e-3)

    def test_flavor_router_scores_do_not_depend_on_batch_size(self):
        """Full-size flavor routers: a token's routing is the same at 1-256 tokens and 16k."""
        for flavor, build_router in FLAVOR_ROUTERS.items():
            with self.subTest(flavor):
                torch.manual_seed(0)
                router = build_router().eval()
                apply_local_compile(["moe_router_topk"])
                gate_TE = torch.randn(16384, router.num_experts, device="cuda")
                expert_bias_E = 1e-3 * torch.randn(router.num_experts, device="cuda")
                with torch.no_grad():
                    # Largest first, as the generator compiles.
                    outputs = {
                        n: router._score_and_select_experts(
                            gate_TE[:n].contiguous(),
                            expert_bias_E=expert_bias_E,
                            padding_mask_T=None,
                        )
                        for n in (16384, 256, 34, 3, 1)
                    }
                torch._dynamo.reset()
                for n in (256, 34, 3, 1):
                    for actual, expected in zip(
                        outputs[n], outputs[16384], strict=True
                    ):
                        self.assertTrue(torch.equal(actual, expected[:n]))


if __name__ == "__main__":
    unittest.main()
