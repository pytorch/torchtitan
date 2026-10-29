# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models import deepseek_v3, kimi_k2_7, kimi_k3, muse_spark, qwen3_5
from torchtitan.models.common.moe import TokenChoiceTopKRouter

TOKENS = 256


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


# Qwen3.5: softmax scores, route_norm and the batch-wise load-balance loss.
# The others: sigmoid scores and route_norm; the tests pass an expert bias, as the MoE does.
# DeepSeek-V3 671B also runs its group-limited top-k (8 expert groups, 4 kept) inside the region.
# Kimi K3 picks 16 of 896 experts.
FLAVOR_ROUTERS = {
    "Qwen3.5-35B-A3B": lambda: _flavor_router(
        qwen3_5.build_model_config("35B-A3B", parallelism=ParallelismConfig())
    ),
    "DeepSeek-V3 671B": lambda: _flavor_router(
        deepseek_v3.build_model_config("671B", parallelism=ParallelismConfig())
    ),
    "Kimi-K2.5": lambda: _flavor_router(
        kimi_k2_7.build_model_config("Kimi-K2.5", parallelism=ParallelismConfig())
    ),
    "Muse Spark 816B-A42B": lambda: _flavor_router(
        muse_spark.build_model_config("816B-A42B")
    ),
    "Kimi K3": lambda: _flavor_router(
        kimi_k3.build_model_config("Kimi-K3", parallelism=ParallelismConfig())
    ),
}


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTokenChoiceRouterLocalCompile(unittest.TestCase):
    def tearDown(self):
        LocalCompileConfig(regions=[]).apply()
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

    def test_flavor_routers_match_eager(self):
        """Full-size flavor routers: compiled fwd + bwd matches eager."""
        for flavor, build_router in FLAVOR_ROUTERS.items():
            with self.subTest(flavor):
                torch.manual_seed(0)
                router = build_router()
                dim = router.gate.weight.shape[1]
                x = torch.randn(TOKENS, dim, device="cuda", dtype=torch.bfloat16)
                expert_bias_E = 1e-3 * torch.randn(router.num_experts, device="cuda")
                LocalCompileConfig(regions=[]).apply()
                eager = self._run(router, x, expert_bias_E=expert_bias_E)
                LocalCompileConfig(regions=["moe_router_topk", "moe_aux_loss"]).apply()
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
        """Bitwise: a token's router outputs at 1-256 tokens, and at 16k with grad on, equal its no-grad 16k outputs."""
        for flavor, build_router in FLAVOR_ROUTERS.items():
            with self.subTest(flavor):
                torch.manual_seed(0)
                router = build_router().eval()
                LocalCompileConfig(regions=["moe_router_topk"]).apply()
                gate_TE = torch.randn(16384, router.num_experts, device="cuda")
                expert_bias_E = 1e-3 * torch.randn(router.num_experts, device="cuda")
                # The trainer's graph first, as in training: grad on and a static 16k shape, so the
                # compiled forward also saves tensors for backward.
                trainer_outputs = router._score_and_select_experts(
                    gate_TE.detach().requires_grad_(),
                    expert_bias_E=expert_bias_E,
                    padding_mask_T=None,
                )
                with torch.no_grad():
                    # Largest first, as the generator compiles; T turns dynamic at 256.
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
                for actual, expected in zip(
                    trainer_outputs, outputs[16384], strict=True
                ):
                    self.assertTrue(torch.equal(actual.detach(), expected))


if __name__ == "__main__":
    unittest.main()
