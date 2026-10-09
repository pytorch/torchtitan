# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher

NUM_TOKENS, NUM_EXPERTS, TOP_K, DIM = 128, 8, 4, 256


def _dispatch_scale_combine(
    dispatcher, x_TD, scores_TK, expert_ids_TK, row_scale_N, dtype
):
    """Dispatch, scale each routed row (a stand-in for the experts), combine."""
    routed_ND, _, metadata = dispatcher.dispatch(
        x_TD.to(dtype),
        scores_TK,
        expert_ids_TK,
        torch.bincount(expert_ids_TK.flatten()),
    )
    return dispatcher.combine(
        routed_ND * row_scale_N.to(dtype).unsqueeze(-1), metadata, x_TD.to(dtype)
    )


def _reference(x_TD, scores_TK, expert_ids_TK, row_scale_N):
    """Each token sums its K routed rows: score * row_scale(row position) * x, in FP64."""
    sorted_N = torch.argsort(expert_ids_TK.flatten(), stable=True)
    positions_N = torch.empty_like(sorted_N)
    positions_N[sorted_N] = torch.arange(sorted_N.numel(), device=sorted_N.device)
    scale_TK = row_scale_N[positions_N].view_as(scores_TK)
    return x_TD * (scores_TK * scale_TK).sum(dim=-1, keepdim=True)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMoEDispatchCombineCompile(unittest.TestCase):
    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def test_eager_and_compiled_match_fp64_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        dispatcher = LocalTokenDispatcher.Config(
            num_experts=NUM_EXPERTS, top_k=TOP_K
        ).build()
        expert_ids_TK = (
            torch.rand(NUM_TOKENS, NUM_EXPERTS, device="cuda", generator=generator)
            .topk(TOP_K, dim=-1)
            .indices
        )
        x_TD = torch.randn(NUM_TOKENS, DIM, device="cuda", generator=generator)
        scores_TK = torch.rand(NUM_TOKENS, TOP_K, device="cuda", generator=generator)
        row_scale_N = torch.rand(NUM_TOKENS * TOP_K, device="cuda", generator=generator)
        grad_out_TD = torch.randn(NUM_TOKENS, DIM, device="cuda", generator=generator)

        leaves64 = [t.double().requires_grad_() for t in (x_TD, scores_TK)]
        out64 = _reference(
            leaves64[0], leaves64[1], expert_ids_TK, row_scale_N.double()
        )
        expected = [out64, *torch.autograd.grad(out64, leaves64, grad_out_TD.double())]

        for regions in ([], ["moe_dispatch_combine"]):
            apply_local_compile(regions)
            for dtype, tolerance in ((torch.float32, 1e-5), (torch.bfloat16, 1e-2)):
                leaves = [t.clone().requires_grad_() for t in (x_TD, scores_TK)]
                out = _dispatch_scale_combine(
                    dispatcher, leaves[0], leaves[1], expert_ids_TK, row_scale_N, dtype
                )
                actual = [out, *torch.autograd.grad(out, leaves, grad_out_TD.to(dtype))]
                for name, a, e in zip(("out", "x", "scores"), actual, expected):
                    with self.subTest(regions=regions, dtype=dtype, tensor=name):
                        error = ((a.double() - e).norm() / e.norm()).item()
                        self.assertLess(error, tolerance)


if __name__ == "__main__":
    unittest.main()
