# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch

import torchtitan.models.common.moe.token_dispatcher as token_dispatcher
from torchtitan.models.common.moe.token_dispatcher import LocalTokenDispatcher


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestCompiledTrainingDeterminism(unittest.TestCase):
    def test_backward_is_bitwise_deterministic(self):
        # The compiled backward sums each token's routes; no atomics anywhere.
        T, K, E, D = 8192, 8, 256, 2048
        gen = torch.Generator(device="cuda").manual_seed(0)
        ids_TK = torch.rand(T, E, device="cuda", generator=gen).topk(K, dim=1).indices
        x_TD = torch.randn(T, D, device="cuda", dtype=torch.bfloat16, generator=gen)
        scores_TK = torch.rand(T, K, device="cuda", generator=gen)
        grad_TD = torch.randn(T, D, device="cuda", dtype=torch.bfloat16, generator=gen)
        dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(num_experts=E, top_k=K)
        )
        names = (
            "_sort_routes_for_training",
            "_gather_weighted_sum",
            "_sum_routes_per_token",
            "_gather_weighted_sum_backward",
        )

        def grads():
            x = x_TD.clone().requires_grad_()
            scores = scores_TK.clone().requires_grad_()
            routed_ND, _, metadata = dispatcher.dispatch(x, scores, ids_TK, None)
            out_TD = dispatcher.combine(routed_ND, metadata, x)
            return torch.autograd.grad(out_TD, (x, scores), grad_TD)

        with patch.multiple(
            token_dispatcher,
            **{n: torch.compile(getattr(token_dispatcher, n)) for n in names},
        ):
            first, second = grads(), grads()
        for a, b in zip(first, second, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
