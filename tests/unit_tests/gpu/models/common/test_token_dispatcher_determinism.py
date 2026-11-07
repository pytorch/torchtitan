# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.moe.token_dispatcher import LocalTokenDispatcher


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestCompiledTrainingDeterminism(unittest.TestCase):
    def test_backward_is_bitwise_deterministic(self):
        # The compiled backward sums each token's routes; no atomics anywhere.
        T, K, E, D = 8192, 8, 256, 2048
        gen = torch.Generator(device="cuda").manual_seed(0)
        ids_TK = torch.rand(T, E, device="cuda", generator=gen).topk(K, dim=1).indices
        routing_map_TE = torch.zeros(T, E, dtype=torch.bool, device="cuda").scatter_(
            1, ids_TK, True
        )
        x_TD = torch.randn(T, D, device="cuda", dtype=torch.bfloat16, generator=gen)
        scores_TK = torch.rand(T, K, device="cuda", generator=gen)
        grad_TD = torch.randn(T, D, device="cuda", dtype=torch.bfloat16, generator=gen)
        dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(num_experts=E, top_k=K)
        )

        def grads():
            x = x_TD.clone().requires_grad_()
            scores = scores_TK.clone().requires_grad_()
            routed_ND, _, metadata = dispatcher.dispatch(
                x, scores, ids_TK, routing_map_TE
            )
            self.assertIsNotNone(metadata.row_of_route_N)  # the compiled sort ran
            out_TD = dispatcher.combine(routed_ND, metadata, x)
            return torch.autograd.grad(out_TD, (x, scores), grad_TD)

        LocalCompileConfig(regions=["moe_dispatch_combine"]).apply()
        try:
            first, second = grads(), grads()
        finally:
            LocalCompileConfig().apply()
        for a, b in zip(first, second, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
