# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.moe.token_dispatcher import LocalTokenDispatcher


class TestArgsortExpertIds(unittest.TestCase):
    def test_narrow_key_argsort_matches_int64_argsort(self):
        # Each side of the uint8 and int16 thresholds, with the largest id present:
        # an off-by-one in a threshold would wrap that id and change the order.
        torch.manual_seed(0)
        for num_experts in (256, 257, 2**15, 2**15 + 1):
            dispatcher = LocalTokenDispatcher.Config(
                num_experts=num_experts, top_k=1
            ).build()
            ids_TK = torch.randint(0, num_experts, (4096, 1))
            ids_TK[0] = num_experts - 1
            routed_ND, _ = dispatcher._local_reorder(
                torch.arange(4096)[:, None], ids_TK
            )
            self.assertTrue(
                torch.equal(
                    routed_ND[:, 0], torch.argsort(ids_TK.view(-1), stable=True)
                )
            )


class TestCombine(unittest.TestCase):
    def test_combine_sums_in_router_order_before_rounding(self):
        dispatcher = LocalTokenDispatcher.Config(num_experts=3, top_k=3).build()
        ids_TK = torch.tensor([[2, 0, 1], [1, 2, 0]])
        scores_TK = torch.tensor([[0.25, 0.25, 0.5], [0.4995, 0.5, 0.0005]])
        x_TD = torch.ones(2, 1, dtype=torch.bfloat16)
        routing_map_TE = torch.ones(2, 3, dtype=torch.bool)
        _, _, metadata = dispatcher.dispatch(x_TD, scores_TK, ids_TK, routing_map_TE)

        # Each expert returns one row per token. In expert order token 0 loses
        # the small term. Rounding the products to bf16 erases token 1's sum.
        rows_ND = torch.tensor([-(2**27), 0, 1, -1, 2**27, 1], dtype=x_TD.dtype)[
            :, None
        ]
        actual = dispatcher.combine(rows_ND, metadata, x_TD)
        expected = torch.tensor([[0.5], [0.0005]], dtype=x_TD.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_combine_gradients_in_eager_and_compiled(self):
        torch.manual_seed(0)
        dispatcher = LocalTokenDispatcher.Config(num_experts=4, top_k=2).build()
        ids_TK = torch.tensor([[2, 0], [0, 1], [2, 1]])
        routing_map_TE = torch.nn.functional.one_hot(ids_TK, 4).any(dim=1)
        scores_TK = torch.rand(3, 2, requires_grad=True)
        x_TD = torch.randn(3, 16, dtype=torch.bfloat16, requires_grad=True)
        grad_TD = torch.randn_like(x_TD)

        # Each expert scales its input by its id + 1.
        rows_TKD = x_TD[:, None, :] * (ids_TK[:, :, None] + 1)
        expected = (rows_TKD * scores_TK[:, :, None]).sum(1).to(x_TD.dtype)
        expected_grads = torch.autograd.grad(expected, (x_TD, scores_TK), grad_TD)

        compiled = torch.compile(
            dispatcher.combine, backend="aot_eager", fullgraph=True
        )
        for combine in (dispatcher.combine, compiled):
            routed_ND, counts_E, metadata = dispatcher.dispatch(
                x_TD, scores_TK, ids_TK, routing_map_TE
            )
            scales_N = torch.arange(1, 5).repeat_interleave(counts_E)
            actual = combine(routed_ND * scales_N[:, None], metadata, x_TD)
            grads = torch.autograd.grad(actual, (x_TD, scores_TK), grad_TD)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(grads, expected_grads, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
