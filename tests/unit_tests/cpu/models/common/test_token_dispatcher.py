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
            _, token_indices_experts_sorted_N, _ = dispatcher._local_reorder(
                torch.randn(4096, 2), torch.rand(4096, 1), ids_TK
            )
            self.assertTrue(
                torch.equal(
                    token_indices_experts_sorted_N,
                    torch.argsort(ids_TK.view(-1), stable=True),
                )
            )


class TestGatherTopKRows(unittest.TestCase):
    def test_backward_matches_indexing_backward(self):
        # The hand-written backward reduces each token's top_k slots through an
        # inverted permutation instead of scattering. The forward stays correct
        # if that inversion is wrong, so only the gradient catches it.
        torch.manual_seed(0)
        num_tokens, dim, num_experts, top_k = 12, 16, 8, 4
        dispatcher = LocalTokenDispatcher.Config(
            num_experts=num_experts, top_k=top_k
        ).build()
        expert_ids_TK = torch.randint(0, num_experts, (num_tokens, top_k))
        scores_TK = torch.rand(num_tokens, top_k)
        x_TD = torch.randn(num_tokens, dim, requires_grad=True)
        x_reference_TD = x_TD.detach().clone().requires_grad_()

        routed_ND, _, _ = dispatcher._local_reorder(x_TD, scores_TK, expert_ids_TK)
        perm_N = torch.argsort(expert_ids_TK.view(-1), stable=True)
        routed_reference_ND = x_reference_TD[perm_N // top_k]
        self.assertTrue(torch.equal(routed_ND, routed_reference_ND))

        grad_ND = torch.randn_like(routed_ND)
        routed_ND.backward(grad_ND)
        routed_reference_ND.backward(grad_ND)
        torch.testing.assert_close(x_TD.grad, x_reference_TD.grad)


if __name__ == "__main__":
    unittest.main()
