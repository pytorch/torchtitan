# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.moe.token_dispatcher import LocalTokenDispatcher


def _routing(num_tokens: int, top_k: int, num_experts: int):
    """Distinct experts per token, so a token's rows all carry different scales."""
    expert_ids_TK = torch.stack(
        [torch.randperm(num_experts)[:top_k] for _ in range(num_tokens)]
    )
    routing_map_TE = torch.nn.functional.one_hot(expert_ids_TK, num_experts).any(dim=1)
    return expert_ids_TK, routing_map_TE


def _expert_scale_per_row(expert_ids_TK: torch.Tensor) -> torch.Tensor:
    """``expert_id + 1`` for each row of the expert-sorted output."""
    return expert_ids_TK.reshape(-1).sort(stable=True).values[:, None] + 1


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
            _, token_indices_experts_sorted_N, _, _ = dispatcher._local_reorder(
                torch.randn(4096, 2), torch.rand(4096, 1), ids_TK
            )
            self.assertTrue(
                torch.equal(
                    token_indices_experts_sorted_N,
                    torch.argsort(ids_TK.view(-1), stable=True),
                )
            )


class TestCombine(unittest.TestCase):
    """``combine`` sums each token's ``top_k`` expert rows in float32, in token order."""

    def test_combine_sums_each_tokens_routes_in_float32(self):
        # Catches a regression to an accumulation that rounds to bfloat16 after
        # every route, or that adds the routes in expert-sorted order: both
        # silently change the loss of every MoE model.
        torch.manual_seed(0)
        num_tokens, top_k, num_experts, dim = 37, 8, 16, 64
        x_TD = torch.randn(num_tokens, dim, dtype=torch.bfloat16)
        topk_scores_TK = torch.rand(num_tokens, top_k)
        expert_ids_TK, routing_map_TE = _routing(num_tokens, top_k, num_experts)
        dispatcher = LocalTokenDispatcher.Config(
            num_experts=num_experts, top_k=top_k
        ).build()

        routed_ND, _, metadata = dispatcher.dispatch(
            x_TD, topk_scores_TK, expert_ids_TK, routing_map_TE
        )
        # Stand-in expert: scale each row by its expert id + 1, so a token's rows
        # differ from each other and a misrouted row shows up.
        actual_TD = dispatcher.combine(
            routed_ND * _expert_scale_per_row(expert_ids_TK), metadata, x_TD
        )

        # Same arithmetic, written out in token order: bfloat16 rows times float32
        # scores, summed in float32, rounded once.
        rows_TKD = x_TD[:, None, :] * (expert_ids_TK[..., None] + 1)
        expected_TD = (rows_TKD * topk_scores_TK[..., None]).sum(dim=1).to(x_TD.dtype)
        torch.testing.assert_close(actual_TD, expected_TD, rtol=0, atol=0)

    def test_combine_gradients_match_float64_reference(self):
        # combine's backward is hand-written, so check both gradients end to end.
        torch.manual_seed(0)
        num_tokens, top_k, num_experts, dim = 13, 4, 8, 16
        x_TD = torch.randn(num_tokens, dim, requires_grad=True)
        topk_scores_TK = torch.rand(num_tokens, top_k, requires_grad=True)
        expert_ids_TK, routing_map_TE = _routing(num_tokens, top_k, num_experts)
        grad_out_TD = torch.randn(num_tokens, dim)
        dispatcher = LocalTokenDispatcher.Config(
            num_experts=num_experts, top_k=top_k
        ).build()

        routed_ND, _, metadata = dispatcher.dispatch(
            x_TD, topk_scores_TK, expert_ids_TK, routing_map_TE
        )
        out_TD = dispatcher.combine(
            routed_ND * _expert_scale_per_row(expert_ids_TK), metadata, x_TD
        )
        grads = torch.autograd.grad(out_TD, (x_TD, topk_scores_TK), grad_out_TD)

        # out[t] = sum_k scores[t, k] * x[t] * (expert_ids[t, k] + 1)
        x64 = x_TD.detach().double().requires_grad_()
        scores64 = topk_scores_TK.detach().double().requires_grad_()
        rows64 = x64[:, None, :] * (expert_ids_TK[..., None] + 1).double()
        out64 = (rows64 * scores64[..., None]).sum(dim=1)
        reference = torch.autograd.grad(out64, (x64, scores64), grad_out_TD.double())

        for grad, expected in zip(grads, reference, strict=True):
            torch.testing.assert_close(grad.double(), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
