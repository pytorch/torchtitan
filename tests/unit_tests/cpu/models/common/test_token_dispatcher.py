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


class TestLocalTokenDispatcherInference(unittest.TestCase):
    def _dispatch_combine(self, x_TD, scores_TK, ids_TK, num_experts):
        dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(num_experts=num_experts, top_k=ids_TK.shape[1])
        )
        counts_E = torch.bincount(ids_TK.flatten(), minlength=num_experts)
        routed_ND, _, metadata = dispatcher.dispatch(x_TD, scores_TK, ids_TK, counts_E)
        # Stand-in expert: scale each row by its expert id + 1.
        experts_N = ids_TK.flatten().sort(stable=True).values
        return dispatcher.combine(routed_ND * (experts_N[:, None] + 1), metadata, x_TD)

    def test_compiled_combine_sums_each_tokens_routes_in_float32(self):
        # The scatter-add rounds to bf16 after every route (89% of outputs match
        # here); the compiled gather-sum rounds once, so all match.
        torch.manual_seed(0)
        T, K, D, E = 37, 8, 64, 16
        x_TD = torch.randn(T, D, dtype=torch.bfloat16)
        scores_TK = torch.rand(T, K, dtype=torch.bfloat16)
        ids_TK = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
        expert_rows_TKD = (x_TD[:, None, :] * (ids_TK[..., None] + 1)).double()
        expected_TD = (
            (expert_rows_TKD * scores_TK.double()[..., None]).sum(1).bfloat16()
        )
        compiled = torch.compile(token_dispatcher._gather_weighted_sum, backend="eager")
        with torch.no_grad(), patch.object(
            token_dispatcher, "_gather_weighted_sum", compiled
        ):
            actual_TD = self._dispatch_combine(x_TD, scores_TK, ids_TK, E)
        torch.testing.assert_close(actual_TD, expected_TD, rtol=0, atol=0)

    def test_eager_no_grad_combine_matches_the_scatter_path(self):
        torch.manual_seed(0)
        T, K, D, E = 37, 4, 8, 5
        x_TD = torch.randn(T, D, dtype=torch.float64)
        scores_TK = torch.rand(T, K, dtype=torch.float64)
        ids_TK = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
        with torch.enable_grad():
            expected = self._dispatch_combine(x_TD, scores_TK, ids_TK, E)
        with torch.no_grad():
            actual = self._dispatch_combine(x_TD, scores_TK, ids_TK, E)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestArgsortExpertIds(unittest.TestCase):
    def test_narrow_key_argsort_matches_int64_argsort(self):
        torch.manual_seed(0)
        for num_experts in (5, 300, 40000):  # uint8, int16 and int64 keys
            ids_N = torch.randint(0, num_experts, (4096,))
            self.assertTrue(
                torch.equal(
                    token_dispatcher._argsort_expert_ids(ids_N, num_experts),
                    torch.argsort(ids_N, stable=True),
                )
            )


if __name__ == "__main__":
    unittest.main()
