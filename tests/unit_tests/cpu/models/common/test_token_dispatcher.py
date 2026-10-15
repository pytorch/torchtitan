# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch

import torchtitan.models.common.token_dispatcher as token_dispatcher
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher


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
        with torch.no_grad(), patch.multiple(
            token_dispatcher,
            _sort_routes_and_gather=torch.compile(
                token_dispatcher._sort_routes_and_gather, backend="eager"
            ),
            _gather_weighted_sum=torch.compile(
                token_dispatcher._gather_weighted_sum, backend="eager"
            ),
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

    def test_counting_sort_matches_stable_argsort(self):
        # Compiled, small inputs take the counting sort and large ones argsort; both
        # give eager's stable order.
        sort_routes = token_dispatcher._sort_routes_and_gather
        torch.manual_seed(0)
        T, K, D, E = 37, 4, 8, 5
        x_TD = torch.randn(T, D)
        scores_TK = torch.rand(T, K)
        # Duplicate experts within a token, as router replay allows.
        ids_TK = torch.randint(0, E, (T, K))
        eager_outputs = sort_routes(x_TD, scores_TK, ids_TK, E)
        counting_outputs = torch.compile(sort_routes, backend="eager")(
            x_TD, scores_TK, ids_TK, E
        )
        torch._dynamo.reset()
        with patch.object(token_dispatcher, "_MAX_COUNTING_SORT_ONE_HOT_ELEMENTS", -1):
            argsort_outputs = torch.compile(sort_routes, backend="eager")(
                x_TD, scores_TK, ids_TK, E
            )
        torch._dynamo.reset()
        for eager_t, counting_t, argsort_t in zip(
            eager_outputs[:3], counting_outputs[:3], argsort_outputs[:3]
        ):
            self.assertTrue(torch.equal(counting_t, eager_t))
            self.assertTrue(torch.equal(argsort_t, eager_t))
        self.assertTrue(torch.equal(counting_outputs[3], argsort_outputs[3]))


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

def _compiled_regions():
    """Run the dispatcher's regions under torch.compile (eager backend)."""
    names = (
        "_sort_routes_and_gather",
        "_sort_routes_for_training",
        "_gather_weighted_sum",
        "_sum_routes_per_token",
        "_gather_weighted_sum_backward",
    )
    return patch.multiple(
        token_dispatcher,
        **{
            name: torch.compile(getattr(token_dispatcher, name), backend="eager")
            for name in names
        },
    )


class TestLocalTokenDispatcherTraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.T, self.K, self.E, self.D = 37, 4, 8, 32
        self.dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(num_experts=self.E, top_k=self.K)
        )
        self.ids_TK = torch.stack(
            [torch.randperm(self.E)[: self.K] for _ in range(self.T)]
        )

    def _forward(self, x_TD, scores_TK, weight_D):
        routed_ND, _, metadata = self.dispatcher.dispatch(
            x_TD, scores_TK, self.ids_TK, None
        )
        return self.dispatcher.combine(routed_ND * weight_D, metadata, x_TD)

    def _scatter_path(self, x_TD, scores_TK, weight_D):
        routed_ND, tokens_N, scores_N = self.dispatcher._local_reorder(
            x_TD, scores_TK, self.ids_TK
        )
        return LocalTokenDispatcher._score_and_scatter_add(
            routed_ND * weight_D, scores_N, tokens_N, x_TD
        )

    def _inputs(self, dtype):
        x_TD = torch.randn(self.T, self.D, dtype=dtype, requires_grad=True)
        scores_TK = torch.rand(self.T, self.K, requires_grad=True)
        return x_TD, scores_TK, torch.randn(self.D, dtype=dtype)

    def test_compiled_training_combine_matches_inference_bitwise(self):
        # Trainer and generator sum a token's routes the same way (float32, one rounding).
        x_TD, scores_TK, weight_D = self._inputs(torch.bfloat16)
        with _compiled_regions():
            training_TD = self._forward(x_TD, scores_TK, weight_D)
            with torch.no_grad():
                inference_TD = self._forward(x_TD, scores_TK, weight_D)
        torch.testing.assert_close(training_TD, inference_TD, rtol=0, atol=0)

    def test_compiled_training_grads_match_float64_reference(self):
        x_TD, scores_TK, weight_D = self._inputs(torch.float32)
        grad_TD = torch.randn(self.T, self.D)
        with _compiled_regions():
            out_TD = self._forward(x_TD, scores_TK, weight_D)
            grads = torch.autograd.grad(out_TD, (x_TD, scores_TK), grad_TD)
        # Each route carries x * weight, so out = sum_k score_k * x * weight.
        x64 = x_TD.detach().double().requires_grad_()
        scores64 = scores_TK.detach().double().requires_grad_()
        out64 = scores64.sum(1, keepdim=True) * x64 * weight_D.double()
        ref_grads = torch.autograd.grad(out64, (x64, scores64), grad_TD.double())
        for grad, ref in zip(grads, ref_grads, strict=True):
            torch.testing.assert_close(grad.double(), ref, rtol=1e-5, atol=1e-5)

    def test_compiled_training_without_score_grads(self):
        # Detached scores skip the score-gradient work; grad_x is unchanged.
        x_TD, scores_TK, weight_D = self._inputs(torch.float32)
        grad_TD = torch.randn(self.T, self.D)
        with _compiled_regions():
            (expected_grad_x,) = torch.autograd.grad(
                self._forward(x_TD, scores_TK, weight_D), (x_TD,), grad_TD
            )
            (grad_x,) = torch.autograd.grad(
                self._forward(x_TD, scores_TK.detach(), weight_D), (x_TD,), grad_TD
            )
        torch.testing.assert_close(grad_x, expected_grad_x, rtol=0, atol=0)

    def test_eager_training_matches_the_scatter_path(self):
        # Uncompiled, dispatch and combine keep the scatter-add forward and backward.
        x_TD, scores_TK, weight_D = self._inputs(torch.bfloat16)
        grad_TD = torch.randn(self.T, self.D, dtype=torch.bfloat16)
        actual = self._forward(x_TD, scores_TK, weight_D)
        expected = self._scatter_path(x_TD, scores_TK, weight_D)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for a, e in zip(
            torch.autograd.grad(actual, (x_TD, scores_TK), grad_TD),
            torch.autograd.grad(expected, (x_TD, scores_TK), grad_TD),
            strict=True,
        ):
            torch.testing.assert_close(a, e, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
