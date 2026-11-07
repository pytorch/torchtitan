# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import contextlib
import unittest
from unittest.mock import patch

import torch

import torchtitan.models.common.moe.token_dispatcher as token_dispatcher
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.moe.token_dispatcher import LocalTokenDispatcher

# Calls per compiled function, counted by _EagerBackendLocalCompileConfig.
_compiled_calls: collections.Counter[str] = collections.Counter()


class _EagerBackendLocalCompileConfig(LocalCompileConfig):
    """Compiles the enabled regions with the eager backend and counts their calls."""

    @staticmethod
    def _compile(reference, **compile_kwargs):
        compile_kwargs["backend"] = "eager"
        compiled = torch.compile(reference, **compile_kwargs)

        def counted(*args, **kwargs):
            _compiled_calls[reference.__name__] += 1
            return compiled(*args, **kwargs)

        return counted


@contextlib.contextmanager
def _compiled_regions():
    """Turn on the dispatch/combine region by name, as a model's region list does."""
    torch._dynamo.reset()
    _compiled_calls.clear()
    _EagerBackendLocalCompileConfig(regions=["moe_dispatch_combine"]).apply()
    try:
        yield
    finally:
        LocalCompileConfig().apply()
        torch._dynamo.reset()


class TestLocalTokenDispatcherInference(unittest.TestCase):
    def _dispatch_combine(self, x_TD, scores_TK, ids_TK, num_experts):
        dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(num_experts=num_experts, top_k=ids_TK.shape[1])
        )
        routing_map_TE = torch.nn.functional.one_hot(ids_TK, num_experts).any(dim=1)
        routed_ND, _, metadata = dispatcher.dispatch(
            x_TD, scores_TK, ids_TK, routing_map_TE
        )
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
        with torch.no_grad(), _compiled_regions():
            actual_TD = self._dispatch_combine(x_TD, scores_TK, ids_TK, E)
        torch.testing.assert_close(actual_TD, expected_TD, rtol=0, atol=0)

    def test_pairwise_rank_matches_stable_argsort(self):
        # Compiled, small inputs take the pairwise rank and large ones argsort; both
        # give eager's stable order.
        sort_routes = token_dispatcher._sort_routes_and_gather.__wrapped__
        torch.manual_seed(0)
        T, K, D, E = 37, 4, 8, 5
        x_TD = torch.randn(T, D)
        scores_TK = torch.rand(T, K)
        # Duplicate experts within a token: the sort must stay stable.
        ids_TK = torch.randint(0, E, (T, K))
        eager_outputs = sort_routes(x_TD, scores_TK, ids_TK, E)
        pairwise_outputs = torch.compile(sort_routes, backend="eager")(
            x_TD, scores_TK, ids_TK, E
        )
        torch._dynamo.reset()
        with patch.object(token_dispatcher, "_MAX_PAIRWISE_RANK_ROUTES", -1):
            argsort_outputs = torch.compile(sort_routes, backend="eager")(
                x_TD, scores_TK, ids_TK, E
            )
        torch._dynamo.reset()
        for eager_t, pairwise_t, argsort_t in zip(
            eager_outputs[:3], pairwise_outputs[:3], argsort_outputs[:3]
        ):
            self.assertTrue(torch.equal(pairwise_t, eager_t))
            self.assertTrue(torch.equal(argsort_t, eager_t))
        self.assertTrue(torch.equal(pairwise_outputs[3], argsort_outputs[3]))


class TestDispatchCombineRegion(unittest.TestCase):
    def test_one_region_name_compiles_every_function(self):
        # A function registered under another name runs eager and drops out of the calls.
        torch.manual_seed(0)
        T, K, D, E = 37, 4, 8, 5
        ids_TK = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
        dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(num_experts=E, top_k=K)
        )
        routing_map_TE = torch.nn.functional.one_hot(ids_TK, E).any(dim=1)
        x_TD = torch.randn(T, D, requires_grad=True)
        scores_TK = torch.rand(T, K, requires_grad=True)
        with _compiled_regions():
            with torch.no_grad():
                routed_ND, _, metadata = dispatcher.dispatch(
                    x_TD, scores_TK, ids_TK, routing_map_TE
                )
                dispatcher.combine(routed_ND, metadata, x_TD)
            routed_ND, _, metadata = dispatcher.dispatch(
                x_TD, scores_TK, ids_TK, routing_map_TE
            )
            dispatcher.combine(routed_ND, metadata, x_TD).sum().backward()
            self.assertEqual(
                set(_compiled_calls),
                {
                    "_sort_routes_and_gather",
                    "_sort_routes_for_training",
                    "_gather_weighted_sum",
                    "_sum_routes_per_token",
                    "_gather_weighted_sum_backward",
                },
            )


class TestArgsortExpertIds(unittest.TestCase):
    def test_narrow_key_argsort_matches_int64_argsort(self):
        # Each side of the uint8 and int16 thresholds, with the largest id present:
        # an off-by-one in a threshold would wrap that id and change the order.
        torch.manual_seed(0)
        for num_experts in (256, 257, 2**15, 2**15 + 1):
            ids_N = torch.randint(0, num_experts, (4096,))
            ids_N[0] = num_experts - 1
            self.assertTrue(
                torch.equal(
                    token_dispatcher._argsort_expert_ids(ids_N, num_experts),
                    torch.argsort(ids_N, stable=True),
                )
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
        self.routing_map_TE = torch.nn.functional.one_hot(self.ids_TK, self.E).any(
            dim=1
        )

    def _forward(self, x_TD, scores_TK, weight_D):
        routed_ND, _, metadata = self.dispatcher.dispatch(
            x_TD, scores_TK, self.ids_TK, self.routing_map_TE
        )
        return self.dispatcher.combine(routed_ND * weight_D, metadata, x_TD)

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
        # Each route carries x * weight, so out = sum_k score_k * x * weight.
        x64 = x_TD.detach().double().requires_grad_()
        scores64 = scores_TK.detach().double().requires_grad_()
        out64 = scores64.sum(1, keepdim=True) * x64 * weight_D.double()
        ref_grads = torch.autograd.grad(out64, (x64, scores64), grad_TD.double())
        # -1 sends these 148 routes down the argsort branch, as more than 2048 would.
        for max_pairwise_routes in (token_dispatcher._MAX_PAIRWISE_RANK_ROUTES, -1):
            with self.subTest(max_pairwise_routes=max_pairwise_routes), patch.object(
                token_dispatcher, "_MAX_PAIRWISE_RANK_ROUTES", max_pairwise_routes
            ), _compiled_regions():
                out_TD = self._forward(x_TD, scores_TK, weight_D)
                grads = torch.autograd.grad(out_TD, (x_TD, scores_TK), grad_TD)
            for grad, ref in zip(grads, ref_grads, strict=True):
                torch.testing.assert_close(grad.double(), ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
