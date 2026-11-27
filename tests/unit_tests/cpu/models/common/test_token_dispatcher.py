# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import unittest
from typing import Any

import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.moe.token_dispatcher import LocalTokenDispatcher


class _EagerBackendLocalCompileConfig(LocalCompileConfig):
    """Traces the enabled regions but keeps the eager backend, so tests stay cheap."""

    @staticmethod
    def _compile(reference: Any, **compile_kwargs: Any) -> Any:
        return torch.compile(reference, **{**compile_kwargs, "backend": "eager"})


@contextlib.contextmanager
def _compiled_combine():
    """Turn on moe_weighted_gather_sum by name, the way Qwen3.5's region list does."""
    torch._dynamo.reset()
    _EagerBackendLocalCompileConfig(regions=["moe_weighted_gather_sum"]).apply()
    try:
        yield
    finally:
        LocalCompileConfig().apply()
        torch._dynamo.reset()


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

    def test_compiled_region_matches_eager(self):
        # Qwen3.5 lists moe_weighted_gather_sum, so its combine runs both helpers
        # under torch.compile(fullgraph=True) -- including the backward one, from
        # inside autograd. Catches a graph break or an untraceable rewrite there,
        # which would otherwise only show up on a GPU running Qwen3.5. The eager
        # backend keeps the ops identical, so the results must match bit for bit.
        torch.manual_seed(0)
        num_tokens, top_k, num_experts, dim = 13, 4, 8, 16
        expert_ids_TK, routing_map_TE = _routing(num_tokens, top_k, num_experts)
        scale_N = _expert_scale_per_row(expert_ids_TK)
        grad_out_TD = torch.randn(num_tokens, dim)
        dispatcher = LocalTokenDispatcher.Config(
            num_experts=num_experts, top_k=top_k
        ).build()

        def forward_backward():
            x_TD = torch.randn(num_tokens, dim, generator=generator).requires_grad_()
            scores_TK = torch.rand(num_tokens, top_k, generator=generator)
            scores_TK.requires_grad_()
            routed_ND, _, metadata = dispatcher.dispatch(
                x_TD, scores_TK, expert_ids_TK, routing_map_TE
            )
            out_TD = dispatcher.combine(routed_ND * scale_N, metadata, x_TD)
            return out_TD, torch.autograd.grad(out_TD, (x_TD, scores_TK), grad_out_TD)

        generator = torch.Generator().manual_seed(1)
        eager_out, eager_grads = forward_backward()
        generator = torch.Generator().manual_seed(1)
        with _compiled_combine():
            compiled_out, compiled_grads = forward_backward()

        torch.testing.assert_close(compiled_out, eager_out, rtol=0, atol=0)
        for compiled, eager in zip(compiled_grads, eager_grads, strict=True):
            torch.testing.assert_close(compiled, eager, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
