# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher


@pytest.fixture(autouse=True)
def reset_local_compile():
    apply_local_compile([])
    torch._dynamo.reset()
    yield
    apply_local_compile([])


def _dispatch_expert_combine(dispatcher, x_TD, scores_TK, expert_ids_TK, weight_ND):
    routed_input_ND, num_tokens_per_expert_E, metadata = dispatcher.dispatch(
        x_TD,
        scores_TK,
        expert_ids_TK,
        torch.bincount(expert_ids_TK.flatten(), minlength=dispatcher.num_experts),
    )
    # A per-row "expert" so every routed row gets a distinct gradient.
    return dispatcher.combine(routed_input_ND * weight_ND, metadata, x_TD)


def test_inverse_row_indices_match_docstring_example() -> None:
    dispatcher = LocalTokenDispatcher.Config(num_experts=2, top_k=2).build()
    x_TD = torch.arange(2, dtype=torch.float32).unsqueeze(-1)
    _, token_indices_N, _, row_indices_TK = dispatcher._local_reorder(
        x_TD, torch.ones(2, 2), torch.tensor([[1, 0], [0, 1]])
    )
    assert token_indices_N.tolist() == [0, 1, 0, 1]
    assert row_indices_TK.tolist() == [[2, 0], [1, 3]]


def test_compiled_dispatch_and_combine_match_eager() -> None:
    torch.manual_seed(0)
    num_tokens, top_k, dim, num_experts = 16, 3, 8, 5
    dispatcher = LocalTokenDispatcher.Config(
        num_experts=num_experts, top_k=top_k
    ).build()
    expert_ids_TK = torch.stack(
        [torch.randperm(num_experts)[:top_k] for _ in range(num_tokens)]
    )
    x_TD = torch.randn(num_tokens, dim)
    scores_TK = torch.rand(num_tokens, top_k)
    weight_ND = torch.randn(num_tokens * top_k, dim)
    grad_TD = torch.randn(num_tokens, dim)

    results = []
    for regions in ([], ["moe_dispatch_combine"]):
        apply_local_compile(regions)
        x = x_TD.clone().requires_grad_(True)
        scores = scores_TK.clone().requires_grad_(True)
        out_TD = _dispatch_expert_combine(
            dispatcher, x, scores, expert_ids_TK, weight_ND
        )
        out_TD.backward(grad_TD)
        results.append((out_TD.detach(), x.grad, scores.grad))

    for eager, compiled in zip(*results, strict=True):
        torch.testing.assert_close(compiled, eager)
