# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.common.nn_modules import Linear
from torchtitan.ops.topk import deterministic_topk


def _has_stable_deterministic_topk() -> bool:
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    x = torch.tensor([[3.0, 3.0, 2.0, 3.0, 1.0], [1.0, 0.0, 0.0, 0.0, 2.0]])
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        _, indices = torch.topk(x, k=3, dim=-1, largest=True, sorted=True)
        return torch.equal(indices, torch.tensor([[0, 1, 3], [4, 0, 1]]))
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn_only)


_HAS_STABLE_DETERMINISTIC_TOPK = _has_stable_deterministic_topk()


class TestDeterministicTopK(unittest.TestCase):
    def tearDown(self) -> None:
        torch.use_deterministic_algorithms(False)

    def test_matches_topk(self):
        x = torch.tensor([[1.0, 4.0, 2.0, 3.0], [5.0, 0.0, 5.0, 1.0]])

        actual_values, actual_indices = deterministic_topk(
            x, k=2, dim=-1, largest=True, sorted=True
        )
        expected_values, expected_indices = torch.topk(
            x, k=2, dim=-1, largest=True, sorted=True
        )

        torch.testing.assert_close(actual_values, expected_values)
        torch.testing.assert_close(actual_indices, expected_indices)

    @unittest.skipUnless(
        _HAS_STABLE_DETERMINISTIC_TOPK,
        "requires PyTorch deterministic topk tie-breaking",
    )
    def test_ties_use_stable_indices(self):
        x = torch.tensor([[3.0, 3.0, 2.0, 3.0, 1.0], [1.0, 0.0, 0.0, 0.0, 2.0]])

        values, indices = deterministic_topk(x, k=3, dim=-1, largest=True, sorted=True)
        torch.testing.assert_close(
            values, torch.tensor([[3.0, 3.0, 3.0], [2.0, 1.0, 0.0]])
        )
        torch.testing.assert_close(indices, torch.tensor([[0, 1, 3], [4, 0, 1]]))

        values, indices = deterministic_topk(x, k=4, dim=-1, largest=False, sorted=True)
        torch.testing.assert_close(
            values, torch.tensor([[1.0, 2.0, 3.0, 3.0], [0.0, 0.0, 0.0, 1.0]])
        )
        torch.testing.assert_close(indices, torch.tensor([[4, 2, 0, 1], [1, 2, 3, 0]]))

    def test_restores_deterministic_state(self):
        torch.use_deterministic_algorithms(False)
        deterministic_topk(torch.randn(4), k=2)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())

        torch.use_deterministic_algorithms(True, warn_only=True)
        deterministic_topk(torch.randn(4), k=2)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_backward_matches_topk(self):
        x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 4.0, 1.0]], requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        values, _ = deterministic_topk(x, k=2, dim=-1, largest=True, sorted=True)
        values_ref, _ = torch.topk(x_ref, k=2, dim=-1, largest=True, sorted=True)

        values.sum().backward()
        values_ref.sum().backward()

        torch.testing.assert_close(x.grad, x_ref.grad)

    def test_fake_tensor(self):
        with FakeTensorMode():
            x = torch.empty(2, 5, dtype=torch.bfloat16)
            values, indices = deterministic_topk(x, k=3, dim=-1)

        self.assertEqual(values.shape, torch.Size([2, 3]))
        self.assertEqual(values.dtype, torch.bfloat16)
        self.assertEqual(indices.shape, torch.Size([2, 3]))
        self.assertEqual(indices.dtype, torch.long)


class TestMoERouterDeterministicTopK(unittest.TestCase):
    @unittest.skipUnless(
        _HAS_STABLE_DETERMINISTIC_TOPK,
        "requires PyTorch deterministic topk tie-breaking",
    )
    def test_router_uses_stable_tie_breaking(self):
        router = TokenChoiceTopKRouter(
            TokenChoiceTopKRouter.Config(
                num_experts=4,
                gate=Linear.Config(in_features=2, out_features=4, bias=True),
                top_k=3,
                score_func="sigmoid",
            )
        )
        torch.nn.init.zeros_(router.gate.weight)
        torch.nn.init.zeros_(router.gate.bias)

        _, expert_ids, _ = router(torch.ones(1, 2, 2))

        torch.testing.assert_close(
            expert_ids.sort(dim=-1).values,
            torch.tensor([[[0, 1, 2], [0, 1, 2]]]),
        )

    @unittest.skipUnless(
        _HAS_STABLE_DETERMINISTIC_TOPK,
        "requires PyTorch deterministic topk tie-breaking",
    )
    def test_node_limited_router_uses_stable_tie_breaking(self):
        router = TokenChoiceTopKRouter(
            TokenChoiceTopKRouter.Config(
                num_experts=4,
                gate=Linear.Config(in_features=2, out_features=4, bias=True),
                num_expert_groups=2,
                num_limited_groups=1,
                top_k=2,
                score_func="sigmoid",
            )
        )
        torch.nn.init.zeros_(router.gate.weight)
        torch.nn.init.zeros_(router.gate.bias)

        _, expert_ids, _ = router(torch.ones(1, 2, 2))

        torch.testing.assert_close(
            expert_ids.sort(dim=-1).values,
            torch.tensor([[[0, 1], [0, 1]]]),
        )


if __name__ == "__main__":
    unittest.main()
