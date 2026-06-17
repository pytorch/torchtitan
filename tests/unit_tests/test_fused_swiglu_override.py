# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import (
    LocalTokenDispatcher,
    MinimalAsyncEPTokenDispatcher,
)
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel_minimal_async_ep,
)
from torchtitan.overrides.fused_grouped_experts import (
    fused_grouped_experts,
    FusedGroupedExperts,
    silu_and_mul_backward_kernel,
    silu_and_mul_forward_kernel,
    silu_and_mul_op,
)


class TestFusedSwiGLUOverride(unittest.TestCase):
    def test_minimal_async_ep_config_imports_override(self):
        config = deepseek_v3_debugmodel_minimal_async_ep()

        self.assertIn(
            "torchtitan.overrides.fused_grouped_experts",
            config.override.imports,
        )

    def test_minimal_async_ep_grouped_experts_config_is_replaced(self):
        cfg = GroupedExperts.Config(
            dim=16,
            hidden_dim=32,
            num_experts=4,
            token_dispatcher=MinimalAsyncEPTokenDispatcher.Config(
                num_experts=4,
                top_k=1,
            ),
        )

        replacement = fused_grouped_experts(cfg)

        self.assertIsInstance(replacement, FusedGroupedExperts.Config)
        self.assertIs(replacement.token_dispatcher, cfg.token_dispatcher)

    def test_local_grouped_experts_config_is_replaced(self):
        cfg = GroupedExperts.Config(
            dim=16,
            hidden_dim=32,
            num_experts=4,
            token_dispatcher=LocalTokenDispatcher.Config(
                num_experts=4,
                top_k=1,
            ),
        )

        replacement = fused_grouped_experts(cfg)

        self.assertIsInstance(replacement, FusedGroupedExperts.Config)
        self.assertIs(replacement.token_dispatcher, cfg.token_dispatcher)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFusedSwiGLUOverrideKernels(unittest.TestCase):
    def test_silu_and_mul_custom_op_matches_reference_with_offsets(self):
        gate = torch.randn(3, 2, device="cuda", requires_grad=True)
        up = torch.randn(3, 2, device="cuda", requires_grad=True)
        offsets = torch.tensor([1, 2], device="cuda", dtype=torch.int32)

        out = silu_and_mul_op(gate, up, offsets)
        out[:2].sum().backward()

        ref_gate = gate.detach().clone().requires_grad_()
        ref_up = up.detach().clone().requires_grad_()
        expected = torch.nn.functional.silu(ref_gate) * ref_up
        expected[:2].sum().backward()

        assert gate.grad is not None
        assert up.grad is not None
        assert ref_gate.grad is not None
        assert ref_up.grad is not None
        torch.testing.assert_close(out[:2], expected[:2])
        torch.testing.assert_close(gate.grad[:2], ref_gate.grad[:2])
        torch.testing.assert_close(up.grad[:2], ref_up.grad[:2])

    def test_silu_and_mul_custom_op_matches_reference_without_offsets(self):
        gate = torch.randn(3, 2, device="cuda", requires_grad=True)
        up = torch.randn(3, 2, device="cuda", requires_grad=True)

        out = silu_and_mul_op(gate, up)
        out.sum().backward()

        ref_gate = gate.detach().clone().requires_grad_()
        ref_up = up.detach().clone().requires_grad_()
        expected = torch.nn.functional.silu(ref_gate) * ref_up
        expected.sum().backward()

        assert gate.grad is not None
        assert up.grad is not None
        assert ref_gate.grad is not None
        assert ref_up.grad is not None
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(gate.grad, ref_gate.grad)
        torch.testing.assert_close(up.grad, ref_up.grad)

    def test_silu_and_mul_kernels_match_reference_with_offsets(self):
        gate = torch.tensor(
            [
                [0.0, 1.0],
                [2.0, -3.0],
                [4.0, 5.0],
            ],
            device="cuda",
            requires_grad=True,
        )
        up = torch.tensor(
            [
                [2.0, 3.0],
                [5.0, 7.0],
                [11.0, 13.0],
            ],
            device="cuda",
            requires_grad=True,
        )
        offsets = torch.tensor([1, 2], device="cuda", dtype=torch.int32)

        out = silu_and_mul_forward_kernel(gate, up, offsets)
        expected = torch.nn.functional.silu(gate) * up
        torch.testing.assert_close(out[:2], expected[:2])

        grad_out = torch.tensor(
            [
                [17.0, 19.0],
                [23.0, 29.0],
                [31.0, 37.0],
            ],
            device="cuda",
        )
        grad_gate, grad_up = silu_and_mul_backward_kernel(
            grad_out,
            gate,
            up,
            offsets,
        )
        expected[:2].backward(grad_out[:2])
        assert gate.grad is not None
        assert up.grad is not None
        torch.testing.assert_close(grad_gate[:2], gate.grad[:2])
        torch.testing.assert_close(grad_up[:2], up.grad[:2])
