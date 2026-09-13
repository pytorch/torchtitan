# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.moe import ExpertActivation
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel_minimal_async_ep,
)
from torchtitan.overrides.fused_swiglu import (
    fused_grouped_experts,
    FusedExpertActivation,
    silu_and_mul_backward_kernel,
    silu_and_mul_forward_kernel,
    silu_and_mul_op,
)


class TestFusedSwiGLUOverride(unittest.TestCase):
    def test_minimal_async_ep_config_imports_override(self):
        config = deepseek_v3_debugmodel_minimal_async_ep(seq_len=2048)

        self.assertIn(
            "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
            config.override.imports,
        )

    def test_expert_activation_config_is_replaced(self):
        cfg = ExpertActivation.Config()

        replacement = fused_grouped_experts(cfg)

        self.assertIsInstance(replacement, FusedExpertActivation.Config)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFusedExpertActivationNumerics(unittest.TestCase):
    def test_matches_reference_forward_and_backward(self):
        torch.manual_seed(0)
        reference = ExpertActivation.Config().build().cuda()
        fused = FusedExpertActivation.Config().build().cuda()
        gate = torch.randn(8, 32, device="cuda")
        up = torch.randn(8, 32, device="cuda")
        offsets = torch.tensor([3, 5, 6, 8], device="cuda", dtype=torch.int32)
        gate_reference = gate.detach().clone().requires_grad_()
        up_reference = up.detach().clone().requires_grad_()
        gate_fused = gate.detach().clone().requires_grad_()
        up_fused = up.detach().clone().requires_grad_()

        out_reference = reference(gate_reference, up_reference, offsets)
        out_fused = fused(gate_fused, up_fused, offsets)
        torch.testing.assert_close(out_fused, out_reference)

        out_reference.sum().backward()
        out_fused.sum().backward()
        torch.testing.assert_close(gate_fused.grad, gate_reference.grad)
        torch.testing.assert_close(up_fused.grad, up_reference.grad)


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
