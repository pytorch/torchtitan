# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.activation import SwiGLU
from torchtitan_recipes.overrides.fused_swiglu import (
    fused_swiglu,
    FusedSwiGLU,
    silu_and_mul_backward_kernel,
    silu_and_mul_forward_kernel,
    silu_and_mul_op,
)


class TestFusedSwiGLUOverride(unittest.TestCase):
    def test_swiglu_config_is_replaced(self):
        cfg = SwiGLU.Config()

        replacement = fused_swiglu(cfg)

        self.assertIsInstance(replacement, FusedSwiGLU.Config)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFusedSwiGLUNumerics(unittest.TestCase):
    def test_matches_reference_forward_and_backward(self):
        torch.manual_seed(0)
        reference = SwiGLU.Config().build()
        fused = FusedSwiGLU.Config().build()
        gate = torch.randn(8, 32, device="cuda")
        up = torch.randn(8, 32, device="cuda")
        gate_up = torch.stack((gate, up), dim=1)
        offsets = torch.tensor([3, 5, 6, 8], device="cuda", dtype=torch.int32)
        gate_up_reference = gate_up.detach().clone().requires_grad_()
        gate_up_fused = gate_up.detach().clone().requires_grad_()

        out_reference = reference(gate_up_reference, offsets=offsets)
        out_fused = fused(gate_up_fused, offsets=offsets)
        torch.testing.assert_close(out_fused, out_reference)

        out_reference.sum().backward()
        out_fused.sum().backward()
        torch.testing.assert_close(gate_up_fused.grad, gate_up_reference.grad)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFusedSwiGLUOverrideKernels(unittest.TestCase):
    def test_silu_and_mul_custom_op_matches_reference_with_offsets(self):
        gate = torch.randn(3, 2, device="cuda")
        up = torch.randn(3, 2, device="cuda")
        gate_up = torch.stack((gate, up), dim=1).requires_grad_()
        offsets = torch.tensor([1, 2], device="cuda", dtype=torch.int32)

        out = silu_and_mul_op(gate_up, offsets)
        out[:2].sum().backward()

        ref_gate_up = gate_up.detach().clone().requires_grad_()
        expected = torch.nn.functional.silu(ref_gate_up[:, 0]) * ref_gate_up[:, 1]
        expected[:2].sum().backward()

        assert gate_up.grad is not None
        assert ref_gate_up.grad is not None
        torch.testing.assert_close(out[:2], expected[:2])
        torch.testing.assert_close(gate_up.grad[:2], ref_gate_up.grad[:2])

    def test_silu_and_mul_custom_op_matches_reference_without_offsets(self):
        gate = torch.randn(3, 2, device="cuda")
        up = torch.randn(3, 2, device="cuda")
        gate_up = torch.stack((gate, up), dim=1).requires_grad_()

        out = silu_and_mul_op(gate_up)
        out.sum().backward()

        ref_gate_up = gate_up.detach().clone().requires_grad_()
        expected = torch.nn.functional.silu(ref_gate_up[:, 0]) * ref_gate_up[:, 1]
        expected.sum().backward()

        assert gate_up.grad is not None
        assert ref_gate_up.grad is not None
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(gate_up.grad, ref_gate_up.grad)

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
        gate_up = torch.stack((gate, up), dim=1).detach()

        out = silu_and_mul_forward_kernel(gate_up, offsets)
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
        grad_gate_up = silu_and_mul_backward_kernel(grad_out, gate_up, offsets)
        expected[:2].backward(grad_out[:2])
        assert gate.grad is not None
        assert up.grad is not None
        torch.testing.assert_close(grad_gate_up[:2, 0], gate.grad[:2])
        torch.testing.assert_close(grad_gate_up[:2, 1], up.grad[:2])

    def test_silu_and_mul_uses_int64_row_stride_arithmetic(self):
        row = 262_144
        row_stride = 8192
        storage_numel = row * row_stride + 2
        required_bytes = storage_numel * 2 + 512 * 1024**2
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < required_bytes:
            self.skipTest(
                f"need at least {required_bytes} free CUDA bytes, got {free_bytes}"
            )

        storage = torch.empty(storage_numel, device="cuda", dtype=torch.bfloat16)
        # gate at storage[r * row_stride], up at storage[r * row_stride + 1].
        gate_up = torch.as_strided(storage, (row + 1, 2, 1), (row_stride, 1, 2))
        gate_up[row, 0] = 1.0
        gate_up[row, 1] = 2.0
        offsets = torch.tensor([row + 1], device="cuda", dtype=torch.int32)

        out = silu_and_mul_forward_kernel(gate_up, offsets)
        grad_out = torch.zeros_like(out)
        grad_out[row] = 3.0
        grad_gate_up = silu_and_mul_backward_kernel(grad_out, gate_up, offsets)
        torch.cuda.synchronize()

        ref_gate = torch.tensor(
            [1.0], device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        ref_up = torch.tensor(
            [2.0], device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        expected = torch.nn.functional.silu(ref_gate) * ref_up
        expected.backward(torch.tensor([3.0], device="cuda", dtype=torch.bfloat16))
        assert ref_gate.grad is not None
        assert ref_up.grad is not None
        torch.testing.assert_close(out[row], expected.detach())
        torch.testing.assert_close(grad_gate_up[row, 0], ref_gate.grad)
        torch.testing.assert_close(grad_gate_up[row, 1], ref_up.grad)
