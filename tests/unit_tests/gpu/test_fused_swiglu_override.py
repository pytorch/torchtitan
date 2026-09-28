# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.activation import SwiGLU
from torchtitan.overrides.fused_swiglu import (
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
        offsets = torch.tensor([3, 5, 6, 8], device="cuda", dtype=torch.int32)
        gate_reference = gate.detach().clone().requires_grad_()
        up_reference = up.detach().clone().requires_grad_()
        gate_fused = gate.detach().clone().requires_grad_()
        up_fused = up.detach().clone().requires_grad_()

        out_reference = reference(gate_reference, up_reference, offsets=offsets)
        out_fused = fused(gate_fused, up_fused, offsets=offsets)
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
        gate = torch.as_strided(storage, (row + 1, 1), (row_stride, 2))
        up = torch.as_strided(storage, (row + 1, 1), (row_stride, 2), 1)
        gate[row] = 1.0
        up[row] = 2.0
        offsets = torch.tensor([row + 1], device="cuda", dtype=torch.int32)

        out = silu_and_mul_forward_kernel(gate, up, offsets)
        grad_out = torch.zeros_like(out)
        grad_out[row] = 3.0
        grad_gate, grad_up = silu_and_mul_backward_kernel(
            grad_out,
            gate,
            up,
            offsets,
        )
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
        torch.testing.assert_close(grad_gate[row], ref_gate.grad)
        torch.testing.assert_close(grad_up[row], ref_up.grad)
