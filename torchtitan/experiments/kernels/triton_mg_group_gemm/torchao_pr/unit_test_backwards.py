# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import logging
import unittest
from typing import Tuple

import torch
import torch.nn as nn

from mg_grouped_gemm import (
    grouped_gemm_backward,
    grouped_gemm_dw_optimized,
    grouped_gemm_dx_optimized,
    grouped_gemm_forward,
    mg_grouped_gemm,
)


class TestMG_GroupedGEMM_Backward(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2024)  # Set seed for reproducibility

    def _run_grouped_gemm_backward_test(
        self,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        atol: float = 1e-5,
        rtol: float = 1.6e-2,
    ) -> None:
        G, M, N, K = shape
        # Set up inputs for forward pass
        # In M*G grouping, input is [M*G, K] and weights are [N, K]
        a = torch.randn(M * G, K, dtype=dtype, device=device, requires_grad=True)
        b = torch.randn(N, K, dtype=dtype, device=device, requires_grad=True)

        # Create equal-sized groups for simplicity
        m_size = M
        m_sizes = torch.full((G,), m_size, device=device, dtype=torch.int32)

        # Run forward pass with autograd tracking
        def run_forward():
            result = grouped_gemm_forward(a, b, m_sizes)
            # Ensure result has correct shape
            self.assertTrue(result.shape == (M * G, N))
            return result

        # Compute expected result using PyTorch operations
        def compute_expected_forward():
            expected = torch.zeros(M * G, N, dtype=dtype, device=device)
            for g in range(G):
                m_start = g * m_size
                m_end = (g + 1) * m_size
                expected[m_start:m_end, :] = a[m_start:m_end, :] @ b.T
            return expected

        # Run forward pass and save outputs
        result = run_forward()
        expected_result = compute_expected_forward()

        # Verify forward pass correctness
        torch.testing.assert_close(
            result.to(dtype), expected_result, atol=atol, rtol=rtol
        )

        # Create a gradient for backpropagation
        grad_output = torch.randn_like(result)

        # Compute gradients using our custom backward implementation
        grad_a, grad_b = grouped_gemm_backward(grad_output, a, b, m_sizes)

        # Compute expected gradients using PyTorch autograd
        a_for_torch = a.detach().clone().requires_grad_(True)
        b_for_torch = b.detach().clone().requires_grad_(True)

        # Compute the expected forward output
        expected_output = torch.zeros(M * G, N, dtype=dtype, device=device)
        for g in range(G):
            m_start = g * m_size
            m_end = (g + 1) * m_size
            expected_output[m_start:m_end, :] = (
                a_for_torch[m_start:m_end, :] @ b_for_torch.T
            )

        # Backpropagate through the expected output
        expected_output.backward(grad_output)

        # Get expected gradients
        expected_grad_a = a_for_torch.grad
        expected_grad_b = b_for_torch.grad

        # Verify gradient correctness
        torch.testing.assert_close(grad_a, expected_grad_a, atol=atol, rtol=rtol)
        torch.testing.assert_close(grad_b, expected_grad_b, atol=atol, rtol=rtol)

    def _run_end_to_end_autograd_test(
        self,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        atol: float = 1e-5,
        rtol: float = 1.6e-2,
    ) -> None:
        G, M, N, K = shape
        # Set up inputs
        a = torch.randn(M * G, K, dtype=dtype, device=device, requires_grad=True)
        b = torch.randn(N, K, dtype=dtype, device=device, requires_grad=True)

        # Create equal-sized groups
        m_size = M
        m_sizes = torch.full((G,), m_size, device=device, dtype=torch.int32)

        # Create copies for PyTorch reference computation
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        # Forward pass using our custom grouped_gemm with autograd
        result = mg_grouped_gemm(a, b, m_sizes)

        # Reference implementation using PyTorch operations
        result_ref = torch.zeros(M * G, N, dtype=dtype, device=device)
        for g in range(G):
            m_start = g * m_size
            m_end = (g + 1) * m_size
            result_ref[m_start:m_end, :] = a_ref[m_start:m_end, :] @ b_ref.T

        # Verify forward pass
        torch.testing.assert_close(
            result.to(dtype), result_ref.to(dtype), atol=atol, rtol=rtol
        )

        # Create a gradient for backpropagation
        grad_output = torch.randn_like(result)
        grad_output_ref = grad_output.clone()

        # Backward pass
        result.backward(grad_output)
        result_ref.backward(grad_output_ref)

        # Verify gradients
        torch.testing.assert_close(a.grad, a_ref.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(b.grad, b_ref.grad, atol=atol, rtol=rtol)

    def test_MG_grouped_gemm_backward_bf16(self) -> None:
        for G in (1, 4, 16):
            for M in (64, 512, 1024):
                print(f"Testing BF16 M*G GroupGeMM Backward with G={G}, M={M}")
                self._run_grouped_gemm_backward_test(
                    (G, M, 1024, 1024),
                    torch.device("cuda"),
                    dtype=torch.bfloat16,
                    atol=1e-5,
                    rtol=1.6e-2,
                )

    def test_MG_grouped_gemm_backward_deepseek_shapes(self) -> None:
        """Test backward pass with shapes from Deepseek model."""
        deepseek_shapes = [
            (4, 2048, 4096, 7168),  # G, M, N, K
            (4, 2048, 7168, 2048),
            (8, 512, 4096, 7168),
            (8, 512, 7168, 2048),
        ]

        device = torch.device("cuda")

        for shape in deepseek_shapes:
            G, M, N, K = shape
            print(
                f"Testing BF16 M*G Deepseek Backward shape: G={G}, M={M}, N={N}, K={K}"
            )
            self._run_grouped_gemm_backward_test(
                shape, device, dtype=torch.bfloat16, atol=1e-5, rtol=1.6e-2
            )

    def test_MG_grouped_gemm_autograd_integration(self) -> None:
        """Test autograd integration with the custom GroupedGEMM_mg class."""
        shapes = [
            (1, 64, 128, 256),  # Small shapes for quick testing
            (4, 512, 1024, 1024),  # Medium shapes
        ]

        device = torch.device("cuda")

        for shape in shapes:
            G, M, N, K = shape
            print(f"Testing M*G AutoGrad Integration: G={G}, M={M}, N={N}, K={K}")
            self._run_end_to_end_autograd_test(
                shape, device, dtype=torch.bfloat16, atol=1e-5, rtol=1.6e-2
            )

    def test_MG_dx_optimized(self) -> None:
        """Test specifically the dx (gradient w.r.t. input) computation."""
        G, M, N, K = 4, 512, 1024, 2048
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Set up inputs
        a = torch.randn(M * G, K, dtype=dtype, device=device, requires_grad=True)
        b = torch.randn(N, K, dtype=dtype, device=device)

        # Create equal-sized groups
        m_size = M
        m_sizes = torch.full((G,), m_size, device=device, dtype=torch.int32)

        # Forward pass
        result = grouped_gemm_forward(a, b, m_sizes)

        # Create gradient for backward
        grad_output = torch.randn_like(result)

        # Compute gradient using our optimized function
        grad_a = grouped_gemm_dx_optimized(grad_output, b, m_sizes, num_sms=108)

        # Compute expected gradient using PyTorch
        a_torch = a.detach().clone().requires_grad_(True)
        result_torch = torch.zeros(M * G, N, dtype=dtype, device=device)
        for g in range(G):
            m_start = g * m_size
            m_end = (g + 1) * m_size
            result_torch[m_start:m_end, :] = a_torch[m_start:m_end, :] @ b.T

        result_torch.backward(grad_output)
        expected_grad_a = a_torch.grad

        # Verify gradient
        torch.testing.assert_close(grad_a, expected_grad_a, atol=1e-5, rtol=1.6e-2)

    def test_MG_dw_optimized(self) -> None:
        """Test specifically the dw (gradient w.r.t. weights) computation."""
        G, M, N, K = 4, 512, 1024, 2048
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Set up inputs
        a = torch.randn(M * G, K, dtype=dtype, device=device)
        b = torch.randn(N, K, dtype=dtype, device=device, requires_grad=True)

        # Create equal-sized groups
        m_size = M
        m_sizes = torch.full((G,), m_size, device=device, dtype=torch.int32)

        # Forward pass
        result = grouped_gemm_forward(a, b, m_sizes)

        # Create gradient for backward
        grad_output = torch.randn_like(result)

        # Compute gradient using our optimized function
        grad_b = grouped_gemm_dw_optimized(a, grad_output, m_sizes, num_sms=108)

        # Compute expected gradient using PyTorch
        b_torch = b.detach().clone().requires_grad_(True)
        result_torch = torch.zeros(M * G, N, dtype=dtype, device=device)
        for g in range(G):
            m_start = g * m_size
            m_end = (g + 1) * m_size
            result_torch[m_start:m_end, :] = a[m_start:m_end, :] @ b_torch.T

        result_torch.backward(grad_output)
        expected_grad_b = b_torch.grad

        # Verify gradient
        torch.testing.assert_close(grad_b, expected_grad_b, atol=1e-5, rtol=1.6e-2)
