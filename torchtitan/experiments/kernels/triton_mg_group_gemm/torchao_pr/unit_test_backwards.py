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
    grouped_gemm_dw_tma,
    grouped_gemm_dx_tma,
    grouped_gemm_forward,
    mg_grouped_gemm,
)

from reference_utils import (
    analyze_tensor_differences,
    compute_reference_backward,
    compute_reference_forward,
)


class TestMG_GroupedGEMM_Backward(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2020)  # Set seed for reproducibility

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

        # Run forward pass with our implementation
        result = grouped_gemm_forward(a, b, m_sizes)
        # Ensure result has correct shape
        self.assertTrue(result.shape == (M * G, N))

        # Compute expected result using reference implementation
        expected_result = compute_reference_forward(a, b, m_sizes)

        # Verify forward pass correctness
        forward_close = analyze_tensor_differences(
            result, expected_result, "Forward output"
        )
        self.assertTrue(forward_close)

        # Create a gradient for backpropagation
        grad_output = torch.randn_like(result)

        # Compute gradients using our custom backward implementation
        grad_a, grad_b = grouped_gemm_backward(grad_output, a, b, m_sizes)

        # Compute expected gradients using reference implementation
        expected_grad_a, expected_grad_b = compute_reference_backward(
            a, b, m_sizes, grad_output
        )

        # Verify gradient correctness
        grad_a_close = analyze_tensor_differences(grad_a, expected_grad_a, "grad_x")
        grad_b_close = analyze_tensor_differences(grad_b, expected_grad_b, "grad_w")

        self.assertTrue(grad_a_close)
        self.assertTrue(grad_b_close)

    def test_MG_grouped_gemm_backward_bf16(self) -> None:
        for G in (1, 8, 16):
            for M in (512, 1024):
                print(f"Testing BF16 M*G GroupGeMM Backward with G={G}, M={M}")
                self._run_grouped_gemm_backward_test(
                    (G, M, 1024, 1024),
                    torch.device("cuda"),
                    dtype=torch.float16,
                    atol=1e-2,
                    rtol=1e-2,
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
                shape, device, dtype=torch.float16, atol=1e-2, rtol=1e-2
            )

    def test_MG_dx(self) -> None:
        """Test specifically the dx (gradient w.r.t. input) computation."""
        G, M, N, K = 4, 512, 1024, 2048
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Set up inputs
        a = torch.randn(M * G, K, dtype=dtype, device=device, requires_grad=True)
        b = torch.randn(N, K, dtype=dtype, device=device, requires_grad=True)

        # Create equal-sized groups
        m_size = M
        m_sizes = torch.full((G,), m_size, device=device, dtype=torch.int32)

        # Forward pass
        result = grouped_gemm_forward(a, b, m_sizes)

        # Create gradient for backward
        grad_output = torch.randn_like(result)

        # Compute gradient using our optimized function
        grad_a, _ = grouped_gemm_backward(grad_output, a, b, m_sizes)

        # Compute expected gradient using reference implementation
        expected_grad_a, _ = compute_reference_backward(a, b, m_sizes, grad_output)

        # Verify gradient
        dx_close = analyze_tensor_differences(grad_a, expected_grad_a, "grad_a (dx)")
        self.assertTrue(dx_close)

    def test_MG_dw(self) -> None:
        """Test specifically the dw (gradient w.r.t. weights) computation."""
        G, M, N, K = 4, 512, 1024, 2048
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Set up inputs
        a = torch.randn(M * G, K, dtype=dtype, device=device, requires_grad=True)
        b = torch.randn(N, K, dtype=dtype, device=device, requires_grad=True)

        # Create equal-sized groups
        m_size = M
        m_sizes = torch.full((G,), m_size, device=device, dtype=torch.int32)

        # Forward pass
        result = grouped_gemm_forward(a, b, m_sizes)

        # Create gradient for backward
        grad_output = torch.randn_like(result)

        # Compute gradient using our optimized function
        _, grad_b = grouped_gemm_backward(grad_output, a, b, m_sizes)

        # Compute expected gradient using reference implementation
        _, expected_grad_b = compute_reference_backward(a, b, m_sizes, grad_output)

        # Verify gradient
        dw_close = analyze_tensor_differences(grad_b, expected_grad_b, "grad_b (dw)")
        self.assertTrue(dw_close)
