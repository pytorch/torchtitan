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

from mg_grouped_gemm import grouped_gemm_forward


class TestMG_GroupedGEMM(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2020)

    def _run_grouped_gemm_test(
        self,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        atol: float = 1e-5,
        rtol: float = 1.6e-2,
    ) -> None:
        G, M, N, K = shape
        # In M*G grouping, input is [M*G, K] and weights are [N*G, K]
        a = torch.randn(M * G, K, dtype=dtype, device=device)
        b = torch.randn(N * G, K, dtype=dtype, device=device)

        # Create equal-sized groups for simplicity
        m_size = M
        m_sizes = torch.full((G,), m_size, device=device, dtype=torch.int32)

        result = grouped_gemm_forward(a, b, m_sizes)
        self.assertTrue(result.shape == (M * G, N))

        expected_result = torch.zeros(M * G, N, dtype=dtype, device=device)
        m_start = 0
        for g in range(G):
            m_end = m_start + m_sizes[g]
            b_slice = b[N * g : N * (g + 1), :]
            expected_result[m_start:m_end, :] = a[m_start:m_end, :] @ b_slice.T
            m_start = m_end

        # Convert result to match input dtype if needed
        result = result.to(dtype)
        torch.testing.assert_close(result, expected_result, atol=atol, rtol=rtol)

    def test_MG_grouped_gemm_bf16(self) -> None:
        for G in (1, 4, 16):
            for M in (128, 512, 1024):
                print(f"Testing BF16 M*G GroupGeMM with G={G}, M={M}")
                self._run_grouped_gemm_test(
                    (G, M, 1024, 1024),
                    torch.device("cuda"),
                    dtype=torch.bfloat16,
                    atol=1e-5,
                    rtol=1.6e-2,
                )

    def test_MG_grouped_gemm_deepseek_shapes(self) -> None:
        """Test with shapes from Deepseek model."""
        deepseek_shapes = [
            (4, 2048, 4096, 7168),  # G, M, N, K
            (4, 2048, 7168, 2048),
            (8, 512, 4096, 7168),
            (8, 512, 7168, 2048),
        ]

        device = torch.device("cuda")

        for shape in deepseek_shapes:
            G, M, N, K = shape
            print(f"Testing BF16 M*G Deepseek shape: G={G}, M={M}, N={N}, K={K}")
            self._run_grouped_gemm_test(
                shape, device, dtype=torch.bfloat16, atol=1e-5, rtol=1.6e-2
            )
