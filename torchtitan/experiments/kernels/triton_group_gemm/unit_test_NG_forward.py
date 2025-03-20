# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import unittest
from typing import Tuple

import torch
import torch.nn as nn

from NG_forward import grouped_gemm_forward

# from fbgemm_baseline import grouped_gemm_forward

# refactored and expanded from:  https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gemm/test/grouped_gemm_test.py


class TestNG_GroupedGEMM(unittest.TestCase):
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
        a = torch.randn(M, K, dtype=dtype, device=device)
        b = torch.randn(N * G, K, dtype=dtype, device=device)
        m_ends, _ = torch.sort(
            torch.randint(low=0, high=M, size=[G - 1], device=device, dtype=torch.int32)
        )
        m_ends = m_ends.tolist()
        m_starts = [0] + m_ends
        m_ends = m_ends + [M]
        m_sizes = torch.tensor(
            [m_ends[i] - m_starts[i] for i in range(G)], device=device
        ).to(torch.int32)

        result = grouped_gemm_forward(a, b, m_sizes)
        self.assertTrue(result.shape == (M, N))

        expected_result = torch.zeros(M, N, dtype=dtype, device=device)
        for g in range(G):
            m_start = m_starts[g]
            m_end = m_ends[g]
            expected_result[m_start:m_end, :] = (
                a[m_start:m_end, :] @ b[g * N : (g + 1) * N, :].T
            )

        torch.testing.assert_close(result, expected_result, atol=atol, rtol=rtol)

    def test_NG_grouped_gemm_bf16(self) -> None:
        for G in (1, 4, 16):
            for M in (64, 512, 2048):
                print(f"Testing BF16 N*G GroupGeMM with G={G}, M={M}")
                self._run_grouped_gemm_test(
                    (G, M, 1024, 1024),
                    torch.device("cuda"),
                    dtype=torch.bfloat16,
                    atol=1e-5,
                    rtol=1.6e-2,
                )

    def test_NG_grouped_gemm_deepseek_shapes(self) -> None:
        """Test with shapes from Deepseek model."""
        deepseek_shapes = [
            (4, 8192, 4096, 7168),  # G, M, N, K
            (4, 8192, 7168, 2048),
            (8, 4096, 4096, 7168),
            (8, 4096, 7168, 2048),
        ]

        device = torch.device("cuda")

        for shape in deepseek_shapes:
            G, M, N, K = shape
            print(f"Testing BF16 N*G Deepseek shape: G={G}, M={M}, N={N}, K={K}")
            self._run_grouped_gemm_test(
                shape, device, dtype=torch.bfloat16, atol=1e-2, rtol=1e-2
            )
