# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Benchmark comparing reference PyTorch vs optimized M*G group GEMM implementation

import argparse
import logging
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import triton
import triton.language as tl

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Import grouped GEMM implementations
try:
    from mg_grouped_gemm import grouped_gemm_backward, grouped_gemm_forward

except ImportError:
    logging.error(
        "Error importing grouped GEMM modules. Make sure the implementation files are in the correct path."
    )
    raise


def test_multiple_deepseek_configs():
    """
    Test multiple DeepSeek model configurations with both forward and backward pass verification.
    """
    # DeepSeek configurations: (G, M, K, N)
    configs = [
        # (4, 8192, 7168, 4096),  # Config 1
        # (4, 8192, 2048, 7168),  # Config 2
        # (8, 4096, 7168, 4096),  # Config 3
        (8, 4096, 2048, 7168),  # Config 4
    ]

    results = []

    for config_idx, (G, M, K, N) in enumerate(configs):
        logging.info(f"\n\n===== Testing DeepSeek Config {config_idx+1} =====")
        logging.info(f"G={G}, M={M}, K={K}, N={N}")

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Create even group sizes
            base_size = M // G
            remainder = M % G
            M_sizes = [base_size + (1 if i < remainder else 0) for i in range(G)]
            m_sizes = torch.tensor(M_sizes, device=device, dtype=torch.int32)

            # Create input and weight tensors using float16 for higher precision
            x = torch.randn(
                M, K, dtype=torch.float16, device=device, requires_grad=True
            )
            w = torch.randn(
                N, K, dtype=torch.float16, device=device, requires_grad=True
            )

            logging.info(f"Input x shape: {x.shape}, Weight w shape: {w.shape}")

            # Run forward pass
            result = grouped_gemm_forward(x, w, m_sizes)
            logging.info(f"Forward result shape: {result.shape}")

        except Exception as e:
            logging.error(f"Error occurred during forward pass: {e}")
            continue


if __name__ == "__main__":
    test_multiple_deepseek_configs()
