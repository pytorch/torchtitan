#!/usr/bin/env python3
"""
Simple test script to verify CUTLASS GEMM functionality.
"""

import torch
from torchtitan.components.gemm_utils import CUTLASS_AVAILABLE, CutlassGemmManager


torch.backends.cuda.matmul.allow_tf32 = True


def test_basic_gemm():
    """Test basic GEMM functionality."""
    print(f"CUTLASS Available: {CUTLASS_AVAILABLE}")

    if not CUTLASS_AVAILABLE:
        print("CUTLASS not available, skipping test")
        return

    # Test parameters
    device = "cuda"
    dtype = torch.float32
    m, n, k = 256, 256, 256

    print(f"Testing GEMM with matrix sizes: {m}x{k} @ {k}x{n}")

    # Create test matrices
    torch.manual_seed(42)
    A = torch.randn(m, k, dtype=dtype, device=device)
    B = torch.randn(k, n, dtype=dtype, device=device)

    # PyTorch reference
    torch_result = torch.mm(A, B)
    print(f"PyTorch result shape: {torch_result.shape}")

    # CUTLASS GEMM
    try:
        gemm_manager = CutlassGemmManager(dtype=dtype, device=device)
        cutlass_result = gemm_manager.gemm(A, B)
        print(f"CUTLASS result shape: {cutlass_result.shape}")

        # Check correctness
        if torch.allclose(torch_result, cutlass_result, rtol=1e-4, atol=1e-6):
            print("✓ Results match!")
            max_diff = torch.max(torch.abs(torch_result - cutlass_result)).item()
            print(f"  Max absolute difference: {max_diff:.2e}")
        else:
            print("✗ Results don't match!")
            max_diff = torch.max(torch.abs(torch_result - cutlass_result)).item()
            print(f"  Max absolute difference: {max_diff:.2e}")

    except Exception as e:
        print(f"CUTLASS GEMM failed: {e}")


if __name__ == "__main__":
    test_basic_gemm()
