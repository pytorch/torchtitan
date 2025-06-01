#!/usr/bin/env python3
"""
Simple driver script for CUTLASS CUTE DenseGemmKernel

Generates PyTorch tensors, runs DenseGemmKernel, and compares with PyTorch reference.
All computations in float32.
"""

import time

import torch

# Import CUTLASS CUTE components
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack

    # Import the DenseGemmKernel (assuming it's in the same directory)
    from dense_gemm import DenseGemmKernel

    HAS_CUTLASS = True
    print("✓ CUTLASS CUTE imported successfully")
except ImportError as e:
    HAS_CUTLASS = False
    print(f"✗ CUTLASS CUTE import failed: {e}")
    print(
        "Make sure CUTLASS CUTE is properly installed and the dense_gemm.py file is available"
    )
    exit(1)


def convert_to_mnkl_format(A, B, C):
    """Convert PyTorch tensors to MNKL format for CUTLASS CUTE."""
    # A: (M, K) -> (M, K, 1)
    A_mnkl = A.unsqueeze(-1).contiguous()

    # B: (K, N) -> (N, K, 1)
    B_mnkl = B.transpose(0, 1).unsqueeze(-1).contiguous()

    # C: (M, N) -> (M, N, 1)
    C_mnkl = C.unsqueeze(-1).contiguous()

    return A_mnkl, B_mnkl, C_mnkl


def create_cute_tensors(A_mnkl, B_mnkl, C_mnkl, dtype):
    """Convert PyTorch tensors to CUTE tensors with proper setup."""

    # Convert to CUTE tensors using DLPack
    A_cute = from_dlpack(A_mnkl, assumed_align=16)
    B_cute = from_dlpack(B_mnkl, assumed_align=16)
    C_cute = from_dlpack(C_mnkl, assumed_align=16)

    # Set CUTLASS data types
    A_cute.element_type = dtype
    B_cute.element_type = dtype
    C_cute.element_type = dtype

    # Mark layouts as dynamic with correct leading dimensions
    # For A (M,K,1), the K dimension has stride 1
    A_cute = A_cute.mark_layout_dynamic(leading_dim=1)

    # For B (N,K,1), the K dimension has stride 1
    B_cute = B_cute.mark_layout_dynamic(leading_dim=1)

    # For C (M,N,1), the N dimension has stride 1
    C_cute = C_cute.mark_layout_dynamic(leading_dim=1)

    return A_cute, B_cute, C_cute


def run_dense_gemm_test(M=1024, N=512, K=256, tolerance=1e-4):
    """
    Main test function that runs DenseGemmKernel and compares with PyTorch.

    Args:
        M, N, K: Matrix dimensions for C = A @ B where A is (M,K) and B is (K,N)
        tolerance: Tolerance for comparing results
    """

    print(f"\n{'='*60}")
    print(f"DenseGemmKernel Test: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return False

    device = torch.device("cuda")
    dtype_torch = torch.float32
    dtype_cutlass = cutlass.Float32

    # 1. Generate PyTorch tensors
    print("\n1. Generating PyTorch tensors...")
    torch.manual_seed(42)  # For reproducible results

    A = torch.randn(M, K, dtype=dtype_torch, device=device)  # (M, K)
    B = torch.randn(K, N, dtype=dtype_torch, device=device)  # (K, N)
    C = torch.zeros(M, N, dtype=dtype_torch, device=device)  # (M, N)

    print(f"  A: {A.shape} {A.dtype}")
    print(f"  B: {B.shape} {B.dtype}")
    print(f"  C: {C.shape} {C.dtype}")

    # 2. Compute PyTorch reference
    print("\n2. Computing PyTorch reference...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    C_ref = torch.mm(A, B)

    torch.cuda.synchronize()
    pytorch_time = time.perf_counter() - start_time

    # Store the pytorch_time as an attribute of the function
    run_dense_gemm_test.pytorch_time = pytorch_time

    print(f"  PyTorch time: {pytorch_time*1000:.2f} ms")
    print(f"  Result norm: {torch.norm(C_ref).item():.6f}")

    # 3. Convert to MNKL format
    print("\n3. Converting to MNKL format...")
    A_mnkl, B_mnkl, C_mnkl = convert_to_mnkl_format(A, B, C)

    print(f"  A_mnkl: {A_mnkl.shape} (M, K, L)")
    print(f"  B_mnkl: {B_mnkl.shape} (N, K, L)")
    print(f"  C_mnkl: {C_mnkl.shape} (M, N, L)")

    # 4. Create CUTE tensors
    print("\n4. Creating CUTE tensors...")
    A_cute, B_cute, C_cute = create_cute_tensors(A_mnkl, B_mnkl, C_mnkl, dtype_cutlass)

    print(f"  A_cute: {A_cute.shape}, dtype: {A_cute.element_type}")
    print(f"  B_cute: {B_cute.shape}, dtype: {B_cute.element_type}")
    print(f"  C_cute: {C_cute.shape}, dtype: {C_cute.element_type}")

    # 5. Setup DenseGemmKernel
    print("\n5. Setting up DenseGemmKernel...")

    try:
        gemm_kernel = DenseGemmKernel(
            acc_dtype=cutlass.Float32,  # Accumulator type
            use_2cta_instrs=True,  # Paired CTA
            mma_tiler_mn=(256, 256),  # Tile size
            cluster_shape_mn=(4, 4),  # Cluster size
            use_tma_store=True,  #  store method
        )

        print(f"  ✓ Kernel configured successfully")
        print(f"    MMA tiler: {gemm_kernel.mma_tiler}")
        print(f"    Cluster shape: {gemm_kernel.cluster_shape_mn}")
        print(f"    Use 2CTA: {gemm_kernel.use_2cta_instrs}")
        print(f"    Use TMA store: {gemm_kernel.use_tma_store}")

    except Exception as e:
        print(f"  ✗ Kernel setup failed: {e}")
        return False

    # 6. Create CUDA stream and compile kernel
    print("\n6. Compiling kernel...")

    try:
        torch_stream = torch.cuda.Stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        # Compile the kernel
        compiled_kernel = cute.compile(gemm_kernel, A_cute, B_cute, C_cute, stream)
        print(f"  ✓ Kernel compiled successfully")

    except Exception as e:
        print(f"  ✗ Kernel compilation failed: {e}")
        return False

    # 7. Execute DenseGemmKernel
    print("\n7. Executing DenseGemmKernel...")

    try:
        # Warm up
        for _ in range(3):
            compiled_kernel(A_cute, B_cute, C_cute, stream)
        torch.cuda.synchronize()

        # Actual timing
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        compiled_kernel(A_cute, B_cute, C_cute, stream)

        torch.cuda.synchronize()
        cutlass_time = time.perf_counter() - start_time

        # Store the cutlass_time as an attribute of the function
        run_dense_gemm_test.cutlass_time = cutlass_time

        print(f"  ✓ Kernel executed successfully")
        print(f"  CUTLASS time: {cutlass_time*1000:.2f} ms")

    except Exception as e:
        print(f"  ✗ Kernel execution failed: {e}")
        return False

    # 8. Compare results
    print("\n8. Comparing results...")

    try:
        # Convert CUTLASS result back to standard format
        C_cutlass = C_mnkl.squeeze(-1)  # (M, N, 1) -> (M, N)

        print(f"  PyTorch result norm: {torch.norm(C_ref).item():.6f}")
        print(f"  CUTLASS result norm: {torch.norm(C_cutlass).item():.6f}")

        # Print first 3 elements of each result
        print(f"  PyTorch first 3 elements: {C_ref.flatten()[:3].tolist()}")
        print(f"  CUTLASS first 3 elements: {C_cutlass.flatten()[:3].tolist()}")

        # Compute difference
        diff = C_cutlass - C_ref
        max_diff = torch.max(torch.abs(diff)).item()
        norm_diff = torch.norm(diff).item()
        relative_error = norm_diff / torch.norm(C_ref).item()

        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Norm of difference: {norm_diff:.2e}")
        print(f"  Relative error: {relative_error:.2e}")

        # Check if results match within tolerance
        if max_diff < tolerance:
            print(f"  ✓ Results match within tolerance ({tolerance:.1e})")
            success = True
        else:
            print(f"  ✗ Results exceed tolerance ({tolerance:.1e})")
            success = False

        # Performance comparison
        speedup = pytorch_time / cutlass_time
        print(f"\n  Performance:")
        print(f"    PyTorch: {pytorch_time*1000:.2f} ms")
        print(f"    CUTLASS: {cutlass_time*1000:.2f} ms")
        print(f"    Speedup: {speedup:.2f}x")

        return success

    except Exception as e:
        print(f"  ✗ Result comparison failed: {e}")
        return False


def main():
    """Run multiple test cases."""

    print("CUTLASS CUTE DenseGemmKernel Driver")
    print("Using Float32 for all computations")

    # Test cases: (M, N, K)
    test_cases = [
        (512, 512, 512),  # Small
        (1024, 1024, 512),  # Medium
        (2048, 1024, 1024),  # Large
        (4096, 4096, 2048),  # Extra large
        (8192, 8192, 2048),  # Extra extra large
    ]

    results = []
    speedups = []
    tolerance = 9e-2

    for M, N, K in test_cases:
        success = run_dense_gemm_test(M, N, K, tolerance=tolerance)

        # Extract speedup from the last line of output
        torch_time = getattr(run_dense_gemm_test, "pytorch_time", 0)
        cutlass_time = getattr(run_dense_gemm_test, "cutlass_time", 0)
        speedup = torch_time / cutlass_time if cutlass_time > 0 else 0

        results.append((M, N, K, success))
        speedups.append((M, N, K, torch_time * 1000, cutlass_time * 1000, speedup))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for _, _, _, success in results if success)
    total = len(results)

    for M, N, K, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {M:4d}x{N:4d}x{K:4d}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    # Print speedup table
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(
        f"{'Size (M×N×K)':<20} {'PyTorch (ms)':<15} {'CUTLASS (ms)':<15} {'Speedup':<10}"
    )
    print(f"{'-'*60}")

    for M, N, K, torch_ms, cutlass_ms, speedup in speedups:
        size_str = f"{M}×{N}×{K}"
        print(f"{size_str:<20} {torch_ms:<15.2f} {cutlass_ms:<15.2f} {speedup:<5.2f}x")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
