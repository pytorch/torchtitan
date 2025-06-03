#!/usr/bin/env python3
"""
CUTLASS CUTE DenseGemmKernel Benchmark using Triton's do_bench

Keeps the original CUTLASS CUTE kernel but uses Triton's robust benchmarking
infrastructure for more accurate and consistent timing measurements.
"""

import torch
import triton.testing

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
    print("CUTLASS Python imported successfully")
except ImportError as e:
    HAS_CUTLASS = False
    print(f"CUTLASS Python import failed: {e}")
    print(
        "Make sure CUTLASS Python is properly installed and the dense_gemm.py file is available"
    )
    exit(1)


class CutlassGemmBenchmark:
    """Wrapper class to encapsulate CUTLASS GEMM operations for benchmarking."""

    def __init__(self, M, N, K):
        """Initialize benchmark with given matrix dimensions."""
        self.M, self.N, self.K = M, N, K
        self.device = torch.device("cuda")
        self.dtype_torch = torch.float32
        self.dtype_cutlass = cutlass.Float32

        # Pre-generate tensors
        torch.manual_seed(42)
        self.A = torch.randn(M, K, dtype=self.dtype_torch, device=self.device)
        self.B = torch.randn(K, N, dtype=self.dtype_torch, device=self.device)
        self.C = torch.zeros(M, N, dtype=self.dtype_torch, device=self.device)

        # Setup CUTLASS tensors and kernel once
        self._setup_cutlass()

    def _convert_to_mnkl_format(self, A, B, C):
        """Convert PyTorch tensors to MNKL format for CUTLASS CUTE."""
        A_mnkl = A.unsqueeze(-1).contiguous()
        B_mnkl = B.transpose(0, 1).unsqueeze(-1).contiguous()
        C_mnkl = C.unsqueeze(-1).contiguous()
        return A_mnkl, B_mnkl, C_mnkl

    def _create_cute_tensors(self, A_mnkl, B_mnkl, C_mnkl, dtype):
        """Convert PyTorch tensors to CUTE tensors with proper setup."""
        A_cute = from_dlpack(A_mnkl, assumed_align=16)
        B_cute = from_dlpack(B_mnkl, assumed_align=16)
        C_cute = from_dlpack(C_mnkl, assumed_align=16)

        A_cute.element_type = dtype
        B_cute.element_type = dtype
        C_cute.element_type = dtype

        A_cute = A_cute.mark_layout_dynamic(leading_dim=1)
        B_cute = B_cute.mark_layout_dynamic(leading_dim=1)
        C_cute = C_cute.mark_layout_dynamic(leading_dim=1)

        return A_cute, B_cute, C_cute

    def _setup_cutlass(self):
        """Setup CUTLASS kernel and tensors."""
        # Convert to MNKL format
        self.A_mnkl, self.B_mnkl, self.C_mnkl = self._convert_to_mnkl_format(
            self.A, self.B, self.C
        )

        # Create CUTE tensors
        self.A_cute, self.B_cute, self.C_cute = self._create_cute_tensors(
            self.A_mnkl, self.B_mnkl, self.C_mnkl, self.dtype_cutlass
        )

        # Setup kernel
        self.gemm_kernel = DenseGemmKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=True,
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(4, 4),
            use_tma_store=True,
        )

        # Setup stream and compile
        self.torch_stream = torch.cuda.Stream()
        self.stream = cuda.CUstream(self.torch_stream.cuda_stream)
        self.compiled_kernel = cute.compile(
            self.gemm_kernel, self.A_cute, self.B_cute, self.C_cute, self.stream
        )

    def pytorch_gemm(self):
        """Execute PyTorch GEMM."""
        return torch.mm(self.A, self.B)

    def cutlass_gemm(self):
        """Execute CUTLASS GEMM."""
        # Reset output tensor
        self.C_mnkl.zero_()
        # Execute kernel
        self.compiled_kernel(self.A_cute, self.B_cute, self.C_cute, self.stream)
        # Return result in standard format
        return self.C_mnkl.squeeze(-1)


def validate_implementation(M, N, K, tolerance=1e-4):
    """
    Validate that CUTLASS implementation matches PyTorch reference.

    Args:
        M, N, K: Matrix dimensions
        tolerance: Acceptable difference threshold

    Returns:
        bool: True if results match within tolerance
    """
    print(f"\nValidating M={M}, N={N}, K={K}")

    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return False

    try:
        benchmark = CutlassGemmBenchmark(M, N, K)

        # Compute both results
        pytorch_result = benchmark.pytorch_gemm()
        cutlass_result = benchmark.cutlass_gemm()

        # Compare results
        diff = torch.abs(pytorch_result - cutlass_result)
        max_diff = torch.max(diff).item()
        rel_error = torch.norm(diff) / torch.norm(pytorch_result)

        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Relative error: {rel_error:.2e}")

        if max_diff < tolerance:
            print(f"  ✓ PASSED (tolerance: {tolerance:.1e})")
            return True
        else:
            print(f"  ✗ FAILED (tolerance: {tolerance:.1e})")
            return False

    except Exception as e:
        print(f"  ✗ Error during validation: {e}")
        return False


def benchmark_gemm(M, N, K, warmup=3, rep=10):
    """
    Benchmark CUTLASS vs PyTorch GEMM using Triton's do_bench.

    Args:
        M, N, K: Matrix dimensions
        warmup: Number of warmup iterations
        rep: Number of benchmark repetitions

    Returns:
        dict: Timing results and metrics
    """
    print(f"\nBenchmarking M={M}, N={N}, K={K}")

    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return None

    try:
        benchmark = CutlassGemmBenchmark(M, N, K)

        # Benchmark PyTorch
        pytorch_time = triton.testing.do_bench(
            benchmark.pytorch_gemm, warmup=warmup, rep=rep
        )

        # Benchmark CUTLASS
        cutlass_time = triton.testing.do_bench(
            benchmark.cutlass_gemm, warmup=warmup, rep=rep
        )

        # Calculate metrics
        flops = 2.0 * M * N * K  # GEMM FLOPs
        pytorch_tflops = flops / (pytorch_time * 1e-3) / 1e12
        cutlass_tflops = flops / (cutlass_time * 1e-3) / 1e12
        speedup = pytorch_time / cutlass_time

        results = {
            "M": M,
            "N": N,
            "K": K,
            "pytorch_ms": pytorch_time,
            "cutlass_ms": cutlass_time,
            "pytorch_tflops": pytorch_tflops,
            "cutlass_tflops": cutlass_tflops,
            "speedup": speedup,
            "flops": flops,
        }

        print(f"  PyTorch: {pytorch_time:.2f} ms ({pytorch_tflops:.2f} TFLOPS)")
        print(f"  CUTLASS: {cutlass_time:.2f} ms ({cutlass_tflops:.2f} TFLOPS)")
        print(f"  Speedup: {speedup:.2f}x")

        return results

    except Exception as e:
        print(f"  ✗ Benchmark failed: {e}")
        return None


def run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    print("CUTLASS CUTE DenseGemmKernel Benchmark Suite")
    print("Using Triton's do_bench for accurate timing")
    print("=" * 60)

    # Test cases: (M, N, K)
    test_cases = [
        (512, 512, 512),  # Small
        (1024, 1024, 512),  # Medium
        (1024, 1024, 1024),  # Medium square
        (2048, 1024, 1024),  # Large rectangular
        (2048, 2048, 1024),  # Large
        (4096, 4096, 2048),  # Extra large
        (8192, 8192, 2048),  # Very large
        (8192, 8192, 4096),  # Very large square
    ]

    # Validation phase
    print("\n" + "=" * 60)
    print("VALIDATION PHASE")
    print("=" * 60)

    validation_results = []
    tolerance = 5e-1  # Relaxed tolerance for large matrices

    for M, N, K in test_cases:
        success = validate_implementation(M, N, K, tolerance=tolerance)
        validation_results.append((M, N, K, success))

    # Benchmark phase
    print("\n" + "=" * 60)
    print("BENCHMARK PHASE")
    print("=" * 60)

    benchmark_results = []

    for M, N, K in test_cases:
        result = benchmark_gemm(M, N, K, warmup=3, rep=10)
        if result:
            benchmark_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, _, _, success in validation_results if success)
    total = len(validation_results)

    for M, N, K, success in validation_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {M:4d}×{N:4d}×{K:4d}: {status}")

    print(f"\nValidation: {passed}/{total} tests passed")

    if benchmark_results:
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(
            f"{'Size (M×N×K)':<15} {'PyTorch ms':<8} {'CUTLASS ms':<8} {'Speedup x':<8} {'CUTLASS TFLOPS':<8}"
        )
        print("-" * 70)

        avg_speedup = 0
        for result in benchmark_results:
            size_str = f"{result['M']}×{result['N']}×{result['K']}"
            print(
                f"{size_str:<18} {result['pytorch_ms']:<11.2f} {result['cutlass_ms']:<10.2f} "
                f"{result['speedup']:<6.2f} {result['cutlass_tflops']:<18.2f}"
            )
            avg_speedup += result["speedup"]

        avg_speedup /= len(benchmark_results)
        print(f"\nAverage speedup: {avg_speedup:.2f}x")

    # Final status
    if passed == total and benchmark_results:
        print("\n🎉 All tests passed and benchmarks completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed or benchmarks incomplete")
        return 1


if __name__ == "__main__":
    exit(run_benchmark_suite())
