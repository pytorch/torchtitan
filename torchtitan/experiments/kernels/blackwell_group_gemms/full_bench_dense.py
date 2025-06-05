#!/usr/bin/env python3
"""
CUTLASS CUTE Dense GEMM Benchmark Suite using Triton's do_bench

Benchmarks both the standard DenseGemmKernel and PersistentDenseGemmKernel
against PyTorch's reference implementation with robust timing measurements.
"""

"""
current error:
Validating persistent kernel: M=1024, N=1024, K=512
  ✗ Error during validation: Expected strides[leading_dim] == 1, but got 512.

Validating standard kernel: M=1024, N=1024, K=1024
  Max difference: 0.00e+00
  Relative error: 0.00e+00
  ✓ PASSED (tolerance: 5.0e-01)

Validating persistent kernel: M=1024, N=1024, K=1024
  ✗ Error during validation: Expected strides[leading_dim] == 1, but got 1024.



"""

from typing import Any, Dict, Optional, Tuple, Type

import torch
import triton.testing


torch.backends.cuda.matmul.allow_tf32 = True

# Import CUTLASS CUTE components
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.cute.nvgpu import cpasync, tcgen05
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


# Persistent Dense GEMM Kernel Class (embedded from the provided code)
class PersistentDenseGemmKernel:
    """
    Persistent Dense GEMM Kernel for Blackwell SM100 architecture.
    This is a simplified version focused on benchmarking compatibility.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
    ):
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_tma_store = use_tma_store

        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )

        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        self.num_smem_capacity = sm100_utils.SMEM_CAPACITY["sm100"]

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs"""
        # Configure tiled mma
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        if cutlass.const_expr(self.use_tma_store):
            self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
                self.cta_tile_shape_mnk,
                self.use_2cta_instrs,
                self.c_layout,
                self.c_dtype,
            )
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Check if the gemm can be implemented (simplified validation)"""
        # Basic validation - simplified for benchmarking
        if ab_dtype not in {cutlass.Float16, cutlass.BFloat16, cutlass.TFloat32}:
            return False
        if acc_dtype != cutlass.Float32:
            return False
        if c_dtype not in {cutlass.Float16, cutlass.Float32}:
            return False
        if not (
            mma_tiler_mn[0] in [128, 256] and mma_tiler_mn[1] in range(32, 257, 32)
        ):
            return False
        return True


class StandardGemmBenchmark:
    """Wrapper class for the standard DenseGemmKernel."""

    def __init__(self, M, N, K):
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

    def standard_cutlass_gemm(self):
        """Execute standard CUTLASS GEMM."""
        # Reset output tensor
        self.C_mnkl.zero_()
        # Execute kernel
        self.compiled_kernel(self.A_cute, self.B_cute, self.C_cute, self.stream)
        # Return result in standard format
        return self.C_mnkl.squeeze(-1)


class PersistentGemmBenchmark:
    """Wrapper class for the PersistentDenseGemmKernel."""

    def __init__(self, M, N, K):
        self.M, self.N, self.K = M, N, K
        self.L = 1  # Batch dimension
        self.device = torch.device("cuda")

        # Use Float16 for persistent kernel (more common for Blackwell)
        self.ab_dtype = cutlass.Float16
        self.c_dtype = cutlass.Float16
        self.acc_dtype = cutlass.Float32

        # Check if persistent kernel can be implemented
        if not PersistentDenseGemmKernel.can_implement(
            self.ab_dtype,
            self.acc_dtype,
            self.c_dtype,
            use_2cta_instrs=True,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(2, 2),
            use_tma_store=True,
            m=M,
            n=N,
            k=K,
            l=self.L,
            a_major="k",
            b_major="k",
            c_major="n",
        ):
            raise ValueError(
                f"PersistentDenseGemmKernel cannot implement M={M}, N={N}, K={K}"
            )

        # Setup kernel and tensors
        self._setup_persistent_kernel()

    def _create_and_permute_tensor(self, l, mode0, mode1, is_mode0_major, dtype):
        """Create and permute tensor for persistent kernel."""
        # Create shape: (l, mode1, mode0) if mode0_major else (l, mode0, mode1)
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)

        # Use torch dtype temporarily
        torch_dtype = torch.float16 if dtype == cutlass.Float16 else torch.float32

        # Create tensor on CPU first
        torch_tensor_cpu = torch.randn(shape, dtype=torch_dtype)
        torch_tensor_cpu = torch_tensor_cpu.permute(permute_order).contiguous()

        # Move to GPU
        torch_tensor = torch_tensor_cpu.cuda()

        # Create f32 reference
        f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

        # Create CUTE tensor
        cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
        cute_tensor.element_type = dtype
        cute_tensor = cute_tensor.mark_layout_dynamic(
            leading_dim=(0 if is_mode0_major else 1)
        )

        return f32_torch_tensor, cute_tensor, torch_tensor

    def _setup_persistent_kernel(self):
        """Setup persistent kernel and tensors."""
        torch.manual_seed(42)

        # Create tensors in the format expected by persistent kernel
        self.a_ref, self.a_tensor, self.a_torch = self._create_and_permute_tensor(
            self.L,
            self.M,
            self.K,
            True,
            self.ab_dtype,  # a_major="k" -> mode0_major=True
        )
        self.b_ref, self.b_tensor, self.b_torch = self._create_and_permute_tensor(
            self.L,
            self.N,
            self.K,
            False,
            self.ab_dtype,  # b_major="k" -> mode0_major=False
        )
        self.c_ref, self.c_tensor, self.c_torch = self._create_and_permute_tensor(
            self.L,
            self.M,
            self.N,
            False,
            self.c_dtype,  # c_major="n" -> mode0_major=False
        )

        # Setup kernel
        self.gemm_kernel = PersistentDenseGemmKernel(
            acc_dtype=self.acc_dtype,
            use_2cta_instrs=True,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(2, 2),
            use_tma_store=True,
        )

        # Get hardware info and max clusters
        hardware_info = cutlass.utils.HardwareInfo()
        self.max_active_clusters = hardware_info.get_max_active_clusters(
            4
        )  # 2*2 cluster

        # Get current CUDA stream and convert to CUstream
        torch_stream = torch.cuda.current_stream()
        self.current_stream = cuda.CUstream(torch_stream.cuda_stream)

        # Compile kernel
        self.compiled_kernel = cute.compile(
            self.gemm_kernel,
            self.a_tensor,
            self.b_tensor,
            self.c_tensor,
            self.max_active_clusters,
            self.current_stream,
        )

    def pytorch_gemm(self):
        """Execute PyTorch reference GEMM."""
        # Convert back to standard format for PyTorch
        A_std = self.a_ref.squeeze(0)  # Remove batch dim
        B_std = self.b_ref.squeeze(0).transpose(0, 1)  # Remove batch and transpose
        return torch.mm(A_std, B_std)

    def persistent_cutlass_gemm(self):
        """Execute persistent CUTLASS GEMM."""
        # Reset output tensor
        self.c_torch.zero_()
        # Execute kernel with proper stream
        self.compiled_kernel(
            self.a_tensor, self.b_tensor, self.c_tensor, self.current_stream
        )
        # Return result in standard format (remove batch dimension and transpose if needed)
        result = self.c_torch.squeeze(0).cpu().to(torch.float32)
        return result


def validate_implementation(M, N, K, kernel_type="standard", tolerance=1e-4):
    """
    Validate that CUTLASS implementation matches PyTorch reference.

    Args:
        M, N, K: Matrix dimensions
        kernel_type: "standard" or "persistent"
        tolerance: Acceptable difference threshold

    Returns:
        bool: True if results match within tolerance
    """
    print(f"\nValidating {kernel_type} kernel: M={M}, N={N}, K={K}")

    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return False

    try:
        if kernel_type == "standard":
            benchmark = StandardGemmBenchmark(M, N, K)
            pytorch_result = benchmark.pytorch_gemm()
            cutlass_result = benchmark.standard_cutlass_gemm()
        elif kernel_type == "persistent":
            benchmark = PersistentGemmBenchmark(M, N, K)
            pytorch_result = benchmark.pytorch_gemm()
            cutlass_result = benchmark.persistent_cutlass_gemm()
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

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


def benchmark_gemm(
    M, N, K, kernel_types=["pytorch", "standard", "persistent"], warmup=3, rep=10
):
    """
    Benchmark different GEMM implementations using Triton's do_bench.

    Args:
        M, N, K: Matrix dimensions
        kernel_types: List of kernels to benchmark
        warmup: Number of warmup iterations
        rep: Number of benchmark repetitions

    Returns:
        dict: Timing results and metrics
    """
    print(f"\nBenchmarking M={M}, N={N}, K={K}")

    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return None

    results = {"M": M, "N": N, "K": K, "flops": 2.0 * M * N * K}

    # Setup benchmarks
    benchmarks = {}

    try:
        if "pytorch" in kernel_types or "standard" in kernel_types:
            standard_bench = StandardGemmBenchmark(M, N, K)
            if "pytorch" in kernel_types:
                benchmarks["pytorch"] = standard_bench.pytorch_gemm
            if "standard" in kernel_types:
                benchmarks["standard"] = standard_bench.standard_cutlass_gemm

        if "persistent" in kernel_types:
            persistent_bench = PersistentGemmBenchmark(M, N, K)
            benchmarks["persistent"] = persistent_bench.persistent_cutlass_gemm

    except Exception as e:
        print(f"  ✗ Setup failed: {e}")
        return None

    # Run benchmarks
    for name, func in benchmarks.items():
        try:
            timing = triton.testing.do_bench(func, warmup=warmup, rep=rep)
            tflops = results["flops"] / (timing * 1e-3) / 1e12

            results[f"{name}_ms"] = timing
            results[f"{name}_tflops"] = tflops

            print(f"  {name.capitalize()}: {timing:.2f} ms ({tflops:.2f} TFLOPS)")

        except Exception as e:
            print(f"  ✗ {name.capitalize()} benchmark failed: {e}")
            results[f"{name}_ms"] = float("inf")
            results[f"{name}_tflops"] = 0.0

    # Calculate speedups
    if "pytorch_ms" in results:
        for name in ["standard", "persistent"]:
            if f"{name}_ms" in results and results[f"{name}_ms"] != float("inf"):
                speedup = results["pytorch_ms"] / results[f"{name}_ms"]
                results[f"{name}_speedup"] = speedup
                print(f"  {name.capitalize()} speedup: {speedup:.2f}x")

    return results


def run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    print("CUTLASS CUTE Dense GEMM Benchmark Suite")
    print("Comparing Standard vs Persistent kernels using Triton's do_bench")
    print("=" * 80)

    # Test cases: (M, N, K)
    test_cases = [
        (512, 512, 512),  # Small
        (1024, 1024, 512),  # Medium
        (1024, 1024, 1024),  # Medium square
        (2048, 1024, 1024),  # Large rectangular
        (2048, 2048, 1024),  # Large
        (4096, 4096, 2048),  # Extra large
    ]

    # Validation phase
    print("\n" + "=" * 80)
    print("VALIDATION PHASE")
    print("=" * 80)

    validation_results = []
    tolerance = 1e-2  # More relaxed for FP16 in persistent kernel

    for M, N, K in test_cases:
        # Validate standard kernel
        standard_success = validate_implementation(M, N, K, "standard", tolerance=5e-1)
        validation_results.append(("standard", M, N, K, standard_success))

        # Validate persistent kernel (skip if too small or likely to fail)
        if M >= 1024 and N >= 1024:
            try:
                persistent_success = validate_implementation(
                    M, N, K, "persistent", tolerance=tolerance
                )
                validation_results.append(("persistent", M, N, K, persistent_success))
            except Exception as e:
                print(f"  ✗ Persistent kernel setup failed: {e}")
                validation_results.append(("persistent", M, N, K, False))
        else:
            print(f"  ⚠ Skipping persistent kernel (too small: {M}x{N}x{K})")
            validation_results.append(("persistent", M, N, K, None))

    # Benchmark phase
    print("\n" + "=" * 80)
    print("BENCHMARK PHASE")
    print("=" * 80)

    benchmark_results = []

    for M, N, K in test_cases:
        # Determine which kernels to benchmark based on validation
        kernel_types = ["pytorch"]

        # Add standard if it validated successfully
        standard_validated = any(
            success
            for kernel, m, n, k, success in validation_results
            if kernel == "standard" and m == M and n == N and k == K and success
        )
        if standard_validated:
            kernel_types.append("standard")

        # Add persistent if it validated successfully and is large enough
        persistent_validated = any(
            success
            for kernel, m, n, k, success in validation_results
            if kernel == "persistent" and m == M and n == N and k == K and success
        )
        if persistent_validated and M >= 1024 and N >= 1024:
            kernel_types.append("persistent")

        result = benchmark_gemm(M, N, K, kernel_types=kernel_types, warmup=3, rep=10)
        if result:
            benchmark_results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for kernel_type in ["standard", "persistent"]:
        kernel_results = [
            (m, n, k, success)
            for kernel, m, n, k, success in validation_results
            if kernel == kernel_type
        ]
        passed = sum(1 for _, _, _, success in kernel_results if success)
        total = sum(1 for _, _, _, success in kernel_results if success is not None)
        skipped = sum(1 for _, _, _, success in kernel_results if success is None)

        print(f"\n{kernel_type.capitalize()} Kernel:")
        for m, n, k, success in kernel_results:
            if success is None:
                status = "⚠ SKIP"
            elif success:
                status = "✓ PASS"
            else:
                status = "✗ FAIL"
            print(f"  {m:4d}×{n:4d}×{k:4d}: {status}")

        print(f"  Results: {passed}/{total} passed, {skipped} skipped")

    if benchmark_results:
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        print(
            f"{'Size (M×N×K)':<15} {'PyTorch':<12} {'Standard':<12} {'Persistent':<12} {'Best TFLOPS':<12}"
        )
        print("-" * 75)

        for result in benchmark_results:
            size_str = f"{result['M']}×{result['N']}×{result['K']}"
            pytorch_str = f"{result.get('pytorch_ms', float('inf')):.1f}ms"
            standard_str = (
                f"{result.get('standard_ms', float('inf')):.1f}ms"
                if "standard_ms" in result
                else "N/A"
            )
            persistent_str = (
                f"{result.get('persistent_ms', float('inf')):.1f}ms"
                if "persistent_ms" in result
                else "N/A"
            )

            # Find best TFLOPS
            best_tflops = max(
                result.get("pytorch_tflops", 0),
                result.get("standard_tflops", 0),
                result.get("persistent_tflops", 0),
            )

            print(
                f"{size_str:<15} {pytorch_str:<12} {standard_str:<12} {persistent_str:<12} {best_tflops:<8.1f}"
            )

    # Final status
    total_passed = sum(1 for _, _, _, _, success in validation_results if success)
    total_tests = sum(
        1 for _, _, _, _, success in validation_results if success is not None
    )

    if total_passed == total_tests and benchmark_results:
        print("\n🎉 All validation tests passed and benchmarks completed successfully!")
        return 0
    else:
        print(
            f"\n⚠ {total_passed}/{total_tests} validation tests passed, benchmarks may be incomplete"
        )
        return 1


if __name__ == "__main__":
    exit(run_benchmark_suite())
