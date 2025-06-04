#!/usr/bin/env python3
"""
Grouped GEMM Benchmark using Triton's do_bench

Compares CUTLASS GroupedGemmKernel against PyTorch manual looping
with robust timing measurements and various problem size configurations.
"""

import time
from typing import Any, Dict, List, Tuple

import torch
import triton.testing

# Import CUTLASS components
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils
    from cute_grouped_gemm import GroupedGemmKernel
    from cutlass.cute.runtime import from_dlpack

    HAS_CUTLASS = True
    print("✓ CUTLASS and GroupedGemmKernel imported successfully")
except ImportError as e:
    HAS_CUTLASS = False
    print(f"✗ CUTLASS import failed: {e}")
    print("Make sure CUTLASS and cute_grouped_gemm.py are available")
    exit(1)


class GroupedGemmBenchmark:
    """Wrapper class for CUTLASS GroupedGemmKernel benchmarking."""

    def __init__(self, problem_sizes: List[Tuple[int, int, int, int]]):
        """
        Initialize grouped GEMM benchmark.

        Args:
            problem_sizes: List of (M, N, K, L) tuples defining each GEMM problem
        """
        self.problem_sizes = problem_sizes
        self.num_groups = len(problem_sizes)
        self.device = torch.device("cuda")
        self.dtype_torch = torch.float16
        self.dtype_cutlass = cutlass.Float16

        print(f"Setting up grouped GEMM with {self.num_groups} groups:")
        for i, (M, N, K, L) in enumerate(problem_sizes):
            print(f"  Group {i}: C[{M},{N}] = A[{M},{K}] @ B[{K},{N}]")

        # Setup tensors and kernel
        self._setup_tensors()
        self._setup_kernel()

    def _create_tensor_with_strides(self, M, N, K):
        """Create PyTorch tensors and extract their actual strides."""
        # Create standard PyTorch tensors (row-major by default)
        A = torch.randn(M, K, dtype=self.dtype_torch, device=self.device)
        B = torch.randn(K, N, dtype=self.dtype_torch, device=self.device)
        C = torch.zeros(M, N, dtype=self.dtype_torch, device=self.device)

        # Convert to MNKL format
        A_mnkl = A.unsqueeze(-1).contiguous()  # (M, K) -> (M, K, 1)
        B_mnkl = B.transpose(0, 1).unsqueeze(-1).contiguous()  # (K, N) -> (N, K, 1)
        C_mnkl = C.unsqueeze(-1).contiguous()  # (M, N) -> (M, N, 1)

        # Create CUTE tensors
        A_cute = from_dlpack(A_mnkl, assumed_align=16)
        B_cute = from_dlpack(B_mnkl, assumed_align=16)
        C_cute = from_dlpack(C_mnkl, assumed_align=16)

        # Set CUTE properties
        A_cute.element_type = self.dtype_cutlass
        B_cute.element_type = self.dtype_cutlass
        C_cute.element_type = self.dtype_cutlass

        # Mark layouts as dynamic
        A_cute = A_cute.mark_layout_dynamic(leading_dim=1)
        B_cute = B_cute.mark_layout_dynamic(leading_dim=1)
        C_cute = C_cute.mark_layout_dynamic(leading_dim=1)

        # Extract 2D strides
        A_strides = A_mnkl.stride()[:2]
        B_strides = B_mnkl.stride()[:2]
        C_strides = C_mnkl.stride()[:2]

        return (
            (A, B, C),
            (A_cute, B_cute, C_cute),
            (A_strides, B_strides, C_strides),
            A_mnkl.data_ptr(),
            B_mnkl.data_ptr(),
            C_mnkl.data_ptr(),
        )

    def _setup_tensors(self):
        """Setup all tensors and metadata for grouped GEMM."""
        self.torch_tensors = []
        self.cute_tensors = []
        strides = []
        pointers = []

        # Create tensors for each group
        for M, N, K, L in self.problem_sizes:
            torch_abc, cute_abc, stride_abc, ptr_a, ptr_b, ptr_c = (
                self._create_tensor_with_strides(M, N, K)
            )

            self.torch_tensors.append(torch_abc)
            self.cute_tensors.append(cute_abc)
            strides.append(stride_abc)
            pointers.append([ptr_a, ptr_b, ptr_c])

        # Convert metadata to tensors
        problem_sizes_tensor = torch.tensor(
            self.problem_sizes, dtype=torch.int32, device=self.device
        )
        self.problem_sizes_cute = from_dlpack(problem_sizes_tensor, assumed_align=16)

        strides_tensor = torch.tensor(strides, dtype=torch.int32, device=self.device)
        self.strides_cute = from_dlpack(strides_tensor, assumed_align=16)

        pointers_tensor = torch.tensor(pointers, dtype=torch.int64, device=self.device)
        self.pointers_cute = from_dlpack(pointers_tensor, assumed_align=16)

        # Create tensormap buffer
        hardware_info = utils.HardwareInfo()
        sm_count = hardware_info.get_max_active_clusters(1)

        tensormap_tensor = torch.zeros(
            (sm_count, 3, 128 // 8),
            dtype=torch.int64,
            device=self.device,
        )
        self.tensormap_cute = from_dlpack(tensormap_tensor, assumed_align=16)

    def _setup_kernel(self):
        """Setup and compile the grouped GEMM kernel."""
        # Create grouped GEMM kernel
        self.grouped_gemm = GroupedGemmKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=False,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
            tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
        )

        # Compute grid parameters
        def compute_total_clusters():
            cta_tile_m = 128
            cta_tile_n = 64
            cluster_m = 1
            cluster_n = 1

            cluster_tile_m = cta_tile_m * cluster_m
            cluster_tile_n = cta_tile_n * cluster_n

            total = 0
            for M, N, K, L in self.problem_sizes:
                clusters_m = (M + cluster_tile_m - 1) // cluster_tile_m
                clusters_n = (N + cluster_tile_n - 1) // cluster_tile_n
                total += clusters_m * clusters_n
            return total

        self.total_clusters = compute_total_clusters()

        hardware_info = utils.HardwareInfo()
        self.max_active_clusters = hardware_info.get_max_active_clusters(1)

        # Choose initial tensors (smallest ones for tensormap initialization)
        sizes = [(M * K, N * K, M * N) for M, N, K, L in self.problem_sizes]
        min_a_idx = min(range(self.num_groups), key=lambda i: sizes[i][0])
        min_b_idx = min(range(self.num_groups), key=lambda i: sizes[i][1])
        min_c_idx = min(range(self.num_groups), key=lambda i: sizes[i][2])

        self.initial_A = self.cute_tensors[min_a_idx][0]
        self.initial_B = self.cute_tensors[min_b_idx][1]
        self.initial_C = self.cute_tensors[min_c_idx][2]

        # Setup stream
        self.torch_stream = torch.cuda.Stream()
        self.stream = cuda.CUstream(self.torch_stream.cuda_stream)

        # Compile kernel
        self.compiled_kernel = cute.compile(
            self.grouped_gemm,
            self.initial_A,
            self.initial_B,
            self.initial_C,
            self.num_groups,
            self.problem_sizes_cute,
            self.strides_cute,
            self.pointers_cute,
            self.total_clusters,
            self.tensormap_cute,
            self.max_active_clusters,
            self.stream,
        )

    def pytorch_manual_loop(self):
        """Execute PyTorch manual looping through all GEMMs."""
        results = []
        for i, (A, B, C) in enumerate(self.torch_tensors):
            # Reset output tensor
            C.zero_()
            # Compute GEMM
            result = torch.mm(A, B, out=C)
            results.append(result)
        return results

    def cutlass_grouped_gemm(self):
        """Execute CUTLASS grouped GEMM kernel."""
        # Reset all output tensors
        for A, B, C in self.torch_tensors:
            C.zero_()

        # Execute grouped kernel
        self.compiled_kernel(
            self.initial_A,
            self.initial_B,
            self.initial_C,
            self.problem_sizes_cute,
            self.strides_cute,
            self.pointers_cute,
            self.tensormap_cute,
            self.stream,
        )

        # Return all C tensors
        return [C for A, B, C in self.torch_tensors]


def validate_grouped_gemm(
    problem_sizes: List[Tuple[int, int, int, int]], tolerance=1e-2
):
    """
    Validate that grouped GEMM produces correct results.

    Args:
        problem_sizes: List of (M, N, K, L) tuples
        tolerance: Acceptable relative error

    Returns:
        bool: True if all results match within tolerance
    """
    print(f"\nValidating grouped GEMM with {len(problem_sizes)} groups")

    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return False

    try:
        benchmark = GroupedGemmBenchmark(problem_sizes)

        # Compute both results
        pytorch_results = benchmark.pytorch_manual_loop()
        cutlass_results = benchmark.cutlass_grouped_gemm()

        # Compare results group by group
        all_correct = True
        for i, (pytorch_result, cutlass_result) in enumerate(
            zip(pytorch_results, cutlass_results)
        ):
            diff = torch.abs(pytorch_result - cutlass_result)
            max_diff = torch.max(diff).item()
            norm_diff = torch.norm(diff).item()
            norm_ref = torch.norm(pytorch_result).item()
            rel_error = norm_diff / norm_ref if norm_ref > 0 else float("inf")

            print(f"  Group {i}: max_diff={max_diff:.2e}, rel_error={rel_error:.2e}")

            if rel_error > tolerance:
                print(f"    ✗ Group {i} failed tolerance check")
                all_correct = False
            else:
                print(f"    ✓ Group {i} passed")

        if all_correct:
            print(f"  ✓ All groups passed validation")
        else:
            print(f"  ✗ Some groups failed validation")

        return all_correct

    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        return False


def benchmark_grouped_gemm(
    problem_sizes: List[Tuple[int, int, int, int]], warmup=3, rep=10
):
    """
    Benchmark grouped GEMM vs PyTorch manual looping.

    Args:
        problem_sizes: List of (M, N, K, L) tuples
        warmup: Number of warmup iterations
        rep: Number of benchmark repetitions

    Returns:
        dict: Timing results and metrics
    """
    print(f"\nBenchmarking grouped GEMM with {len(problem_sizes)} groups")

    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return None

    try:
        benchmark = GroupedGemmBenchmark(problem_sizes)

        # Calculate total FLOPs
        total_flops = sum(2 * M * N * K for M, N, K, L in problem_sizes)

        # Benchmark PyTorch manual loop
        pytorch_time = triton.testing.do_bench(
            benchmark.pytorch_manual_loop, warmup=warmup, rep=rep
        )

        # Benchmark CUTLASS grouped GEMM
        cutlass_time = triton.testing.do_bench(
            benchmark.cutlass_grouped_gemm, warmup=warmup, rep=rep
        )

        # Calculate metrics
        pytorch_tflops = total_flops / (pytorch_time * 1e-3) / 1e12
        cutlass_tflops = total_flops / (cutlass_time * 1e-3) / 1e12
        speedup = pytorch_time / cutlass_time

        results = {
            "num_groups": len(problem_sizes),
            "total_flops": total_flops,
            "pytorch_ms": pytorch_time,
            "cutlass_ms": cutlass_time,
            "pytorch_tflops": pytorch_tflops,
            "cutlass_tflops": cutlass_tflops,
            "speedup": speedup,
            "problem_sizes": problem_sizes,
        }

        print(
            f"  PyTorch manual loop: {pytorch_time:.2f} ms ({pytorch_tflops:.2f} TFLOPS)"
        )
        print(f"  CUTLASS grouped: {cutlass_time:.2f} ms ({cutlass_tflops:.2f} TFLOPS)")
        print(f"  Speedup: {speedup:.2f}x")

        return results

    except Exception as e:
        print(f"  ✗ Benchmark failed: {e}")
        return None


def generate_problem_sets():
    """Generate different sets of problem sizes for comprehensive testing."""

    problem_sets = {
        "small_uniform": [(256, 256, 256, 1) for _ in range(8)],
        "medium_uniform": [(512, 512, 512, 1) for _ in range(4)],
        "large_uniform": [(1024, 1024, 1024, 1) for _ in range(2)],
        "mixed_sizes": [
            (256, 256, 256, 1),
            (512, 512, 512, 1),
            (1024, 1024, 512, 1),
            (768, 384, 256, 1),
            (384, 768, 256, 1),
        ],
        "skinny_matrices": [
            (2048, 128, 256, 1),
            (1024, 256, 512, 1),
            (512, 512, 1024, 1),
            (256, 1024, 512, 1),
        ],
        "fat_matrices": [
            (128, 2048, 256, 1),
            (256, 1024, 512, 1),
            (512, 512, 1024, 1),
            (1024, 256, 512, 1),
        ],
        "many_small": [(128, 128, 128, 1) for _ in range(16)],
        "few_large": [
            (2048, 2048, 1024, 1),
            (1536, 1536, 768, 1),
        ],
    }

    return problem_sets


def run_grouped_gemm_benchmark_suite():
    """Run comprehensive grouped GEMM benchmark suite."""
    print("CUTLASS Grouped GEMM Benchmark Suite")
    print("Comparing GroupedGemmKernel vs PyTorch Manual Looping")
    print("Using Triton's do_bench for accurate timing")
    print("=" * 80)

    if not HAS_CUTLASS:
        print("✗ CUTLASS not available")
        return 1

    problem_sets = generate_problem_sets()

    # Validation phase
    print("\n" + "=" * 80)
    print("VALIDATION PHASE")
    print("=" * 80)

    validation_results = []
    tolerance = 1e-2

    for set_name, problem_sizes in problem_sets.items():
        print(f"\nValidating problem set: {set_name}")
        success = validate_grouped_gemm(problem_sizes, tolerance=tolerance)
        validation_results.append((set_name, success))

    # Benchmark phase
    print("\n" + "=" * 80)
    print("BENCHMARK PHASE")
    print("=" * 80)

    benchmark_results = []

    for set_name, problem_sizes in problem_sets.items():
        # Only benchmark if validation passed
        validation_success = next(
            success for name, success in validation_results if name == set_name
        )

        if validation_success:
            print(f"\nBenchmarking problem set: {set_name}")
            result = benchmark_grouped_gemm(problem_sizes, warmup=3, rep=10)
            if result:
                result["set_name"] = set_name
                benchmark_results.append(result)
        else:
            print(f"\nSkipping benchmark for {set_name} (validation failed)")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, success in validation_results if success)
    total = len(validation_results)

    for set_name, success in validation_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {set_name:<15}: {status}")

    print(f"\nValidation: {passed}/{total} problem sets passed")

    if benchmark_results:
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        print(
            f"{'Problem Set':<15} {'Groups':<8} {'PyTorch':<10} {'CUTLASS':<10} {'Speedup (x)':<8} {'Best TFLOPS':<10}"
        )
        print("-" * 80)

        total_speedup = 0
        max_tflops = 0

        for result in benchmark_results:
            set_name = result["set_name"]
            num_groups = result["num_groups"]
            pytorch_ms = result["pytorch_ms"]
            cutlass_ms = result["cutlass_ms"]
            speedup = result["speedup"]
            best_tflops = max(result["pytorch_tflops"], result["cutlass_tflops"])

            print(
                f"{set_name:<15} {num_groups:<8} {pytorch_ms:<8.3f} {cutlass_ms:<12.3f} "
                f"{speedup:<9.2f} {best_tflops:<11.2f}"
            )

            total_speedup += speedup
            max_tflops = max(max_tflops, best_tflops)

        avg_speedup = total_speedup / len(benchmark_results)
        print(f"\nAverage speedup: {avg_speedup:.2f}x")
        print(f"Peak performance: {max_tflops:.1f} TFLOPS")

        # Analysis
        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        # Find best and worst performing configurations
        best_speedup = max(benchmark_results, key=lambda x: x["speedup"])
        worst_speedup = min(benchmark_results, key=lambda x: x["speedup"])
        best_tflops_result = max(
            benchmark_results,
            key=lambda x: max(x["pytorch_tflops"], x["cutlass_tflops"]),
        )

        print(
            f"Best speedup: {best_speedup['speedup']:.2f}x ({best_speedup['set_name']})"
        )
        print(
            f"Worst speedup: {worst_speedup['speedup']:.2f}x ({worst_speedup['set_name']})"
        )
        print(
            f"Best TFLOPS: {max(best_tflops_result['pytorch_tflops'], best_tflops_result['cutlass_tflops']):.1f} ({best_tflops_result['set_name']})"
        )

        # Efficiency analysis
        efficient_sets = [r for r in benchmark_results if r["speedup"] > 1.5]
        print(
            f"\nHigh-efficiency problem sets (>1.5x speedup): {len(efficient_sets)}/{len(benchmark_results)}"
        )

        for result in efficient_sets:
            print(f"  {result['set_name']}: {result['speedup']:.2f}x speedup")

    # Final status
    if passed == total and benchmark_results:
        print("\n🎉 All validation tests passed and benchmarks completed successfully!")
        print(
            "Grouped GEMM shows significant benefits for batch processing multiple smaller GEMMs!"
        )
        return 0
    else:
        print(
            f"\n⚠ {passed}/{total} validation tests passed, benchmarks may be incomplete"
        )
        return 1


if __name__ == "__main__":
    exit(run_grouped_gemm_benchmark_suite())
