#!/usr/bin/env python3
"""
Benchmark script to compare CutlassGemmManager vs torch.mm performance.

This script tests various matrix sizes and configurations to evaluate:
1. Performance differences between CUTLASS and PyTorch GEMM
2. Correctness of CUTLASS implementation
3. Memory usage patterns
4. Overhead analysis for different matrix sizes
"""

import argparse
import statistics
import time
import warnings
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

# Import the CUTLASS GEMM manager
from torchtitan.components.gemm_utils import CUTLASS_AVAILABLE, CutlassGemmManager


torch.backends.cuda.matmul.allow_tf32 = True


def create_test_matrices(
    m: int, n: int, k: int, dtype: torch.dtype, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create test matrices with random values."""
    # Use a fixed seed for reproducible results
    torch.manual_seed(42)

    A = torch.randn(m, k, dtype=dtype, device=device)
    B = torch.randn(k, n, dtype=dtype, device=device)

    return A, B


def benchmark_torch_mm(
    A: torch.Tensor, B: torch.Tensor, num_warmup: int = 10, num_runs: int = 100
) -> Dict[str, Any]:
    """Benchmark torch.mm performance."""
    device = A.device

    # Warmup
    for _ in range(num_warmup):
        _ = torch.mm(A, B)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []

    for _ in range(num_runs):
        iter_start = time.perf_counter()
        result = torch.mm(A, B)
        if device.type == "cuda":
            torch.cuda.synchronize()
        iter_end = time.perf_counter()
        times.append(iter_end - iter_start)

    return {
        "result": result,
        "times": times,
        "mean_time": statistics.mean(times),
        "median_time": statistics.median(times),
        "std_time": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_time": min(times),
        "max_time": max(times),
    }


def benchmark_cutlass_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    gemm_manager: CutlassGemmManager,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Benchmark CUTLASS GEMM performance."""
    device = A.device

    # Warmup
    for _ in range(num_warmup):
        try:
            _ = gemm_manager.gemm(A, B)
        except Exception as e:
            return {"error": f"CUTLASS warmup failed: {e}"}

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    result = None

    for _ in range(num_runs):
        try:
            iter_start = time.perf_counter()
            result = gemm_manager.gemm(A, B)
            if device.type == "cuda":
                torch.cuda.synchronize()
            iter_end = time.perf_counter()
            times.append(iter_end - iter_start)
        except Exception as e:
            return {"error": f"CUTLASS execution failed: {e}"}

    return {
        "result": result,
        "times": times,
        "mean_time": statistics.mean(times),
        "median_time": statistics.median(times),
        "std_time": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_time": min(times),
        "max_time": max(times),
    }


def check_correctness(
    torch_result: torch.Tensor,
    cutlass_result: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """Check if CUTLASS and PyTorch results are numerically equivalent."""
    if torch_result.shape != cutlass_result.shape:
        return {
            "correct": False,
            "error": f"Shape mismatch: torch={torch_result.shape}, cutlass={cutlass_result.shape}",
        }

    try:
        is_close = torch.allclose(torch_result, cutlass_result, rtol=rtol, atol=atol)
        max_abs_diff = torch.max(torch.abs(torch_result - cutlass_result)).item()
        max_rel_diff = torch.max(
            torch.abs((torch_result - cutlass_result) / (torch_result + 1e-8))
        ).item()

        return {
            "correct": is_close,
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "rtol": rtol,
            "atol": atol,
        }
    except Exception as e:
        return {"correct": False, "error": f"Correctness check failed: {e}"}


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    elif seconds >= 1e-3:
        return f"{seconds*1e3:.3f}ms"
    else:
        return f"{seconds*1e6:.3f}μs"


def calculate_gflops(m: int, n: int, k: int, time_seconds: float) -> float:
    """Calculate GFLOPS for matrix multiplication."""
    # Matrix multiplication requires 2*m*n*k operations (multiply-add)
    ops = 2 * m * n * k
    return ops / (time_seconds * 1e9)


def run_benchmark_suite(
    matrix_sizes: List[Tuple[int, int, int]],
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    num_warmup: int = 10,
    num_runs: int = 100,
) -> None:
    """Run comprehensive benchmark suite."""

    print(f"Running GEMM Benchmark Suite")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Warmup runs: {num_warmup}")
    print(f"Benchmark runs: {num_runs}")
    print(f"CUTLASS Available: {CUTLASS_AVAILABLE}")
    print("=" * 80)

    if not CUTLASS_AVAILABLE:
        print("CUTLASS not available. Only running PyTorch benchmarks.")
        print("=" * 80)

    # Initialize CUTLASS manager if available
    cutlass_manager = None
    if CUTLASS_AVAILABLE:
        try:
            cutlass_manager = CutlassGemmManager(dtype=dtype, device=device)
            print("CUTLASS manager initialized successfully")
        except Exception as e:
            print(f"Failed to initialize CUTLASS manager: {e}")
            cutlass_manager = None

    results = []

    for i, (m, n, k) in enumerate(matrix_sizes):
        print(f"\nTest {i+1}/{len(matrix_sizes)}: Matrix sizes M={m}, N={n}, K={k}")
        print("-" * 60)

        # Create test matrices
        A, B = create_test_matrices(m, n, k, dtype, device)

        # Benchmark PyTorch
        print("Benchmarking torch.mm...")
        torch_stats = benchmark_torch_mm(A, B, num_warmup, num_runs)
        torch_gflops = calculate_gflops(m, n, k, torch_stats["mean_time"])

        print(
            f"  PyTorch - Mean: {format_time(torch_stats['mean_time'])}, "
            f"Median: {format_time(torch_stats['median_time'])}, "
            f"GFLOPS: {torch_gflops:.2f}"
        )

        # Benchmark CUTLASS if available
        cutlass_stats = None
        correctness = None

        if cutlass_manager is not None:
            print("Benchmarking CUTLASS GEMM...")
            cutlass_stats = benchmark_cutlass_gemm(
                A, B, cutlass_manager, num_warmup, num_runs
            )

            if "error" in cutlass_stats:
                print(f"  CUTLASS - ERROR: {cutlass_stats['error']}")
            else:
                cutlass_gflops = calculate_gflops(m, n, k, cutlass_stats["mean_time"])
                speedup = torch_stats["mean_time"] / cutlass_stats["mean_time"]

                print(
                    f"  CUTLASS - Mean: {format_time(cutlass_stats['mean_time'])}, "
                    f"Median: {format_time(cutlass_stats['median_time'])}, "
                    f"GFLOPS: {cutlass_gflops:.2f}"
                )
                print(f"  Speedup: {speedup:.2f}x")

                # Check correctness
                correctness = check_correctness(
                    torch_stats["result"], cutlass_stats["result"]
                )
                if correctness["correct"]:
                    print(
                        f"  ✓ Results match (max_abs_diff: {correctness['max_abs_diff']:.2e})"
                    )
                else:
                    print(
                        f"  ✗ Results don't match: {correctness.get('error', 'Numerical mismatch')}"
                    )
                    if "max_abs_diff" in correctness:
                        print(f"    Max abs diff: {correctness['max_abs_diff']:.2e}")
                        print(f"    Max rel diff: {correctness['max_rel_diff']:.2e}")

        # Store results
        result_entry = {
            "matrix_size": (m, n, k),
            "torch_stats": torch_stats,
            "cutlass_stats": cutlass_stats,
            "correctness": correctness,
        }
        results.append(result_entry)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(
        f"{'Matrix Size':<20} {'PyTorch (ms)':<15} {'CUTLASS (ms)':<15} {'Speedup':<10} {'Correct':<10}"
    )
    print("-" * 80)

    for result in results:
        m, n, k = result["matrix_size"]
        size_str = f"{m}x{n}x{k}"

        torch_time = result["torch_stats"]["mean_time"] * 1000  # Convert to ms

        if result["cutlass_stats"] and "error" not in result["cutlass_stats"]:
            cutlass_time = result["cutlass_stats"]["mean_time"] * 1000  # Convert to ms
            speedup = torch_time / cutlass_time
            correct = (
                "✓"
                if result["correctness"] and result["correctness"]["correct"]
                else "✗"
            )
            print(
                f"{size_str:<20} {torch_time:<15.3f} {cutlass_time:<15.3f} {speedup:<10.2f} {correct:<10}"
            )
        else:
            error_msg = (
                "ERROR"
                if result["cutlass_stats"] and "error" in result["cutlass_stats"]
                else "N/A"
            )
            print(
                f"{size_str:<20} {torch_time:<15.3f} {error_msg:<15} {'N/A':<10} {'N/A':<10}"
            )


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUTLASS GEMM vs PyTorch")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda/cpu)")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for matrices",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    parser.add_argument(
        "--runs", type=int, default=100, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        help="Custom matrix sizes as triplets: M N K M N K ...",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark with fewer sizes"
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Define matrix sizes to test
    if args.sizes:
        if len(args.sizes) % 3 != 0:
            raise ValueError("Matrix sizes must be provided as triplets (M N K)")
        matrix_sizes = [
            (args.sizes[i], args.sizes[i + 1], args.sizes[i + 2])
            for i in range(0, len(args.sizes), 3)
        ]
    elif args.quick:
        # Quick test with smaller sizes
        matrix_sizes = [
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
        ]
    else:
        # Comprehensive test suite
        matrix_sizes = [
            # Small matrices (should fallback to PyTorch)
            (64, 64, 64),
            (96, 96, 96),
            # Medium matrices (CUTLASS should engage)
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            # Large matrices
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            # Non-square matrices
            (1024, 512, 256),
            (512, 1024, 256),
            (256, 512, 1024),
            # Common neural network sizes
            (4096, 11008, 4096),  # Llama FFN
            (4096, 4096, 4096),  # Attention
            (8192, 8192, 8192),  # Larger model
        ]

    run_benchmark_suite(
        matrix_sizes=matrix_sizes,
        dtype=dtype,
        device=args.device,
        num_warmup=args.warmup,
        num_runs=args.runs,
    )


if __name__ == "__main__":
    main()
