#!/usr/bin/env python3
"""
CUTE Grouped GEMM Benchmark Driver

A comprehensive benchmarking suite for the NVIDIA CUTLASS CUTE Python DSL
grouped GEMM implementation on Blackwell SM100 architecture.

This driver provides:
- Easy integration with PyTorch workflows
- Performance comparisons with PyTorch baselines
- Comprehensive timing and throughput metrics
- Memory usage monitoring
- Configurable test case generation
- Detailed performance reporting

Usage:
    python cute_grouped_gemm_benchmark.py --help
    python cute_grouped_gemm_benchmark.py --preset small
    python cute_grouped_gemm_benchmark.py --custom-config config.yaml
"""

import argparse
import functools
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn.functional as F
import yaml

# Memory profiling
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Import the original grouped GEMM implementation
# Assuming the original code is available as a module
try:
    import cuda.bindings.driver as cuda
    from grouped_gemm import (
        cute,
        cutlass,
        cutlass_torch,
        from_dlpack,
        GroupedGemmKernel,
        run_grouped_gemm,
        utils,
    )

    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    print("Warning: CUTLASS CUTE not available. Some features will be disabled.")


@dataclass
class BenchmarkConfig:
    """Configuration for grouped GEMM benchmarks."""

    # Problem configuration
    num_groups: int = 4
    problem_sizes: List[Tuple[int, int, int, int]] = None  # (M, N, K, L) for each group

    # Data types
    ab_dtype: str = "Float16"
    c_dtype: str = "Float16"
    acc_dtype: str = "Float32"

    # Matrix layouts
    a_major: str = "k"  # 'k' or 'm'
    b_major: str = "k"  # 'k' or 'n'
    c_major: str = "n"  # 'n' or 'm'

    # Architecture configuration
    mma_tiler_mn: Tuple[int, int] = (128, 64)
    cluster_shape_mn: Tuple[int, int] = (1, 1)
    use_2cta_instrs: bool = False
    tensormap_update_mode: str = "SMEM"  # "SMEM" or "GMEM"

    # Benchmark configuration
    warmup_iterations: int = 5
    benchmark_iterations: int = 20
    tolerance: float = 1e-2
    skip_validation: bool = False

    # Comparison configuration
    compare_pytorch: bool = True
    compare_naive: bool = True

    def __post_init__(self):
        if self.problem_sizes is None:
            # Default problem sizes for 4 groups
            self.problem_sizes = [
                (1024, 1024, 512, 1),
                (512, 2048, 1024, 1),
                (2048, 512, 256, 1),
                (256, 256, 1024, 1),
            ]


class PerformanceMetrics:
    """Container for performance metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.times = []
        self.memory_usage = []
        self.throughput = []
        self.flops = []

    def add_measurement(self, time_ms: float, memory_mb: float = 0, flops: float = 0):
        self.times.append(time_ms)
        self.memory_usage.append(memory_mb)
        if flops > 0:
            self.flops.append(flops)
            self.throughput.append(flops / (time_ms / 1000))  # FLOPS

    def get_stats(self) -> Dict[str, float]:
        if not self.times:
            return {}

        times_np = np.array(self.times)
        stats = {
            "mean_time_ms": float(np.mean(times_np)),
            "std_time_ms": float(np.std(times_np)),
            "min_time_ms": float(np.min(times_np)),
            "max_time_ms": float(np.max(times_np)),
            "median_time_ms": float(np.median(times_np)),
        }

        if self.memory_usage and any(m > 0 for m in self.memory_usage):
            mem_np = np.array(self.memory_usage)
            stats.update(
                {
                    "mean_memory_mb": float(np.mean(mem_np)),
                    "max_memory_mb": float(np.max(mem_np)),
                }
            )

        if self.throughput:
            throughput_np = np.array(self.throughput)
            stats.update(
                {
                    "mean_throughput_gflops": float(np.mean(throughput_np) / 1e9),
                    "peak_throughput_gflops": float(np.max(throughput_np) / 1e9),
                }
            )

        return stats


class CUTEGroupedGemmBenchmark:
    """Main benchmark class for CUTE Grouped GEMM."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this benchmark")

        self.logger = self._setup_logging()
        self.results = {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(__name__)

    def _calculate_flops(self, problem_sizes: List[Tuple[int, int, int, int]]) -> float:
        """Calculate total FLOPs for grouped GEMM."""
        total_flops = 0
        for m, n, k, l in problem_sizes:
            # Each GEMM: 2 * M * N * K operations (multiply-add)
            total_flops += 2 * m * n * k * l
        return total_flops

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def _create_test_tensors(
        self, problem_sizes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Create test tensors for each group."""
        tensors = []

        ab_dtype_torch = getattr(torch, self.config.ab_dtype.lower())
        c_dtype_torch = getattr(torch, self.config.c_dtype.lower())

        for m, n, k, l in problem_sizes:
            # Create tensors with proper layouts
            if self.config.a_major == "m":
                A = torch.randn(l, m, k, dtype=ab_dtype_torch, device=self.device)
            else:
                A = torch.randn(l, k, m, dtype=ab_dtype_torch, device=self.device)

            if self.config.b_major == "n":
                B = torch.randn(l, n, k, dtype=ab_dtype_torch, device=self.device)
            else:
                B = torch.randn(l, k, n, dtype=ab_dtype_torch, device=self.device)

            if self.config.c_major == "m":
                C = torch.zeros(l, m, n, dtype=c_dtype_torch, device=self.device)
            else:
                C = torch.zeros(l, n, m, dtype=c_dtype_torch, device=self.device)

            tensors.append((A, B, C))

        return tensors

    def benchmark_pytorch_baseline(
        self, test_tensors: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> PerformanceMetrics:
        """Benchmark PyTorch baseline implementation."""
        self.logger.info("Benchmarking PyTorch baseline...")

        metrics = PerformanceMetrics()
        flops = self._calculate_flops(self.config.problem_sizes)

        # Warmup
        for _ in range(self.config.warmup_iterations):
            for A, B, C in test_tensors:
                if self.config.a_major == "m" and self.config.b_major == "k":
                    torch.bmm(A, B.transpose(-2, -1))
                elif self.config.a_major == "k" and self.config.b_major == "n":
                    torch.bmm(A.transpose(-2, -1), B)
                else:
                    # Handle other layout combinations
                    torch.einsum("lmk,lkn->lmn", A, B)
            torch.cuda.synchronize()

        # Benchmark
        for i in range(self.config.benchmark_iterations):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            for A, B, C in test_tensors:
                if self.config.a_major == "m" and self.config.b_major == "k":
                    result = torch.bmm(A, B.transpose(-2, -1))
                elif self.config.a_major == "k" and self.config.b_major == "n":
                    result = torch.bmm(A.transpose(-2, -1), B)
                else:
                    result = torch.einsum("lmk,lkn->lmn", A, B)

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            time_ms = (end_time - start_time) * 1000
            memory_mb = self._get_memory_usage()
            metrics.add_measurement(time_ms, memory_mb, flops)

        return metrics

    def benchmark_cute_grouped_gemm(
        self, test_tensors: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> PerformanceMetrics:
        """Benchmark CUTE Grouped GEMM implementation."""
        if not HAS_CUTLASS:
            self.logger.warning("CUTLASS not available, skipping CUTE benchmark")
            return PerformanceMetrics()

        self.logger.info("Benchmarking CUTE Grouped GEMM...")

        metrics = PerformanceMetrics()
        flops = self._calculate_flops(self.config.problem_sizes)

        try:
            # Convert string dtypes to cutlass types
            ab_dtype = getattr(cutlass, self.config.ab_dtype)
            c_dtype = getattr(cutlass, self.config.c_dtype)
            acc_dtype = getattr(cutlass, self.config.acc_dtype)

            # Convert tensormap update mode
            if self.config.tensormap_update_mode == "SMEM":
                tensormap_mode = utils.TensorMapUpdateMode.SMEM
            else:
                tensormap_mode = utils.TensorMapUpdateMode.GMEM

            # Benchmark using the original run_grouped_gemm function
            for i in range(self.config.benchmark_iterations):
                torch.cuda.synchronize()
                start_time = time.perf_counter()

                try:
                    run_grouped_gemm(
                        num_groups=self.config.num_groups,
                        problem_sizes_mnkl=self.config.problem_sizes,
                        ab_dtype=ab_dtype,
                        c_dtype=c_dtype,
                        acc_dtype=acc_dtype,
                        a_major=self.config.a_major,
                        b_major=self.config.b_major,
                        c_major=self.config.c_major,
                        mma_tiler_mn=self.config.mma_tiler_mn,
                        cluster_shape_mn=self.config.cluster_shape_mn,
                        use_2cta_instrs=self.config.use_2cta_instrs,
                        tensormap_update_mode=tensormap_mode,
                        tolerance=self.config.tolerance,
                        warmup_iterations=0 if i > 0 else self.config.warmup_iterations,
                        iterations=1,
                        skip_ref_check=True,
                    )
                except Exception as e:
                    self.logger.error(f"CUTE benchmark failed: {e}")
                    break

                torch.cuda.synchronize()
                end_time = time.perf_counter()

                time_ms = (end_time - start_time) * 1000
                memory_mb = self._get_memory_usage()
                metrics.add_measurement(time_ms, memory_mb, flops)

        except Exception as e:
            self.logger.error(f"Error in CUTE benchmark setup: {e}")

        return metrics

    def run_validation(
        self, test_tensors: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> bool:
        """Validate CUTE results against PyTorch baseline."""
        if self.config.skip_validation or not HAS_CUTLASS:
            return True

        self.logger.info("Running validation...")

        try:
            # Run PyTorch baseline
            pytorch_results = []
            for A, B, C in test_tensors:
                if self.config.a_major == "m" and self.config.b_major == "k":
                    result = torch.bmm(A, B.transpose(-2, -1))
                else:
                    result = torch.einsum("lmk,lkn->lmn", A, B)
                pytorch_results.append(result.cpu())

            # Run CUTE implementation with validation enabled
            ab_dtype = getattr(cutlass, self.config.ab_dtype)
            c_dtype = getattr(cutlass, self.config.c_dtype)
            acc_dtype = getattr(cutlass, self.config.acc_dtype)

            if self.config.tensormap_update_mode == "SMEM":
                tensormap_mode = utils.TensorMapUpdateMode.SMEM
            else:
                tensormap_mode = utils.TensorMapUpdateMode.GMEM

            try:
                run_grouped_gemm(
                    num_groups=self.config.num_groups,
                    problem_sizes_mnkl=self.config.problem_sizes,
                    ab_dtype=ab_dtype,
                    c_dtype=c_dtype,
                    acc_dtype=acc_dtype,
                    a_major=self.config.a_major,
                    b_major=self.config.b_major,
                    c_major=self.config.c_major,
                    mma_tiler_mn=self.config.mma_tiler_mn,
                    cluster_shape_mn=self.config.cluster_shape_mn,
                    use_2cta_instrs=self.config.use_2cta_instrs,
                    tensormap_update_mode=tensormap_mode,
                    tolerance=self.config.tolerance,
                    warmup_iterations=1,
                    iterations=1,
                    skip_ref_check=False,
                )
                self.logger.info("Validation passed!")
                return True
            except Exception as e:
                self.logger.error(f"Validation failed: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Validation setup failed: {e}")
            return False

    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        self.logger.info(f"Starting benchmark with {self.config.num_groups} groups")
        self.logger.info(f"Problem sizes: {self.config.problem_sizes}")

        # Create test tensors
        test_tensors = self._create_test_tensors(self.config.problem_sizes)
        total_flops = self._calculate_flops(self.config.problem_sizes)

        self.logger.info(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOP")

        results = {
            "config": asdict(self.config),
            "total_flops": total_flops,
            "benchmarks": {},
        }

        # Run validation if enabled
        if not self.config.skip_validation:
            validation_passed = self.run_validation(test_tensors)
            results["validation_passed"] = validation_passed
            if not validation_passed:
                self.logger.warning("Validation failed, but continuing with benchmarks")

        # Benchmark PyTorch baseline
        if self.config.compare_pytorch:
            pytorch_metrics = self.benchmark_pytorch_baseline(test_tensors)
            results["benchmarks"]["pytorch"] = pytorch_metrics.get_stats()

        # Benchmark CUTE implementation
        if HAS_CUTLASS:
            cute_metrics = self.benchmark_cute_grouped_gemm(test_tensors)
            results["benchmarks"]["cute"] = cute_metrics.get_stats()

        self.results = results
        return results

    def save_results(self, filepath: str):
        """Save benchmark results to file."""
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Results saved to {filepath}")

    def print_summary(self):
        """Print a summary of benchmark results."""
        if not self.results:
            self.logger.warning("No results to summarize")
            return

        print("\n" + "=" * 80)
        print("CUTE GROUPED GEMM BENCHMARK SUMMARY")
        print("=" * 80)

        print(f"\nConfiguration:")
        print(f"  Groups: {self.config.num_groups}")
        print(f"  Problem sizes: {self.config.problem_sizes}")
        print(
            f"  Data types: AB={self.config.ab_dtype}, C={self.config.c_dtype}, Acc={self.config.acc_dtype}"
        )
        print(f"  MMA tiler: {self.config.mma_tiler_mn}")
        print(f"  Cluster shape: {self.config.cluster_shape_mn}")
        print(f"  Total FLOPs: {self.results['total_flops'] / 1e9:.2f} GFLOP")

        print(f"\nBenchmark Results:")
        print(
            f"{'Implementation':<15} {'Time (ms)':<12} {'Throughput (GFLOP/s)':<20} {'Memory (MB)':<12}"
        )
        print("-" * 65)

        for impl_name, stats in self.results["benchmarks"].items():
            if stats:
                time_str = f"{stats.get('mean_time_ms', 0):.2f} ± {stats.get('std_time_ms', 0):.2f}"
                throughput_str = f"{stats.get('mean_throughput_gflops', 0):.2f}"
                memory_str = f"{stats.get('mean_memory_mb', 0):.1f}"
                print(
                    f"{impl_name.upper():<15} {time_str:<12} {throughput_str:<20} {memory_str:<12}"
                )

        # Calculate speedup
        if (
            "pytorch" in self.results["benchmarks"]
            and "cute" in self.results["benchmarks"]
        ):
            pytorch_time = self.results["benchmarks"]["pytorch"].get("mean_time_ms", 0)
            cute_time = self.results["benchmarks"]["cute"].get("mean_time_ms", 0)
            if pytorch_time > 0 and cute_time > 0:
                speedup = pytorch_time / cute_time
                print(f"\nSpeedup: {speedup:.2f}x (CUTE vs PyTorch)")

        print("=" * 80)


def create_preset_configs() -> Dict[str, BenchmarkConfig]:
    """Create preset benchmark configurations."""
    presets = {}

    # Small test configuration
    presets["small"] = BenchmarkConfig(
        num_groups=2,
        problem_sizes=[(128, 128, 128, 1), (256, 256, 256, 1)],
        warmup_iterations=2,
        benchmark_iterations=5,
    )

    # Medium configuration
    presets["medium"] = BenchmarkConfig(
        num_groups=4,
        problem_sizes=[
            (512, 512, 512, 1),
            (1024, 512, 256, 1),
            (256, 1024, 512, 1),
            (512, 256, 1024, 1),
        ],
        warmup_iterations=5,
        benchmark_iterations=10,
    )

    # Large configuration
    presets["large"] = BenchmarkConfig(
        num_groups=8,
        problem_sizes=[
            (1024, 1024, 512, 1),
            (2048, 1024, 256, 1),
            (1024, 2048, 512, 1),
            (512, 512, 1024, 1),
            (1536, 768, 384, 1),
            (768, 1536, 768, 1),
            (512, 2048, 256, 1),
            (2048, 512, 512, 1),
        ],
        warmup_iterations=10,
        benchmark_iterations=20,
    )

    return presets


def load_config_from_file(filepath: str) -> BenchmarkConfig:
    """Load configuration from YAML file."""
    with open(filepath, "r") as f:
        config_dict = yaml.safe_load(f)

    return BenchmarkConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(
        description="CUTE Grouped GEMM Benchmark Driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run small preset
  python cute_grouped_gemm_benchmark.py --preset small

  # Run with custom configuration
  python cute_grouped_gemm_benchmark.py --custom-config my_config.yaml

  # Run large benchmark and save results
  python cute_grouped_gemm_benchmark.py --preset large --output results.json

  # Quick test without validation
  python cute_grouped_gemm_benchmark.py --preset small --skip-validation
        """,
    )

    # Configuration options
    parser.add_argument(
        "--preset",
        choices=["small", "medium", "large"],
        help="Use a preset configuration",
    )
    parser.add_argument(
        "--custom-config", type=str, help="Path to custom YAML configuration file"
    )

    # Override options
    parser.add_argument("--num-groups", type=int, help="Number of groups")
    parser.add_argument(
        "--warmup-iterations", type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--benchmark-iterations", type=int, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip validation"
    )
    parser.add_argument(
        "--skip-pytorch", action="store_true", help="Skip PyTorch baseline"
    )

    # Output options
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")

    args = parser.parse_args()

    # Determine configuration
    if args.custom_config:
        config = load_config_from_file(args.custom_config)
    elif args.preset:
        presets = create_preset_configs()
        config = presets[args.preset]
    else:
        config = BenchmarkConfig()  # Default configuration

    # Apply command line overrides
    if args.num_groups:
        config.num_groups = args.num_groups
    if args.warmup_iterations:
        config.warmup_iterations = args.warmup_iterations
    if args.benchmark_iterations:
        config.benchmark_iterations = args.benchmark_iterations
    if args.skip_validation:
        config.skip_validation = True
    if args.skip_pytorch:
        config.compare_pytorch = False

    # Run benchmark
    benchmark = CUTEGroupedGemmBenchmark(config)

    try:
        results = benchmark.run_benchmark()

        if not args.quiet:
            benchmark.print_summary()

        if args.output:
            benchmark.save_results(args.output)

    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
