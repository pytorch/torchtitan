import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import triton

# Import our implementation
# Assuming the implementation is in aligned_cggemm_robust.py
from cg_forward import cg_grouped_gemm_forward
from cg_reference import pytorch_reference


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run"""

    num_experts: int
    num_groups: int
    m_per_group: int
    n: int
    k: int
    layout: str = "contiguous"  # "contiguous" or "masked"


def create_benchmark_data(
    config: BenchmarkConfig,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create test data for benchmarking.

    Args:
        config: Benchmark configuration
        device: Device to create tensors on
        dtype: Data type for inputs and weights

    Returns:
        Tuple of (inputs, expert_weights, expert_indices)
    """
    # Calculate total tokens
    M_total = config.num_groups * config.m_per_group

    # Create input tensor
    inputs = torch.randn((M_total, config.k), dtype=dtype, device=device)

    # Create expert weights
    expert_weights = torch.randn(
        (config.num_experts, config.n, config.k), dtype=dtype, device=device
    )

    # Create expert indices based on layout
    expert_indices = torch.zeros(M_total, dtype=torch.int32, device=device)

    if config.layout == "contiguous":
        # For contiguous layout, assign experts in contiguous blocks
        for group_idx in range(config.num_groups):
            start_idx = group_idx * config.m_per_group
            end_idx = (group_idx + 1) * config.m_per_group

            # Assign an expert to this group (cycling through available experts)
            expert_idx = group_idx % config.num_experts
            expert_indices[start_idx:end_idx] = expert_idx
    else:
        # For masked layout (assuming random assignment)
        # This is just a placeholder - adjust based on your actual masking strategy
        for i in range(M_total):
            expert_indices[i] = i % config.num_experts

    return inputs, expert_weights, expert_indices


def benchmark_kernel(
    config: BenchmarkConfig,
    dtype: torch.dtype = torch.float16,
    num_warmup: int = 10,
    num_runs: int = 50,
    reference_timing: bool = True,
) -> Dict:
    """
    Benchmark our kernel implementation against the reference.

    Args:
        config: Benchmark configuration
        dtype: Data type for inputs and weights
        num_warmup: Number of warmup runs
        num_runs: Number of timing runs
        reference_timing: Whether to time the reference implementation

    Returns:
        Dictionary with benchmark results
    """
    torch.cuda.empty_cache()
    device = "cuda"

    # Create benchmark data
    inputs, expert_weights, expert_indices = create_benchmark_data(
        config, device=device, dtype=dtype
    )

    # Warm up
    for _ in range(num_warmup):
        output_triton = cg_grouped_gemm_forward(
            inputs, expert_weights, expert_indices, group_size_m=config.m_per_group
        )
        if reference_timing:
            output_ref = pytorch_reference(
                inputs, expert_weights, expert_indices, group_size_m=config.m_per_group
            )
        torch.cuda.synchronize()

    # Benchmark our implementation
    triton_times = []
    torch.cuda.synchronize()

    for _ in range(num_runs):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        output_triton = cg_grouped_gemm_forward(
            inputs, expert_weights, expert_indices, group_size_m=config.m_per_group
        )
        end_time.record()

        torch.cuda.synchronize()
        triton_times.append(start_time.elapsed_time(end_time))

    # Benchmark reference implementation
    ref_times = []
    if reference_timing:
        torch.cuda.synchronize()

        for _ in range(num_runs):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            output_ref = pytorch_reference(
                inputs, expert_weights, expert_indices, group_size_m=config.m_per_group
            )
            end_time.record()

            torch.cuda.synchronize()
            ref_times.append(start_time.elapsed_time(end_time))

    # Calculate statistics
    triton_avg_ms = np.mean(triton_times)
    triton_std_ms = np.std(triton_times)

    # Calculate TFLOPS and bandwidth
    M_total = config.num_groups * config.m_per_group
    flops = 2 * M_total * config.n * config.k  # multiply-add operations
    tflops = flops / (triton_avg_ms / 1000) / 1e12

    # Memory bandwidth calculation
    # Read inputs: M_total * K
    # Read weights: num_experts * N * K (worst case, could be less with caching)
    # Write outputs: M_total * N
    bytes_read = (
        M_total * config.k + config.num_experts * config.n * config.k
    ) * inputs.element_size()
    bytes_written = M_total * config.n * inputs.element_size()
    total_bytes = bytes_read + bytes_written
    gbps = total_bytes / (triton_avg_ms / 1000) / 1e9

    # Calculate speedup
    speedup = 1.0
    if reference_timing:
        ref_avg_ms = np.mean(ref_times)
        ref_std_ms = np.std(ref_times)
        speedup = ref_avg_ms / triton_avg_ms

    # Prepare results
    results = {
        "config": config,
        "triton_time_ms": triton_avg_ms,
        "triton_std_ms": triton_std_ms,
        "tflops": tflops,
        "gbps": gbps,
        "speedup": speedup,
    }

    if reference_timing:
        results.update(
            {
                "ref_time_ms": ref_avg_ms,
                "ref_std_ms": ref_std_ms,
            }
        )

    # Verify correctness
    if reference_timing:
        is_close = torch.allclose(output_triton, output_ref, rtol=1e-2, atol=1e-2)
        results["is_correct"] = is_close

        if not is_close:
            abs_diff = torch.abs(output_triton - output_ref)
            max_diff = torch.max(abs_diff).item()
            results["max_diff"] = max_diff

    return results


def get_paper_configs() -> List[BenchmarkConfig]:
    """Get the benchmark configurations from the paper tables"""
    configs = []

    # Contiguous layout configurations
    configs.append(
        BenchmarkConfig(
            num_experts=4,
            num_groups=8192,
            m_per_group=4096,
            n=7168,
            k=4096,
            layout="contiguous",
        )
    )
    """configs.append(
        BenchmarkConfig(
            num_experts=4,
            num_groups=8192,
            m_per_group=7168,
            n=20480,
            k=4096,
            layout="contiguous",
        )
    )
    configs.append(
        BenchmarkConfig(
            num_experts=8,
            num_groups=4096,
            m_per_group=4096,
            n=7168,
            k=4096,
            layout="contiguous",
        )
    )
    configs.append(
        BenchmarkConfig(
            num_experts=8,
            num_groups=4096,
            m_per_group=7168,
            n=20480,
            k=4096,
            layout="contiguous",
        )
    )

    # Masked layout configurations - uncomment if needed
    configs.append(
        BenchmarkConfig(
            num_experts=1,
            num_groups=1024,
            m_per_group=4096,
            n=7168,
            k=4096,
            layout="masked",
        )
    )
    configs.append(
        BenchmarkConfig(
            num_experts=1,
            num_groups=1024,
            m_per_group=7168,
            n=20480,
            k=4096,
            layout="masked",
        )
    )
    configs.append(
        BenchmarkConfig(
            num_experts=2,
            num_groups=512,
            m_per_group=4096,
            n=7168,
            k=4096,
            layout="masked",
        )
    )
    configs.append(
        BenchmarkConfig(
            num_experts=2,
            num_groups=512,
            m_per_group=7168,
            n=20480,
            k=4096,
            layout="masked",
        )
    )
    configs.append(
        BenchmarkConfig(
            num_experts=4,
            num_groups=256,
            m_per_group=4096,
            n=7168,
            k=4096,
            layout="masked",
        )
    )
    configs.append(
        BenchmarkConfig(
            num_experts=4,
            num_groups=256,
            m_per_group=7168,
            n=20480,
            k=4096,
            layout="masked",
        )
    """
    # )

    return configs


def print_benchmark_results(results_list):
    """Pretty print benchmark results"""
    print("\n===== Contiguous Grouped GEMM Benchmark Results =====\n")

    contiguous_results = [r for r in results_list if r["config"].layout == "contiguous"]
    masked_results = [r for r in results_list if r["config"].layout == "masked"]

    if contiguous_results:
        print("Grouped GEMMs for MoE models (contiguous layout)")
        print(
            "#Experts\t#Groups\tM/Group\tN\tK\tComputation\tMemory bandwidth\tSpeedup"
        )
        for r in contiguous_results:
            cfg = r["config"]
            print(
                f"{cfg.num_experts}\t{cfg.num_groups}\t{cfg.m_per_group}\t{cfg.n}\t{cfg.k}\t{int(r['tflops'])} TFLOPS\t{int(r['gbps'])} GB/s\t{r['speedup']:.1f}x"
            )
        print()

    if masked_results:
        print("Grouped GEMMs for MoE models (masked layout)")
        print(
            "#Experts\t#Groups\tM/Group\tN\tK\tComputation\tMemory bandwidth\tSpeedup"
        )
        for r in masked_results:
            cfg = r["config"]
            print(
                f"{cfg.num_experts}\t{cfg.num_groups}\t{cfg.m_per_group}\t{cfg.n}\t{cfg.k}\t{int(r['tflops'])} TFLOPS\t{int(r['gbps'])} GB/s\t{r['speedup']:.1f}x"
            )
        print()


def run_scaling_benchmark():
    """Run benchmark with varying expert counts to test scaling"""
    print("\nRunning scaling benchmark...")

    base_config = BenchmarkConfig(
        num_experts=4,
        num_groups=256,
        m_per_group=1024,
        n=4096,
        k=4096,
        layout="contiguous",
    )

    expert_counts = [1, 2, 4, 8, 16, 32]
    results = []

    for num_experts in expert_counts:
        config = BenchmarkConfig(
            num_experts=num_experts,
            num_groups=base_config.num_groups,
            m_per_group=base_config.m_per_group,
            n=base_config.n,
            k=base_config.k,
            layout=base_config.layout,
        )

        print(f"Benchmarking {num_experts} experts...")
        result = benchmark_kernel(config, num_warmup=5, num_runs=10)
        results.append(result)

    print("\n===== Expert Count Scaling Results =====")
    print("Experts\tTime (ms)\tTFLOPS\tGB/s\tSpeedup")
    for r in results:
        print(
            f"{r['config'].num_experts}\t{r['triton_time_ms']:.2f}\t{r['tflops']:.0f}\t{r['gbps']:.0f}\t{r['speedup']:.2f}"
        )


def run_memory_size_benchmark():
    """Run benchmark with varying matrix sizes to test memory scaling"""
    print("\nRunning memory size scaling benchmark...")

    base_config = BenchmarkConfig(
        num_experts=4,
        num_groups=64,
        m_per_group=1024,
        n=4096,
        k=4096,
        layout="contiguous",
    )

    size_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
    results = []

    for multiplier in size_multipliers:
        # Scale the N and K dimensions
        n = int(base_config.n * multiplier)
        k = int(base_config.k * multiplier)

        config = BenchmarkConfig(
            num_experts=base_config.num_experts,
            num_groups=base_config.num_groups,
            m_per_group=base_config.m_per_group,
            n=n,
            k=k,
            layout=base_config.layout,
        )

        print(f"Benchmarking size multiplier {multiplier}x (N={n}, K={k})...")
        result = benchmark_kernel(config, num_warmup=5, num_runs=10)
        results.append(result)

    print("\n===== Memory Size Scaling Results =====")
    print("Size\tN\tK\tTime (ms)\tTFLOPS\tGB/s\tSpeedup")
    for i, r in enumerate(results):
        print(
            f"{size_multipliers[i]}x\t{r['config'].n}\t{r['config'].k}\t{r['triton_time_ms']:.2f}\t{r['tflops']:.0f}\t{r['gbps']:.0f}\t{r['speedup']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Contiguous Grouped GEMM")
    parser.add_argument(
        "--paper", action="store_true", help="Run the paper configurations benchmark"
    )
    parser.add_argument(
        "--scaling", action="store_true", help="Run the expert count scaling benchmark"
    )
    parser.add_argument(
        "--memory", action="store_true", help="Run the memory size scaling benchmark"
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument("--runs", type=int, default=20, help="Number of timing runs")
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Run reference implementation for comparison",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Cannot run benchmarks.")
        return

    # Print GPU info
    device_name = torch.cuda.get_device_name(0)
    print(f"Running on: {device_name}")

    if args.paper or args.all:
        configs = get_paper_configs()
        results = []

        for i, config in enumerate(configs):
            print(f"\nBenchmarking configuration {i+1}/{len(configs)}:")
            print(
                f"  #Experts: {config.num_experts}, #Groups: {config.num_groups}, M/Group: {config.m_per_group}"
            )
            print(f"  N: {config.n}, K: {config.k}, Layout: {config.layout}")

            try:
                result = benchmark_kernel(
                    config,
                    num_warmup=args.warmup,
                    num_runs=args.runs,
                    reference_timing=args.reference,
                )
                results.append(result)

                print(f"  Time: {result['triton_time_ms']:.2f} ms")
                print(
                    f"  Performance: {result['tflops']:.0f} TFLOPS, {result['gbps']:.0f} GB/s"
                )
                if args.reference:
                    print(f"  Speedup: {result['speedup']:.2f}x")
                    print(f"  Correctness: {result['is_correct']}")
            except Exception as e:
                print(f"  Error benchmarking this configuration: {e}")

        # Print final results table
        print_benchmark_results(results)

    if args.scaling or args.all:
        run_scaling_benchmark()

    if args.memory or args.all:
        run_memory_size_benchmark()


if __name__ == "__main__":
    main()
