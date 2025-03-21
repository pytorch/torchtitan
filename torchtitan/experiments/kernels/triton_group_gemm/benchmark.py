#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Benchmark comparing reference PyTorch vs optimized NG group GEMM implementation

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

# Try to import the optimized implementations
try:
    from NG_backward import grouped_gemm_backward
    from NG_forward import grouped_gemm_forward
except ImportError:
    logging.error(
        "Error importing grouped GEMM modules. Make sure the implementation files are in the correct path."
    )
    raise


def compute_reference_forward(x, w, n_sizes):
    """
    Reference PyTorch implementation of N*G grouped GEMM forward pass.

    Args:
        x (torch.Tensor): Input tensor of shape (M, K)
        w (torch.Tensor): Weight tensor of shape (N, K)
        n_sizes (torch.Tensor): Group sizes tensor of shape (G)

    Returns:
        torch.Tensor: Output tensor of shape (M, N)
    """
    result = torch.zeros((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)

    n_start = 0
    for g in range(len(n_sizes)):
        n_size = n_sizes[g].item()
        if n_size > 0:
            n_end = n_start + n_size

            # Extract group weights
            w_g = w[n_start:n_end]

            # Compute group output
            y_g = torch.matmul(x, w_g.T)

            # Store result
            result[:, n_start:n_end] = y_g

            # Update start index
            n_start = n_end

    return result


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],  # We'll vary the batch size (M dimension)
        x_vals=[1024, 2048, 4096, 8192, 16384],  # Different batch sizes to test
        line_arg="provider",  # We'll compare different providers
        line_vals=["pytorch_reference", "optimized_kernel"],
        line_names=["PyTorch Reference", "Optimized Kernel"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="TFLOPS",  # We'll measure TFLOPS
        plot_name="ng_grouped_gemm_comparison",
        args={
            "K": 7168,  # Hidden dimension, fixed for all tests
            "N": 4096,  # Output dimension, fixed for all tests
            "G": 4,  # Number of groups
            "dtype": torch.float16,
            "device": "cuda",
        },
    )
)
def benchmark_forward(M, K, N, G, provider, dtype=torch.float16, device="cuda"):
    """
    Benchmark the forward pass of the grouped GEMM implementation.

    Args:
        M (int): Batch size dimension
        K (int): Hidden dimension
        N (int): Output dimension
        G (int): Number of groups
        provider (str): Provider to use ('pytorch_reference' or 'optimized_kernel')
        dtype (torch.dtype): Data type to use
        device (str): Device to use

    Returns:
        float: Performance in TFLOPS
    """
    # Create group sizes for N dimension (balanced across groups)
    base_size = N // G
    remainder = N % G
    N_sizes = [base_size + (1 if i < remainder else 0) for i in range(G)]
    n_sizes = torch.tensor(N_sizes, device=device, dtype=torch.int32)

    # Create input and weight tensors
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)

    # Pre-compute for PyTorch reference to ensure fair comparison
    if provider == "pytorch_reference":
        # Warmup
        torch.cuda.synchronize()
        compute_reference_forward(x, w, n_sizes)
        torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        for _ in range(10):  # Average over 10 runs
            compute_reference_forward(x, w, n_sizes)
        torch.cuda.synchronize()
        end_time = time.time()
    else:  # Optimized kernel
        # Warmup
        torch.cuda.synchronize()
        grouped_gemm_forward(x, w, n_sizes)
        torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        for _ in range(10):  # Average over 10 runs
            grouped_gemm_forward(x, w, n_sizes)
        torch.cuda.synchronize()
        end_time = time.time()

    # Calculate FLOPs
    # For GEMM: 2 * M * N * K FLOPs (multiply-add counts as 2 FLOPs)
    flops = 2 * M * N * K

    # Convert to TFLOPS (tera-FLOPS)
    avg_time = (end_time - start_time) / 10  # Average time per run
    tflops = flops / avg_time / 1e12

    return tflops


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["G"],  # We'll vary the number of groups
        x_vals=[1, 2, 4, 8, 16],  # Different numbers of groups to test
        line_arg="provider",  # We'll compare different providers
        line_vals=["pytorch_reference", "optimized_kernel"],
        line_names=["PyTorch Reference", "Optimized Kernel"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="TFLOPS",  # We'll measure TFLOPS
        plot_name="ng_grouped_gemm_group_scaling",
        args={
            "M": 8192,  # Batch dimension, fixed for all tests
            "K": 4096,  # Hidden dimension, fixed for all tests
            "N": 8192,  # Output dimension, fixed for all tests
            "dtype": torch.float16,
            "device": "cuda",
        },
    )
)
def benchmark_forward_groups(M, K, N, G, provider, dtype=torch.float16, device="cuda"):
    """
    Benchmark how performance scales with number of groups.

    Args:
        M (int): Batch size dimension
        K (int): Hidden dimension
        N (int): Output dimension
        G (int): Number of groups
        provider (str): Provider to use ('pytorch_reference' or 'optimized_kernel')
        dtype (torch.dtype): Data type to use
        device (str): Device to use

    Returns:
        float: Performance in TFLOPS
    """
    # Create group sizes for N dimension (balanced across groups)
    base_size = N // G
    remainder = N % G
    N_sizes = [base_size + (1 if i < remainder else 0) for i in range(G)]
    n_sizes = torch.tensor(N_sizes, device=device, dtype=torch.int32)

    # Create input and weight tensors
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)

    # Benchmark logic - same as previous function
    if provider == "pytorch_reference":
        torch.cuda.synchronize()
        compute_reference_forward(x, w, n_sizes)
        torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(10):
            compute_reference_forward(x, w, n_sizes)
        torch.cuda.synchronize()
        end_time = time.time()
    else:
        torch.cuda.synchronize()
        grouped_gemm_forward(x, w, n_sizes)
        torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(10):
            grouped_gemm_forward(x, w, n_sizes)
        torch.cuda.synchronize()
        end_time = time.time()

    # Calculate FLOPs and TFLOPS
    flops = 2 * M * N * K
    avg_time = (end_time - start_time) / 10
    tflops = flops / avg_time / 1e12

    return tflops


def benchmark_model_configs():
    """
    Benchmark common model configurations used in DeepSeek-like models.
    """
    # Model configurations: (M, K, N, G)
    configs = [
        (8192, 7168, 4096, 4),  # Config 1
        (8192, 2048, 7168, 4),  # Config 2
        (4096, 7168, 4096, 8),  # Config 3
        (4096, 2048, 7168, 8),  # Config 4
    ]

    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    for config_idx, (M, K, N, G) in enumerate(configs):
        logging.info(f"\n===== Benchmarking DeepSeek Config {config_idx+1} =====")
        logging.info(f"M={M}, K={K}, N={N}, G={G}")

        # Create group sizes for N dimension
        base_size = N // G
        remainder = N % G
        N_sizes = [base_size + (1 if i < remainder else 0) for i in range(G)]
        n_sizes = torch.tensor(N_sizes, device=device, dtype=torch.int32)

        # Create tensors
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)

        # Benchmark PyTorch reference
        torch.cuda.synchronize()
        compute_reference_forward(x, w, n_sizes)  # Warmup
        torch.cuda.synchronize()

        logging.info("Benchmarking PyTorch reference...")
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        for _ in range(10):
            compute_reference_forward(x, w, n_sizes)
        torch.cuda.synchronize()
        end_time = time.time()
        pt_time = (end_time - start_time) / 10
        pt_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

        # Benchmark optimized kernel
        torch.cuda.synchronize()
        grouped_gemm_forward(x, w, n_sizes)  # Warmup
        torch.cuda.synchronize()

        logging.info("Benchmarking optimized kernel...")
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        for _ in range(10):
            grouped_gemm_forward(x, w, n_sizes)
        torch.cuda.synchronize()
        end_time = time.time()
        opt_time = (end_time - start_time) / 10
        opt_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

        # Calculate FLOPs and speedup
        flops = 2 * M * N * K
        pt_tflops = flops / pt_time / 1e12
        opt_tflops = flops / opt_time / 1e12
        speedup = pt_time / opt_time

        # Store results
        results.append(
            {
                "config": f"Config {config_idx+1}",
                "dimensions": f"M={M}, K={K}, N={N}, G={G}",
                "pt_time_ms": pt_time * 1000,
                "opt_time_ms": opt_time * 1000,
                "pt_tflops": pt_tflops,
                "opt_tflops": opt_tflops,
                "speedup": speedup,
                "pt_memory_mb": pt_memory,
                "opt_memory_mb": opt_memory,
                "memory_savings": (
                    (pt_memory - opt_memory) / pt_memory * 100 if pt_memory > 0 else 0
                ),
            }
        )

        logging.info(
            f"PyTorch Reference: {pt_time*1000:.2f} ms, {pt_tflops:.2f} TFLOPS, {pt_memory:.2f} MB"
        )
        logging.info(
            f"Optimized Kernel: {opt_time*1000:.2f} ms, {opt_tflops:.2f} TFLOPS, {opt_memory:.2f} MB"
        )
        logging.info(
            f"Speedup: {speedup:.2f}x, Memory savings: {results[-1]['memory_savings']:.2f}%"
        )

    # Print summary table
    logging.info("\n===== Benchmark Results Summary =====")
    logging.info(
        f"{'Config':<10} | {'Time (ms)':<20} | {'TFLOPS':<20} | {'Speedup':<10} | {'Memory (MB)':<20} | {'Memory Saved':<12}"
    )
    logging.info(
        f"{'':<10} | {'PyTorch':<9} {'Kernel':<9} | {'PyTorch':<9} {'Kernel':<9} | {'':<10} | {'PyTorch':<9} {'Kernel':<9} | {'':<12}"
    )
    logging.info("-" * 100)

    for result in results:
        logging.info(
            f"{result['config']:<10} | "
            f"{result['pt_time_ms']:<9.2f} {result['opt_time_ms']:<9.2f} | "
            f"{result['pt_tflops']:<9.2f} {result['opt_tflops']:<9.2f} | "
            f"{result['speedup']:<10.2f} | "
            f"{result['pt_memory_mb']:<9.2f} {result['opt_memory_mb']:<9.2f} | "
            f"{result['memory_savings']:<12.2f}%"
        )

    return results


def plot_benchmark_results(results):
    """
    Plot benchmark results as bar charts.
    """
    # Extract data
    configs = [r["config"] for r in results]
    pt_tflops = [r["pt_tflops"] for r in results]
    opt_tflops = [r["opt_tflops"] for r in results]
    speedups = [r["speedup"] for r in results]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot TFLOPS comparison
    x = np.arange(len(configs))
    width = 0.35
    ax1.bar(x - width / 2, pt_tflops, width, label="PyTorch Reference")
    ax1.bar(x + width / 2, opt_tflops, width, label="Optimized Kernel")
    ax1.set_xlabel("Model Configuration")
    ax1.set_ylabel("TFLOPS")
    ax1.set_title("Performance Comparison (Higher is Better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot speedup
    ax2.bar(x, speedups, width=0.6, color="green")
    ax2.set_xlabel("Model Configuration")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("Speedup Factor (Higher is Better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Add speedup values on top of bars
    for i, v in enumerate(speedups):
        ax2.text(i, v + 0.1, f"{v:.2f}x", ha="center")

    plt.tight_layout()
    plt.savefig("ng_grouped_gemm_benchmark_results.png")
    logging.info(
        "Benchmark results plot saved to 'ng_grouped_gemm_benchmark_results.png'"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark NG Grouped GEMM implementations"
    )
    parser.add_argument("--run-all", action="store_true", help="Run all benchmarks")
    parser.add_argument(
        "--triton-bench", action="store_true", help="Run Triton performance reports"
    )
    parser.add_argument(
        "--model-configs", action="store_true", help="Benchmark model configurations"
    )
    args = parser.parse_args()

    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.error(
            "CUDA is not available. This benchmark requires a CUDA-capable GPU."
        )
        exit(1)

    if args.run_all or args.model_configs:
        # Benchmark model configurations
        logging.info("Running benchmark for model configurations...")
        results = benchmark_model_configs()
        plot_benchmark_results(results)

    if args.run_all or args.triton_bench:
        # Run Triton performance reports
        logging.info("Running Triton performance reports...")
        benchmark_forward.run(save_path="ng_grouped_gemm_benchmark_results")
        benchmark_forward_groups.run(save_path="ng_grouped_gemm_benchmark_results")
        logging.info(
            "Triton performance reports saved to 'ng_grouped_gemm_benchmark_results' directory"
        )
