# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

# Import the contiguous grouped GEMM implementation
# Make sure this file is in the same directory or in your Python path
from cg_forward import cg_grouped_gemm, CudaUtils

# Set of benchmark configurations from DeepSeek paper
CONTIGUOUS_CONFIGS = [
    # num_groups, m_per_group, n, k
    (4, 8192, 4096, 7168),
    (4, 8192, 7168, 2048),
    (8, 4096, 4096, 7168),
    (8, 4096, 7168, 2048),
]

MASKED_CONFIGS = [
    # num_groups, m_per_group, n, k
    (1, 1024, 4096, 7168),
    (1, 1024, 7168, 2048),
    (2, 512, 4096, 7168),
    (2, 512, 7168, 2048),
    (4, 256, 4096, 7168),
    (4, 256, 7168, 2048),
]


def format_size(size_bytes):
    """Format size in bytes to human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def calculate_metrics(
    num_groups: int, m_per_group: int, n: int, k: int, duration_ms: float
) -> Dict[str, float]:
    """
    Calculate performance metrics for a GEMM operation.

    Args:
        num_groups: Number of expert groups
        m_per_group: M dimension per group
        n: N dimension
        k: K dimension
        duration_ms: Duration in milliseconds

    Returns:
        Dictionary with performance metrics
    """
    # Total number of operations (2*M*N*K for each group)
    m_total = num_groups * m_per_group
    flops = 2 * m_total * n * k  # Each GEMM requires 2*M*N*K FLOPs

    # Convert to TFLOPS
    tflops = flops / (duration_ms / 1000) / 1e12

    # Calculate memory bandwidth
    # For each group: inputs (M*K), weights (N*K), outputs (M*N)
    bytes_read = m_total * k * 2  # Inputs (bfloat16 = 2 bytes)
    bytes_read += num_groups * n * k * 2  # Weights (bfloat16 = 2 bytes)
    bytes_written = m_total * n * 2  # Outputs (bfloat16 = 2 bytes)
    total_bytes = bytes_read + bytes_written

    # Convert to GB/s
    gb_per_s = total_bytes / (duration_ms / 1000) / 1e9

    return {"TFLOPS": tflops, "GB/s": gb_per_s, "ms": duration_ms}


def pytorch_grouped_gemm_reference(inputs, expert_weights, expert_indices):
    """
    Reference implementation using pure PyTorch operations.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert

    Returns:
        Output tensor of shape [M_total, N]
    """
    M_total, K = inputs.shape
    num_experts, N, K_weights = expert_weights.shape

    # Create output tensor
    output = torch.zeros((M_total, N), dtype=inputs.dtype, device=inputs.device)

    # Process each token
    for i in range(M_total):
        expert_idx = expert_indices[i].item()
        # Get the expert weights
        weight = expert_weights[expert_idx]
        # Compute output for this token
        output[i] = torch.matmul(inputs[i], weight.T)

    return output


def run_benchmark_config(
    num_groups: int,
    m_per_group: int,
    n: int,
    k: int,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    # use_tma: bool = True,
    run_pytorch_ref: bool = True,
    verify_output: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run benchmark for a specific configuration.

    Args:
        num_groups: Number of expert groups
        m_per_group: M dimension per group
        n: N dimension
        k: K dimension
        warmup_iters: Number of warmup iterations
        benchmark_iters: Number of benchmark iterations
        use_tma: Whether to use TMA optimization if available
        run_pytorch_ref: Whether to run PyTorch reference implementation
        verify_output: Whether to verify output matches reference
        verbose: Whether to print progress information

    Returns:
        Dictionary with performance metrics
    """
    # Total M dimension
    m_total = num_groups * m_per_group

    # Create input tensors
    inputs = torch.randn((m_total, k), dtype=torch.bfloat16, device="cuda")
    expert_weights = torch.randn(
        (num_groups, n, k), dtype=torch.bfloat16, device="cuda"
    )

    # Create expert indices (each group of tokens assigned to corresponding expert)
    expert_indices = torch.zeros(m_total, dtype=torch.int32, device="cuda")

    # Assign tokens to experts in blocks
    for g in range(num_groups):
        start_idx = g * m_per_group
        end_idx = (g + 1) * m_per_group
        expert_indices[start_idx:end_idx] = g

    # Memory usage information
    if verbose:
        inputs_size = inputs.element_size() * inputs.nelement()
        weights_size = expert_weights.element_size() * expert_weights.nelement()
        indices_size = expert_indices.element_size() * expert_indices.nelement()
        output_size = inputs.shape[0] * n * 2  # bfloat16 = 2 bytes
        total_size = inputs_size + weights_size + indices_size + output_size

        print(f"Memory usage:")
        print(f"  Inputs:  {format_size(inputs_size)}")
        print(f"  Weights: {format_size(weights_size)}")
        print(f"  Indices: {format_size(indices_size)}")
        print(f"  Output:  {format_size(output_size)}")
        print(f"  Total:   {format_size(total_size)}")

    # Warmup Triton implementation
    if verbose:
        print(f"Running {warmup_iters} warmup iterations...")
    for _ in range(warmup_iters):
        output_triton = cg_grouped_gemm(
            inputs,
            expert_weights,
            expert_indices,  # use_tma=use_tma
        )
        torch.cuda.synchronize()

    # Benchmark Triton implementation
    if verbose:
        print(f"Running {benchmark_iters} Triton benchmark iterations...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Start timing
    start_event.record()

    for _ in range(benchmark_iters):
        output_triton = cg_grouped_gemm(
            inputs,
            expert_weights,
            expert_indices,  # use_tma=use_tma
        )

    # End timing
    end_event.record()
    torch.cuda.synchronize()

    # Calculate duration
    triton_duration_ms = start_event.elapsed_time(end_event) / benchmark_iters

    # Calculate metrics for Triton
    triton_metrics = calculate_metrics(
        num_groups, m_per_group, n, k, triton_duration_ms
    )

    # Initialize PyTorch metrics
    pytorch_metrics = None
    pytorch_duration_ms = float("inf")
    output_match = None

    # Run PyTorch reference if requested
    if run_pytorch_ref:
        # Skip PyTorch reference for very large matrices
        should_skip = False
        """if m_total * n * k > 1e9:  # More than ~1 billion elements
            if verbose:
                print(
                    f"Skipping PyTorch reference for large matrix (M={m_total}, N={n}, K={k})"
                )
            should_skip = True
        """

        if not should_skip:
            # Warmup PyTorch reference implementation (just once for small configs)
            if verbose:
                print(f"Running PyTorch reference warmup...")
            pytorch_output = pytorch_grouped_gemm_reference(
                inputs, expert_weights, expert_indices
            )
            torch.cuda.synchronize()

            # Benchmark PyTorch reference implementation
            if verbose:
                print(f"Running PyTorch reference benchmark...")

            # Choose fewer iterations for the slower reference implementation
            pytorch_iters = min(5, benchmark_iters)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Start timing
            start_event.record()

            for _ in range(pytorch_iters):
                pytorch_output = pytorch_grouped_gemm_reference(
                    inputs, expert_weights, expert_indices
                )

            # End timing
            end_event.record()
            torch.cuda.synchronize()

            # Calculate duration
            pytorch_duration_ms = start_event.elapsed_time(end_event) / pytorch_iters

            # Calculate metrics for PyTorch
            pytorch_metrics = calculate_metrics(
                num_groups, m_per_group, n, k, pytorch_duration_ms
            )

            # Verify outputs match
            if verify_output:
                # Compare first 1000 elements (or fewer if smaller) to speed up verification
                sample_size = min(1000, pytorch_output.numel())
                indices = torch.randperm(pytorch_output.numel())[:sample_size]

                pytorch_sample = pytorch_output.view(-1)[indices]
                triton_sample = output_triton.view(-1)[indices]

                # Check if outputs match within tolerance
                rtol, atol = 1e-2, 1e-2  # More permissive for bf16
                errors = torch.abs(pytorch_sample - triton_sample)
                max_error = errors.max().item()
                mean_error = errors.mean().item()

                # Check if error is within tolerance
                output_match = torch.allclose(
                    pytorch_sample, triton_sample, rtol=rtol, atol=atol
                )

                if verbose:
                    if output_match:
                        print(
                            f"Output verification: PASS (max error: {max_error:.6f}, mean error: {mean_error:.6f})"
                        )
                    else:
                        print(
                            f"Output verification: FAIL (max error: {max_error:.6f}, mean error: {mean_error:.6f})"
                        )

    # Calculate speedup if PyTorch reference was run
    speedup = pytorch_duration_ms / triton_duration_ms if pytorch_metrics else None

    # Return metrics
    result = {
        "triton_ms": triton_duration_ms,
        "triton_TFLOPS": triton_metrics["TFLOPS"],
        "triton_GB/s": triton_metrics["GB/s"],
    }

    if pytorch_metrics:
        result.update(
            {
                "pytorch_ms": pytorch_duration_ms,
                "pytorch_TFLOPS": pytorch_metrics["TFLOPS"],
                "pytorch_GB/s": pytorch_metrics["GB/s"],
                "speedup": speedup,
                "output_match": output_match,
            }
        )

    return result


def benchmark_all_configs(
    configs: List[Tuple[int, int, int, int]],
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    use_tma: bool = True,
    run_pytorch_ref: bool = True,
    verify_output: bool = True,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run benchmarks for all configurations.

    Args:
        configs: List of (num_groups, m_per_group, n, k) configurations
        warmup_iters: Number of warmup iterations
        benchmark_iters: Number of benchmark iterations
        use_tma: Whether to use TMA optimization if available
        run_pytorch_ref: Whether to run PyTorch reference implementation
        verify_output: Whether to verify output matches reference
        verbose: Whether to print progress information

    Returns:
        List of dictionaries with benchmark results
    """
    results = []

    for i, (num_groups, m_per_group, n, k) in enumerate(configs):
        if verbose:
            print(
                f"\nBenchmark {i+1}/{len(configs)}: groups={num_groups}, m_per_group={m_per_group}, n={n}, k={k}"
            )

        try:
            metrics = run_benchmark_config(
                num_groups=num_groups,
                m_per_group=m_per_group,
                n=n,
                k=k,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
                # use_tma=use_tma,
                run_pytorch_ref=run_pytorch_ref,
                verify_output=verify_output,
                verbose=verbose,
            )

            result = {
                "num_groups": num_groups,
                "m_per_group": m_per_group,
                "n": n,
                "k": k,
                **metrics,
            }

            results.append(result)

            if verbose:
                print(
                    f"  Triton performance: {metrics['triton_TFLOPS']:.0f} TFLOPS, {metrics['triton_GB/s']:.0f} GB/s, {metrics['triton_ms']:.2f} ms"
                )
                if "pytorch_ms" in metrics:
                    print(
                        f"  PyTorch performance: {metrics['pytorch_TFLOPS']:.0f} TFLOPS, {metrics['pytorch_GB/s']:.0f} GB/s, {metrics['pytorch_ms']:.2f} ms"
                    )
                    print(f"  Speedup: {metrics['speedup']:.2f}x")
                    print(
                        f"  Output match: {'Yes' if metrics['output_match'] else 'No'}"
                    )

        except Exception as e:
            print(
                f"Error benchmarking configuration: groups={num_groups}, m_per_group={m_per_group}, n={n}, k={k}"
            )
            print(f"Exception: {e}")
            import traceback

            traceback.print_exc()

    return results


def print_results_table(results: List[Dict], title: str, include_pytorch: bool = True):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print(f"{title}")
    print("=" * 100)

    # Check if we have PyTorch results
    has_pytorch = include_pytorch and "pytorch_ms" in results[0]

    # Print header
    if has_pytorch:
        print(
            f"{'#Groups':6} {'M/group':8} {'N':6} {'K':6} "
            f"{'Triton':25} {'PyTorch':25} {'Speedup':8} {'Match':6}"
        )
        print(
            f"{' ':6} {' ':8} {' ':6} {' ':6} "
            f"{'TFLOPS':8} {'GB/s':8} {'ms':6} {'TFLOPS':8} {'GB/s':8} {'ms':6} {' ':8} {' ':6}"
        )
        print("-" * 100)
    else:
        print(
            f"{'#Groups':6} {'M/group':8} {'N':6} {'K':6} {'TFLOPS':10} {'GB/s':10} {'ms':8}"
        )
        print("-" * 60)

    # Print results
    for result in results:
        if has_pytorch:
            speedup = result.get("speedup", "N/A")
            speedup_str = f"{speedup:.2f}x" if isinstance(speedup, float) else speedup
            output_match = result.get("output_match", None)
            match_str = (
                "Yes" if output_match else "No" if output_match is False else "N/A"
            )

            print(
                f"{result['num_groups']:6d} {result['m_per_group']:8d} {result['n']:6d} {result['k']:6d} "
                f"{result['triton_TFLOPS']:8.0f} {result['triton_GB/s']:8.0f} {result['triton_ms']:6.2f} "
                f"{result.get('pytorch_TFLOPS', 'N/A'):8} {result.get('pytorch_GB/s', 'N/A'):8} {result.get('pytorch_ms', 'N/A'):6} "
                f"{speedup_str:8} {match_str:6}"
            )
        else:
            print(
                f"{result['num_groups']:6d} {result['m_per_group']:8d} {result['n']:6d} {result['k']:6d} "
                f"{result['triton_TFLOPS']:10.0f} {result['triton_GB/s']:10.0f} {result['triton_ms']:8.2f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek shapes with contiguous grouped GEMM"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--use-tma", action="store_true", help="Use TMA optimization if available"
    )
    parser.add_argument(
        "--contiguous-only",
        action="store_true",
        help="Only run contiguous layout benchmarks",
    )
    parser.add_argument(
        "--masked-only", action="store_true", help="Only run masked layout benchmarks"
    )
    parser.add_argument(
        "--no-pytorch",
        action="store_true",
        help="Skip PyTorch reference implementation",
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip output verification"
    )
    args = parser.parse_args()

    # Check if running on a supported GPU
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a CUDA-capable GPU.")
        return

    device_name = torch.cuda.get_device_name()
    on_hopper = True  # CudaUtilsis_hopper_gpu()
    num_sms = 132  # CudaUtils.get_num_sms()

    print(f"Running on: {device_name}")
    print(f"Hopper architecture: {'Yes' if on_hopper else 'No'}")
    print(f"Number of SMs: {num_sms}")
    print(f"Using TMA: {'Yes' if args.use_tma and on_hopper else 'No'}")
    print(f"Running PyTorch reference: {'No' if args.no_pytorch else 'Yes'}")
    print(f"Verifying outputs: {'No' if args.no_verify else 'Yes'}")

    # Run benchmarks
    all_results = []

    # Run contiguous layout benchmarks
    if not args.masked_only:
        contiguous_results = benchmark_all_configs(
            CONTIGUOUS_CONFIGS,
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
            use_tma=args.use_tma,
            run_pytorch_ref=True,  # not args.no_pytorch,
            verify_output=True,  # not args.no_verify,
            verbose=True,
        )
        print_results_table(
            contiguous_results,
            "Contiguous Layout Results",
            include_pytorch=not args.no_pytorch,
        )
        all_results.extend(contiguous_results)

    # Run masked layout benchmarks
    if not args.contiguous_only:
        masked_results = benchmark_all_configs(
            MASKED_CONFIGS,
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
            use_tma=args.use_tma,
            run_pytorch_ref=True,  # not args.no_pytorch,
            verify_output=True,  # not args.no_verify,
            verbose=True,
        )
        print_results_table(
            masked_results, "Masked Layout Results", include_pytorch=not args.no_pytorch
        )
        all_results.extend(masked_results)

    # Print combined results
    if len(all_results) > 0:
        avg_triton_tflops = sum(r["triton_TFLOPS"] for r in all_results) / len(
            all_results
        )
        avg_triton_gbps = sum(r["triton_GB/s"] for r in all_results) / len(all_results)
        print("\nAverage Triton Performance:")
        print(f"  TFLOPS: {avg_triton_tflops:.0f}")
        print(f"  GB/s:   {avg_triton_gbps:.0f}")

        # Calculate PyTorch averages if available
        if not args.no_pytorch and all("pytorch_ms" in r for r in all_results):
            avg_pytorch_tflops = sum(r["pytorch_TFLOPS"] for r in all_results) / len(
                all_results
            )
            avg_pytorch_gbps = sum(r["pytorch_GB/s"] for r in all_results) / len(
                all_results
            )
            avg_speedup = sum(r["speedup"] for r in all_results) / len(all_results)
            print("\nAverage PyTorch Performance:")
            print(f"  TFLOPS: {avg_pytorch_tflops:.0f}")
            print(f"  GB/s:   {avg_pytorch_gbps:.0f}")
            print(f"\nAverage Speedup: {avg_speedup:.2f}x")

            # Count output matches
            matches = sum(1 for r in all_results if r.get("output_match", False))
            print(f"Output Match: {matches}/{len(all_results)} configurations")


if __name__ == "__main__":
    main()
