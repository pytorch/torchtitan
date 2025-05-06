import math
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Import the groupwise_activation_quant function from the provided module
from dsgemm_kernels import groupwise_activation_quant


def ceil_div(a, b):
    """Ceiling division function"""
    return math.ceil(a / b)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Efficient blockwise quantization for weights using tensor operations.

    Args:
        x (torch.Tensor): The input tensor to be quantized (2D)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Quantized tensor and scaling factors
    """
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


def dequantize_vectorized(
    quant_tensor: torch.Tensor, scale_factors: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantize a quantized tensor using the provided scale factors - vectorized implementation.

    Args:
        quant_tensor (torch.Tensor): The quantized tensor
        scale_factors (torch.Tensor): The scale factors used during quantization
        block_size (int): The block size used during quantization

    Returns:
        torch.Tensor: The dequantized tensor
    """
    # Get original dimensions
    orig_shape = quant_tensor.shape

    # Convert to float32
    dequant = quant_tensor.to(torch.float32)

    # For 2D tensors, use efficient vectorized approach
    if dequant.dim() == 2:
        # Calculate new dimensions with padding
        m, n = orig_shape
        padded_m = ceil_div(m, block_size) * block_size
        padded_n = ceil_div(n, block_size) * block_size

        # Pad if necessary (should match quantization padding)
        if padded_m > m or padded_n > n:
            padded = torch.zeros(
                (padded_m, padded_n), dtype=torch.float32, device=dequant.device
            )
            padded[:m, :n] = dequant
            dequant = padded

        # Reshape to block structure
        blocks_per_row = dequant.size(1) // block_size
        dequant_blocks = dequant.view(-1, block_size, blocks_per_row, block_size)

        # Reshape scale factors for broadcasting
        scale_blocks = scale_factors.view(
            scale_factors.size(0), 1, scale_factors.size(1), 1
        )

        # Apply scaling factors
        dequant_blocks = dequant_blocks * scale_blocks

        # Reshape back
        dequant = dequant_blocks.view(padded_m, padded_n)

        # Trim to original size
        dequant = dequant[:m, :n]
    else:
        # For handling non-2D tensors (fallback)
        raise ValueError(
            "Only 2D tensors are supported with the vectorized dequantization"
        )

    return dequant.reshape(orig_shape)


def benchmark_precision(
    matrix_sizes: List[Tuple[int, int]] = [(1024, 1024), (2048, 2048), (4096, 4096)],
    dtypes: List[torch.dtype] = [torch.float32, torch.float16],
    distributions: List[str] = ["uniform", "normal", "mixed"],
    block_size: int = 128,
    n_runs: int = 3,
    device: str = "cuda",
    visualize: bool = True,
) -> Dict:
    """
    Benchmark the precision of quantization and dequantization using vectorized implementation.

    Args:
        matrix_sizes: List of matrix sizes to test
        dtypes: List of data types to test
        distributions: List of data distributions to test
        block_size: Block size for quantization
        n_runs: Number of runs for timing
        device: Device to run on ("cuda" or "cpu")
        visualize: Whether to visualize the results

    Returns:
        Dict: Dictionary containing benchmark results
    """
    results = {
        "times": {},
        "errors": {},
        "memory": {},
    }

    # Test all combinations
    for size in matrix_sizes:
        for dtype in dtypes:
            for dist in distributions:
                key = f"{size[0]}x{size[1]}_{dtype.__str__().split('.')[-1]}_{dist}"
                print(f"\nTesting configuration: {key}")

                # Create test matrix with appropriate distribution
                if dist == "uniform":
                    matrix = (
                        torch.rand(size, dtype=dtype, device=device) * 2 - 1
                    )  # Range [-1, 1]
                elif dist == "normal":
                    matrix = torch.randn(size, dtype=dtype, device=device)
                elif dist == "mixed":
                    # Mix of small and large values
                    matrix = torch.randn(size, dtype=dtype, device=device)
                    # Add some large values (10x) at random positions
                    mask = torch.rand(size, device=device) > 0.95
                    matrix[mask] *= 10
                    # Add some very small values
                    mask = torch.rand(size, device=device) > 0.95
                    matrix[mask] *= 0.01
                else:
                    raise ValueError(f"Unknown distribution: {dist}")

                # Ensure matrix is 2D for vectorized operations
                if matrix.dim() != 2:
                    matrix = matrix.reshape(size)

                # Function to measure execution time
                def time_execution(func, *args, **kwargs):
                    torch.cuda.synchronize() if device == "cuda" else None
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    torch.cuda.synchronize() if device == "cuda" else None
                    end_time = time.time()
                    return result, end_time - start_time

                # Time the operations (averaged over n_runs)
                quant_time = 0
                dequant_time = 0

                # Warm-up run
                quantized, scales = per_block_cast_to_fp8(matrix)
                dequantized = dequantize_vectorized(quantized, scales, block_size)

                # Benchmark runs
                for _ in range(n_runs):
                    # Quantization timing
                    _, t_quant = time_execution(per_block_cast_to_fp8, matrix)
                    quant_time += t_quant

                    # Get new quantized values for dequantization
                    quantized, scales = per_block_cast_to_fp8(matrix)

                    # Dequantization timing
                    _, t_dequant = time_execution(
                        dequantize_vectorized, quantized, scales, block_size
                    )
                    dequant_time += t_dequant

                # Average times
                quant_time /= n_runs
                dequant_time /= n_runs

                # Record timing results
                results["times"][key] = {
                    "quantization": quant_time,
                    "dequantization": dequant_time,
                    "total": quant_time + dequant_time,
                }

                # Get final results for error analysis
                quantized, scales = per_block_cast_to_fp8(matrix)
                dequantized = dequantize_vectorized(quantized, scales, block_size)

                # Compute errors
                abs_error = torch.abs(matrix - dequantized)
                rel_error = abs_error / (
                    torch.abs(matrix) + 1e-8
                )  # Avoid division by zero

                # Calculate error metrics
                results["errors"][key] = {
                    "mean_abs_error": abs_error.mean().item(),
                    "max_abs_error": abs_error.max().item(),
                    "rmse": torch.sqrt(torch.mean(abs_error**2)).item(),
                    "mean_rel_error": rel_error.mean().item(),
                    "max_rel_error": rel_error.max().item(),
                    "psnr": 20
                    * torch.log10(
                        torch.max(torch.abs(matrix))
                        / torch.sqrt(torch.mean(abs_error**2))
                    ).item(),
                }

                # Calculate memory usage
                orig_bytes = matrix.nelement() * matrix.element_size()
                quant_bytes = (
                    quantized.nelement() * quantized.element_size()
                    + scales.nelement() * scales.element_size()
                )

                results["memory"][key] = {
                    "original_bytes": orig_bytes,
                    "quantized_bytes": quant_bytes,
                    "compression_ratio": orig_bytes / quant_bytes,
                }

                # Print summary
                print(f"  Quantization time: {quant_time:.6f}s")
                print(f"  Dequantization time: {dequant_time:.6f}s")
                print(f"  RMSE: {results['errors'][key]['rmse']:.6f}")
                print(f"  PSNR: {results['errors'][key]['psnr']:.2f} dB")
                print(
                    f"  Compression ratio: {results['memory'][key]['compression_ratio']:.2f}x"
                )

    # Visualize if requested
    if visualize:
        visualize_precision_results(results)

    return results


def visualize_precision_results(results: Dict) -> None:
    """
    Visualize the precision benchmark results.

    Args:
        results: Dictionary containing benchmark results
    """
    # Extract configurations
    configs = list(results["times"].keys())

    # Create figure with multiple subplots
    plt.figure(figsize=(16, 12))

    # 1. Timing comparison
    plt.subplot(2, 2, 1)
    quant_times = [results["times"][cfg]["quantization"] for cfg in configs]
    dequant_times = [results["times"][cfg]["dequantization"] for cfg in configs]

    x = np.arange(len(configs))
    width = 0.35

    plt.bar(x - width / 2, quant_times, width, label="Quantization")
    plt.bar(x + width / 2, dequant_times, width, label="Dequantization")

    plt.ylabel("Time (seconds)")
    plt.title("Execution Time by Configuration")
    plt.xticks(x, configs, rotation=90)
    plt.legend()

    # 2. RMSE comparison
    plt.subplot(2, 2, 2)
    rmse_values = [results["errors"][cfg]["rmse"] for cfg in configs]
    plt.bar(x, rmse_values)
    plt.ylabel("RMSE")
    plt.title("Reconstruction Error (RMSE) by Configuration")
    plt.xticks(x, configs, rotation=90)

    # 3. PSNR comparison
    plt.subplot(2, 2, 3)
    psnr_values = [results["errors"][cfg]["psnr"] for cfg in configs]
    plt.bar(x, psnr_values)
    plt.ylabel("PSNR (dB)")
    plt.title("Reconstruction Quality (PSNR) by Configuration")
    plt.xticks(x, configs, rotation=90)

    # 4. Compression ratio comparison
    plt.subplot(2, 2, 4)
    comp_ratios = [results["memory"][cfg]["compression_ratio"] for cfg in configs]
    plt.bar(x, comp_ratios)
    plt.ylabel("Compression Ratio")
    plt.title("Memory Compression Ratio by Configuration")
    plt.xticks(x, configs, rotation=90)

    plt.tight_layout()
    plt.show()

    # Additional visualizations
    # 1. Error distribution for each configuration
    n_configs = len(configs)
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 4 * n_rows))

    for i, cfg in enumerate(configs):
        # Select a subset of matrices to visualize
        cfg_parts = cfg.split("_")
        size_part = cfg_parts[0]
        size = tuple(map(int, size_part.split("x")))
        dist = cfg_parts[2]
        dtype_str = cfg_parts[1]
        dtype = getattr(torch, dtype_str)

        # Create a small sample matrix for visualization
        if dist == "uniform":
            matrix = torch.rand((64, 64), dtype=dtype, device="cpu") * 2 - 1
        elif dist == "normal":
            matrix = torch.randn((64, 64), dtype=dtype, device="cpu")
        elif dist == "mixed":
            matrix = torch.randn((64, 64), dtype=dtype, device="cpu")
            mask = torch.rand((64, 64)) > 0.95
            matrix[mask] *= 10
            mask = torch.rand((64, 64)) > 0.95
            matrix[mask] *= 0.01

        # Quantize and dequantize
        quantized, scales = per_block_cast_to_fp8(matrix)
        dequantized = dequantize_vectorized(quantized, scales)

        # Calculate error
        error = torch.abs(matrix - dequantized)

        # Plot
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(error.numpy(), cmap="viridis")
        plt.colorbar()
        plt.title(f'Error Map: {cfg}\nRMSE={results["errors"][cfg]["rmse"]:.6f}')

    plt.tight_layout()
    plt.show()

    # 2. Performance vs Matrix Size
    size_configs = {}
    for cfg in configs:
        parts = cfg.split("_")
        size = parts[0]
        dtype = parts[1]
        dist = parts[2]

        # Group by distribution and dtype
        key = f"{dtype}_{dist}"
        if key not in size_configs:
            size_configs[key] = []

        size_configs[key].append(
            (size, results["times"][cfg]["total"], results["errors"][cfg]["rmse"])
        )

    plt.figure(figsize=(15, 10))

    # Plot timing vs matrix size
    plt.subplot(2, 1, 1)
    for key, values in size_configs.items():
        sizes = [v[0] for v in values]
        times = [v[1] for v in values]
        plt.plot(sizes, times, "o-", label=key)

    plt.xlabel("Matrix Size")
    plt.ylabel("Total Time (seconds)")
    plt.title("Performance Scaling with Matrix Size")
    plt.legend()
    plt.grid(True)

    # Plot RMSE vs matrix size
    plt.subplot(2, 1, 2)
    for key, values in size_configs.items():
        sizes = [v[0] for v in values]
        rmse = [v[2] for v in values]
        plt.plot(sizes, rmse, "o-", label=key)

    plt.xlabel("Matrix Size")
    plt.ylabel("RMSE")
    plt.title("Error Scaling with Matrix Size")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def run_precision_benchmark():
    """
    Run the precision benchmark with standard parameters.
    """
    # Using smaller matrix sizes and fewer configurations for faster execution
    matrix_sizes = [(512, 512), (1024, 1024)]
    dtypes = [torch.float32, torch.float16]
    distributions = ["normal", "mixed"]

    print("Running precision benchmark...")
    results = benchmark_precision(
        matrix_sizes=matrix_sizes,
        dtypes=dtypes,
        distributions=distributions,
        block_size=128,
        n_runs=3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        visualize=True,
    )

    # Print summary table
    print("\n=== Precision Benchmark Summary ===")
    print("\nConfiguration | RMSE | PSNR (dB) | Compression | Total Time (s)")
    print("-" * 75)

    for cfg in results["errors"].keys():
        rmse = results["errors"][cfg]["rmse"]
        psnr = results["errors"][cfg]["psnr"]
        comp = results["memory"][cfg]["compression_ratio"]
        time = results["times"][cfg]["total"]

        print(
            f"{cfg:<15} | {rmse:<6.6f} | {psnr:<9.2f} | {comp:<11.2f}x | {time:<14.6f}"
        )


if __name__ == "__main__":
    run_precision_benchmark()
