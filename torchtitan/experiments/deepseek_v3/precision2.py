import math
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

# Import the rowwise FP8 quantization functions
from fp8_gemm import dequantize_fp8_row, get_fp8_constants, quantize_fp8_row

# Import the act quantization functions from kernel.py
from kernel import act_quant, weight_dequant

# Import the existing functions
# from precision_benchmark import dequantize_vectorized, per_block_cast_to_fp8


def generate_test_case(
    case_type: str,
    size: Tuple[int, int] = (1024, 1024),
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate a test matrix with specific characteristics for targeted testing.

    Args:
        case_type: Type of test case to generate
        size: Size of the matrix
        dtype: Data type of the matrix
        device: Device to create tensor on

    Returns:
        torch.Tensor: Generated test matrix
    """
    if case_type == "normal":
        # Standard normal distribution
        return torch.randn(size, dtype=dtype, device=device)

    elif case_type == "uniform_small":
        # Uniform small values [-0.1, 0.1]
        return torch.rand(size, dtype=dtype, device=device) * 0.2 - 0.1

    elif case_type == "uniform_large":
        # Uniform large values [-100, 100]
        return torch.rand(size, dtype=dtype, device=device) * 200 - 100

    elif case_type == "sparse":
        # Mostly zeros with few large values
        matrix = torch.zeros(size, dtype=dtype, device=device)
        mask = torch.rand(size, device=device) > 0.99
        # Ensure the source tensor has the same dtype as the destination
        matrix[mask] = torch.randn(mask.sum().item(), dtype=dtype, device=device) * 10
        return matrix

    elif case_type == "extreme_range":
        # Mix of very small and very large values
        matrix = torch.randn(size, dtype=dtype, device=device)
        # 10% very small values
        small_mask = torch.rand(size, device=device) < 0.1
        matrix[small_mask] *= 1e-5
        # 10% very large values
        large_mask = (torch.rand(size, device=device) < 0.1) & (~small_mask)
        matrix[large_mask] *= 1e5
        return matrix

    elif case_type == "bipolar":
        # Values clustered around -5 and +5
        matrix = torch.randn(size, dtype=dtype, device=device) * 0.5
        mask = torch.rand(size, device=device) > 0.5
        matrix[mask] += 5
        matrix[~mask] -= 5
        return matrix

    elif case_type == "ill_conditioned":
        # Matrix with high condition number (for linear algebra operations)
        n = min(size)
        if n <= 1:
            return torch.randn(size, dtype=dtype, device=device)

        # Create diagonal matrix with exponentially decaying singular values
        diag = torch.logspace(0, -6, n, dtype=dtype, device=device)
        # Create random orthogonal matrices
        q1 = torch.linalg.qr(
            torch.randn(size[0], size[0], dtype=torch.float32, device=device)
        )[0].to(dtype)
        q2 = torch.linalg.qr(
            torch.randn(size[1], size[1], dtype=torch.float32, device=device)
        )[0].to(dtype)
        # Create ill-conditioned matrix A = Q1 * S * Q2^T
        s = torch.zeros(size, dtype=dtype, device=device)
        s[:n, :n] = torch.diag(diag)
        return q1 @ s @ q2.t()

    else:
        raise ValueError(f"Unknown test case type: {case_type}")


def run_targeted_precision_tests(
    test_cases: List[str] = [
        "normal",
        "uniform_small",
        "uniform_large",
        "sparse",
        "extreme_range",
        "bipolar",
        "ill_conditioned",
    ],
    size: Tuple[int, int] = (1024, 1024),
    dtypes: List[torch.dtype] = [torch.float32, torch.float16],
    device: str = "cuda",
    visualize: bool = True,
) -> Dict:
    """
    Run targeted precision tests for specific numerical scenarios.

    Args:
        test_cases: List of test case types to evaluate
        size: Size of test matrices
        dtypes: Data types to test
        device: Device to run tests on
        visualize: Whether to visualize results

    Returns:
        Dict: Test results
    """
    results = {
        "errors": {},
        "times": {},
        "histograms": {},
    }

    for dtype in dtypes:
        for case in test_cases:
            key = f"{case}_{dtype_str}"

            plt.subplot(n_rows, n_cols, i + 1)
            if key in results["histograms"]:
                sample = results["histograms"][key]
                plt.hist(sample, bins=50, alpha=0.7)
                plt.title(f"{case}")
                plt.xlabel("Value")
                plt.ylabel("Frequency")

                # Add error metrics to the plot
                if key in results["errors"]:
                    rmse = results["errors"][key]["rmse"]
                    psnr = results["errors"][key]["psnr"]
                    plt.annotate(
                        f"RMSE: {rmse:.6f}\nPSNR: {psnr:.2f} dB",
                        xy=(0.05, 0.95),
                        xycoords="axes fraction",
                        verticalalignment="top",
                        horizontalalignment="left",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    )

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

    # 5. Plot error distribution for each case
    for dtype in dtypes:
        dtype_str = dtype.__str__().split(".")[-1]
        plt.figure(figsize=(15, 4 * n_rows))
        plt.suptitle(f"Error Distribution Heatmaps ({dtype_str})", fontsize=16)

        for i, case in enumerate(test_cases):
            key = f"{case}_{dtype_str}"

            # Create a small test matrix for visualization
            test_matrix = generate_test_case(case, (64, 64), dtype, "cpu")
            quantized, scales = per_block_cast_to_fp8(test_matrix)
            dequantized = dequantize_vectorized(quantized, scales)
            error = torch.abs(test_matrix - dequantized)

            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(error.numpy(), cmap="viridis")
            plt.colorbar()
            plt.title(f"{case}")

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

    # 6. Plot timing results
    plt.figure(figsize=(12, 6))

    # Prepare data
    x_labels = []
    quant_times = []
    dequant_times = []
    bar_colors = []

    for dtype in dtypes:
        dtype_str = dtype.__str__().split(".")[-1]
        for case in test_cases:
            key = f"{case}_{dtype_str}"
            if key in results["times"]:
                x_labels.append(f"{case}\n({dtype_str})")
                quant_times.append(results["times"][key]["quantization"])
                dequant_times.append(results["times"][key]["dequantization"])

    # Plot
    x = np.arange(len(x_labels))
    width = 0.35

    plt.bar(x - width / 2, quant_times, width, label="Quantization")
    plt.bar(x + width / 2, dequant_times, width, label="Dequantization")

    plt.xticks(x, x_labels, rotation=45)
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time by Test Case and Data Type")
    plt.legend()
    plt.tight_layout()
    plt.show()


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


def test_rowwise_fp8_quantization(
    matrix: torch.Tensor, dtype: torch.dtype = torch.float32
) -> Dict:
    """
    Test rowwise FP8 quantization and dequantization on a given matrix.

    Args:
        matrix: Input tensor to quantize and dequantize
        dtype: Data type of the matrix

    Returns:
        Dict: Dictionary containing error metrics and timing results
    """
    # Ensure matrix is on the correct device and dtype
    if matrix.device == torch.device("cpu"):
        print("Warning: Running on CPU, rowwise quantization may be slower")

    matrix = matrix.to(dtype)

    # Time the quantization process
    start_time = time.time()
    quantized, scales = quantize_fp8_row(matrix)
    quant_time = time.time() - start_time

    # Time the dequantization process
    start_time = time.time()
    dequantized = dequantize_fp8_row(quantized, scales)
    dequant_time = time.time() - start_time

    # Calculate error metrics
    abs_error = torch.abs(matrix - dequantized)
    rel_error = abs_error / (torch.abs(matrix) + 1e-8)  # Avoid division by zero

    # Convert to float32 for quantile calculation
    abs_error_float = abs_error.to(torch.float32)
    rel_error_float = rel_error.to(torch.float32)

    # Compute error metrics
    results = {
        "mean_abs_error": abs_error.mean().item(),
        "median_abs_error": abs_error.median().item(),
        "max_abs_error": abs_error.max().item(),
        "rmse": torch.sqrt(torch.mean(abs_error**2)).item(),
        "mean_rel_error": rel_error.mean().item(),
        "median_rel_error": rel_error.median().item(),
        "max_rel_error": rel_error.max().item(),
        "psnr": 20
        * torch.log10(
            torch.max(torch.abs(matrix)) / torch.sqrt(torch.mean(abs_error**2))
        ).item(),
        "99th_percentile_abs_error": torch.quantile(
            abs_error_float.flatten(), 0.99
        ).item(),
        "99th_percentile_rel_error": torch.quantile(
            rel_error_float.flatten(), 0.99
        ).item(),
        "quantization_time": quant_time,
        "dequantization_time": dequant_time,
        "total_time": quant_time + dequant_time,
    }

    return results


def test_blockwise_fp8_quantization(
    matrix: torch.Tensor, dtype: torch.dtype = torch.float32
) -> Dict:
    """
    Test blockwise FP8 quantization (the existing method) for comparison.

    Args:
        matrix: Input tensor to quantize and dequantize
        dtype: Data type of the matrix

    Returns:
        Dict: Dictionary containing error metrics and timing results
    """
    matrix = matrix.to(dtype)

    # Time the quantization process
    start_time = time.time()
    quantized, scales = per_block_cast_to_fp8(matrix)
    quant_time = time.time() - start_time

    # Time the dequantization process
    start_time = time.time()
    dequantized = dequantize_vectorized(quantized, scales)
    dequant_time = time.time() - start_time

    # Calculate error metrics
    abs_error = torch.abs(matrix - dequantized)
    rel_error = abs_error / (torch.abs(matrix) + 1e-8)  # Avoid division by zero

    # Compute error metrics
    results = {
        "mean_abs_error": abs_error.mean().item(),
        "median_abs_error": abs_error.median().item(),
        "max_abs_error": abs_error.max().item(),
        "rmse": torch.sqrt(torch.mean(abs_error**2)).item(),
        "mean_rel_error": rel_error.mean().item(),
        "median_rel_error": rel_error.median().item(),
        "max_rel_error": rel_error.max().item(),
        "psnr": 20
        * torch.log10(
            torch.max(torch.abs(matrix)) / torch.sqrt(torch.mean(abs_error**2))
        ).item(),
        "99th_percentile_abs_error": torch.quantile(abs_error.flatten(), 0.99).item(),
        "99th_percentile_rel_error": torch.quantile(rel_error.flatten(), 0.99).item(),
        "quantization_time": quant_time,
        "dequantization_time": dequant_time,
        "total_time": quant_time + dequant_time,
    }

    return results


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


def test_act_quantization(
    matrix: torch.Tensor, dtype: torch.dtype = torch.float32, block_size: int = 128
) -> Dict:
    """
    Test act quantization (groupwise 1×128 1D) from kernel.py.

    Args:
        matrix: Input tensor to quantize and dequantize
        dtype: Data type of the matrix
        block_size: Block size for the act quantization (default: 128)

    Returns:
        Dict: Dictionary containing error metrics and timing results
    """
    matrix = matrix.to(dtype)

    # Ensure the last dimension is divisible by block_size
    if matrix.size(-1) % block_size != 0:
        # Pad the matrix to make it divisible by block_size
        pad_size = block_size - (matrix.size(-1) % block_size)
        matrix_padded = torch.nn.functional.pad(matrix, (0, pad_size))
    else:
        matrix_padded = matrix

    # Time the quantization process
    start_time = time.time()
    quantized, scales = act_quant(matrix_padded, block_size=block_size)
    quant_time = time.time() - start_time

    # Time the dequantization process
    start_time = time.time()

    # Convert to float32 for dequantization
    dequant = quantized.to(torch.float32)

    # For act_quant, scales are per row and need to be applied to each block of size block_size
    # Reshape for broadcasting: scales has shape [num_rows, num_blocks_per_row]
    # We need to apply each scale to its corresponding block

    # Get original shape
    orig_shape = matrix_padded.shape

    # For 2D tensors
    if dequant.dim() == 2:
        # Reshape to [num_rows, num_blocks, block_size]
        num_blocks = dequant.size(1) // block_size
        dequant = dequant.view(dequant.size(0), num_blocks, block_size)

        # Reshape scales for broadcasting [num_rows, num_blocks, 1]
        scales_view = scales.view(scales.size(0), scales.size(1), 1)

        # Apply scales
        dequant = dequant * scales_view

        # Reshape back to original
        dequant = dequant.view(orig_shape)
    else:
        # For higher dimensions, reshape to 2D first
        flat_shape = (-1, matrix_padded.size(-1))
        dequant = dequant.reshape(flat_shape)

        # Reshape to [num_rows, num_blocks, block_size]
        num_blocks = dequant.size(1) // block_size
        dequant = dequant.view(dequant.size(0), num_blocks, block_size)

        # Reshape scales for broadcasting [num_rows, num_blocks, 1]
        scales_view = scales.reshape(scales.size(0), scales.size(1), 1)

        # Apply scales
        dequant = dequant * scales_view

        # Reshape back to original
        dequant = dequant.reshape(orig_shape)

    dequantized = dequant
    dequant_time = time.time() - start_time

    # Truncate back to original size if padding was applied
    if matrix_padded.size(-1) != matrix.size(-1):
        dequantized = dequantized[..., : matrix.size(-1)]

    # Calculate error metrics
    abs_error = torch.abs(matrix - dequantized)
    rel_error = abs_error / (torch.abs(matrix) + 1e-8)  # Avoid division by zero

    # Compute error metrics
    results = {
        "mean_abs_error": abs_error.mean().item(),
        "median_abs_error": abs_error.median().item(),
        "max_abs_error": abs_error.max().item(),
        "rmse": torch.sqrt(torch.mean(abs_error**2)).item(),
        "mean_rel_error": rel_error.mean().item(),
        "median_rel_error": rel_error.median().item(),
        "max_rel_error": rel_error.max().item(),
        "psnr": 20
        * torch.log10(
            torch.max(torch.abs(matrix)) / torch.sqrt(torch.mean(abs_error**2))
        ).item(),
        "99th_percentile_abs_error": torch.quantile(abs_error.flatten(), 0.99).item(),
        "99th_percentile_rel_error": torch.quantile(rel_error.flatten(), 0.99).item(),
        "quantization_time": quant_time,
        "dequantization_time": dequant_time,
        "total_time": quant_time + dequant_time,
    }

    return results


def test_act_quantization_old(
    matrix: torch.Tensor, dtype: torch.dtype = torch.float32, block_size: int = 128
) -> Dict:
    """
    Test act quantization (groupwise 1×128 1D) from kernel.py.

    Args:
        matrix: Input tensor to quantize and dequantize
        dtype: Data type of the matrix
        block_size: Block size for the act quantization (default: 128)

    Returns:
        Dict: Dictionary containing error metrics and timing results
    """
    matrix = matrix.to(dtype)

    # Ensure the last dimension is divisible by block_size
    if matrix.size(-1) % block_size != 0:
        # Pad the matrix to make it divisible by block_size
        pad_size = block_size - (matrix.size(-1) % block_size)
        matrix_padded = torch.nn.functional.pad(matrix, (0, pad_size))
    else:
        matrix_padded = matrix

    # Time the quantization process
    start_time = time.time()
    quantized, scales = act_quant(matrix_padded, block_size=block_size)
    quant_time = time.time() - start_time

    # Time the dequantization process
    # If matrix is 2D, we can use weight_dequant, otherwise reshape
    if matrix_padded.dim() == 2:
        start_time = time.time()
        dequantized = dequantize_vectorized(quantized, scales, block_size=block_size)
        dequant_time = time.time() - start_time
    else:
        # For higher dimensions, we need to reshape
        orig_shape = matrix_padded.shape
        reshaped = matrix_padded.reshape(-1, matrix_padded.size(-1))

        start_time = time.time()
        # Reshape scales as well if needed
        scales_reshaped = scales.reshape(-1, scales.size(-1))
        dequantized = dequantize_vectorized(
            quantized.reshape(-1, quantized.size(-1)),
            scales_reshaped,
            block_size=block_size,
        )
        dequantized = dequantized.reshape(orig_shape)
        dequant_time = time.time() - start_time

    # Truncate back to original size if padding was applied
    if matrix_padded.size(-1) != matrix.size(-1):
        dequantized = dequantized[..., : matrix.size(-1)]

    # Calculate error metrics
    abs_error = torch.abs(matrix - dequantized)
    rel_error = abs_error / (torch.abs(matrix) + 1e-8)  # Avoid division by zero

    # Compute error metrics
    results = {
        "mean_abs_error": abs_error.mean().item(),
        "median_abs_error": abs_error.median().item(),
        "max_abs_error": abs_error.max().item(),
        "rmse": torch.sqrt(torch.mean(abs_error**2)).item(),
        "mean_rel_error": rel_error.mean().item(),
        "median_rel_error": rel_error.median().item(),
        "max_rel_error": rel_error.max().item(),
        "psnr": 20
        * torch.log10(
            torch.max(torch.abs(matrix)) / torch.sqrt(torch.mean(abs_error**2))
        ).item(),
        "99th_percentile_abs_error": torch.quantile(abs_error.flatten(), 0.99).item(),
        "99th_percentile_rel_error": torch.quantile(rel_error.flatten(), 0.99).item(),
        "quantization_time": quant_time,
        "dequantization_time": dequant_time,
        "total_time": quant_time + dequant_time,
    }

    return results


def compare_quantization_methods(
    test_cases: List[str] = [
        "normal",
        "uniform_small",
        "uniform_large",
        "sparse",
        "extreme_range",
        "bipolar",
        "ill_conditioned",
    ],
    size: Tuple[int, int] = (1024, 1024),
    dtypes: List[torch.dtype] = [torch.float32, torch.float16],
    device: str = "cuda",
    visualize: bool = True,
) -> Dict:
    """
    Compare four quantization methods across different test cases.

    Args:
        test_cases: List of test case types to evaluate
        size: Size of test matrices
        dtypes: Data types to test
        device: Device to run tests on
        visualize: Whether to visualize results

    Returns:
        Dict: Test results comparing all methods
    """

    results = {
        "rowwise": {},
        "blockwise": {},
        "act_quant_128": {},
        "act_quant_32": {},  # Added 1x32 method
    }

    # Make sure last dimension is divisible by 128 for act quantization
    act_size_128 = list(size)
    if act_size_128[-1] % 128 != 0:
        act_size_128[-1] = ((act_size_128[-1] // 128) + 1) * 128
    act_size_128 = tuple(act_size_128)

    # Make sure last dimension is divisible by 32 for act quantization with block_size=32
    act_size_32 = list(size)
    if act_size_32[-1] % 32 != 0:
        act_size_32[-1] = ((act_size_32[-1] // 32) + 1) * 32
    act_size_32 = tuple(act_size_32)

    for dtype in dtypes:
        dtype_str = dtype.__str__().split(".")[-1]

        for case in test_cases:
            key = f"{case}_{dtype_str}"
            print(f"\nTesting {key}...")

            # Generate test matrices
            matrix = generate_test_case(case, size, dtype, device)
            matrix_act_128 = generate_test_case(case, act_size_128, dtype, device)
            matrix_act_32 = generate_test_case(case, act_size_32, dtype, device)

            # Test rowwise quantization
            print("Testing rowwise quantization...")
            rowwise_results = test_rowwise_fp8_quantization(matrix, dtype)
            results["rowwise"][key] = rowwise_results

            # Test blockwise quantization
            print("Testing blockwise quantization...")
            blockwise_results = test_blockwise_fp8_quantization(matrix, dtype)
            results["blockwise"][key] = blockwise_results

            # Test act quantization with block_size=128
            print("Testing act quantization (1x128)...")
            act_results_128 = test_act_quantization(
                matrix_act_128, dtype, block_size=128
            )
            results["act_quant_128"][key] = act_results_128

            # Test act quantization with block_size=32
            print("Testing act quantization (1x32)...")
            act_results_32 = test_act_quantization(matrix_act_32, dtype, block_size=32)
            results["act_quant_32"][key] = act_results_32

            # Print summary comparison
            print(
                f"Rowwise RMSE: {rowwise_results['rmse']:.6f}, PSNR: {rowwise_results['psnr']:.2f} dB"
            )
            print(
                f"Blockwise RMSE: {blockwise_results['rmse']:.6f}, PSNR: {blockwise_results['psnr']:.2f} dB"
            )
            print(
                f"Act Quant 1x128 RMSE: {act_results_128['rmse']:.6f}, PSNR: {act_results_128['psnr']:.2f} dB"
            )
            print(
                f"Act Quant 1x32 RMSE: {act_results_32['rmse']:.6f}, PSNR: {act_results_32['psnr']:.2f} dB"
            )
            print(f"Rowwise time: {rowwise_results['total_time']:.6f}s")
            print(f"Blockwise time: {blockwise_results['total_time']:.6f}s")
            print(f"Act Quant 1x128 time: {act_results_128['total_time']:.6f}s")
            print(f"Act Quant 1x32 time: {act_results_32['total_time']:.6f}s")

    # Visualize results if requested
    # if visualize:
    #    visualize_comparison_results(results, test_cases, dtypes)

    return results


def compare_quantization_methods_old(
    test_cases: List[str] = [
        "normal",
        "uniform_small",
        "uniform_large",
        "sparse",
        "extreme_range",
        "bipolar",
        "ill_conditioned",
    ],
    size: Tuple[int, int] = (1024, 1024),
    dtypes: List[torch.dtype] = [torch.float32, torch.float16],
    device: str = "cuda",
    visualize: bool = True,
) -> Dict:
    """
    Compare three quantization methods across different test cases.

    Args:
        test_cases: List of test case types to evaluate
        size: Size of test matrices
        dtypes: Data types to test
        device: Device to run tests on
        visualize: Whether to visualize results

    Returns:
        Dict: Test results comparing all methods
    """

    results = {
        "rowwise": {},
        "blockwise": {},
        "act_quant": {},
    }

    # Make sure last dimension is divisible by 128 for act quantization
    act_size = list(size)
    if act_size[-1] % 128 != 0:
        act_size[-1] = ((act_size[-1] // 128) + 1) * 128
    act_size = tuple(act_size)

    for dtype in dtypes:
        dtype_str = dtype.__str__().split(".")[-1]

        for case in test_cases:
            key = f"{case}_{dtype_str}"
            print(f"\nTesting {key}...")

            # Generate test matrices
            matrix = generate_test_case(case, size, dtype, device)
            matrix_act = generate_test_case(case, act_size, dtype, device)

            # Test rowwise quantization
            print("Testing rowwise quantization...")
            rowwise_results = test_rowwise_fp8_quantization(matrix, dtype)
            results["rowwise"][key] = rowwise_results

            # Test blockwise quantization
            print("Testing blockwise quantization...")
            blockwise_results = test_blockwise_fp8_quantization(matrix, dtype)
            results["blockwise"][key] = blockwise_results

            # Test act quantization
            print("Testing act quantization...")
            act_results = test_act_quantization(matrix_act, dtype)
            results["act_quant"][key] = act_results

            # Print summary comparison
            print(
                f"Rowwise RMSE: {rowwise_results['rmse']:.6f}, PSNR: {rowwise_results['psnr']:.2f} dB"
            )
            print(
                f"Blockwise RMSE: {blockwise_results['rmse']:.6f}, PSNR: {blockwise_results['psnr']:.2f} dB"
            )
            print(
                f"Act Quant RMSE: {act_results['rmse']:.6f}, PSNR: {act_results['psnr']:.2f} dB"
            )
            print(f"Rowwise time: {rowwise_results['total_time']:.6f}s")
            print(f"Blockwise time: {blockwise_results['total_time']:.6f}s")
            print(f"Act Quant time: {act_results['total_time']:.6f}s")

    # Visualize results if requested
    if visualize:
        visualize_comparison_results(results, test_cases, dtypes)

    return results


def visualize_comparison_results(
    results: Dict, test_cases: List[str], dtypes: List[torch.dtype]
) -> None:
    """
    Visualize the comparison results between quantization methods.

    Args:
        results: Dictionary containing test results
        test_cases: List of test case types evaluated
        dtypes: Data types tested
    """
    import matplotlib.pyplot as plt

    # Prepare data for plotting
    metrics = ["rmse", "psnr", "mean_rel_error", "total_time"]
    metric_titles = {
        "rmse": "Root Mean Square Error (lower is better)",
        "psnr": "Peak Signal-to-Noise Ratio in dB (higher is better)",
        "mean_rel_error": "Mean Relative Error (lower is better)",
        "total_time": "Total Execution Time in seconds (lower is better)",
    }

    for metric in metrics:
        plt.figure(figsize=(15, 6))

        # Prepare data
        x_labels = []
        rowwise_values = []
        blockwise_values = []
        act_values = []

        for dtype in dtypes:
            dtype_str = dtype.__str__().split(".")[-1]
            for case in test_cases:
                key = f"{case}_{dtype_str}"
                if (
                    key in results["rowwise"]
                    and key in results["blockwise"]
                    and key in results["act_quant"]
                ):
                    x_labels.append(f"{case}\n({dtype_str})")
                    rowwise_values.append(results["rowwise"][key][metric])
                    blockwise_values.append(results["blockwise"][key][metric])
                    act_values.append(results["act_quant"][key][metric])

        # Plot
        x = np.arange(len(x_labels))
        width = 0.25

        plt.bar(x - width, rowwise_values, width, label="Rowwise")
        plt.bar(x, blockwise_values, width, label="Blockwise")
        plt.bar(x + width, act_values, width, label="Act Quant")

        plt.xticks(x, x_labels, rotation=45)
        plt.ylabel(metric_titles[metric])
        plt.title(f"Comparison of {metric_titles[metric]} between Quantization Methods")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Plot relative improvement/difference between methods
    for ref_method in ["blockwise"]:
        plt.figure(figsize=(15, 8))

        # Calculate relative improvement for RMSE
        x_labels = []
        rowwise_rmse_improvement = []
        act_rmse_improvement = []
        rowwise_time_improvement = []
        act_time_improvement = []

        for dtype in dtypes:
            dtype_str = dtype.__str__().split(".")[-1]
            for case in test_cases:
                key = f"{case}_{dtype_str}"
                if (
                    key in results["rowwise"]
                    and key in results["blockwise"]
                    and key in results["act_quant"]
                ):
                    x_labels.append(f"{case}\n({dtype_str})")

                    # For RMSE, lower is better so (ref-method)/ref
                    rowwise_rel_imp = (
                        (
                            results[ref_method][key]["rmse"]
                            - results["rowwise"][key]["rmse"]
                        )
                        / results[ref_method][key]["rmse"]
                        * 100
                    )
                    rowwise_rmse_improvement.append(rowwise_rel_imp)

                    act_rel_imp = (
                        (
                            results[ref_method][key]["rmse"]
                            - results["act_quant"][key]["rmse"]
                        )
                        / results[ref_method][key]["rmse"]
                        * 100
                    )
                    act_rmse_improvement.append(act_rel_imp)

                    # For time, lower is better so (ref-method)/ref
                    rowwise_time_imp = (
                        (
                            results[ref_method][key]["total_time"]
                            - results["rowwise"][key]["total_time"]
                        )
                        / results[ref_method][key]["total_time"]
                        * 100
                    )
                    rowwise_time_improvement.append(rowwise_time_imp)

                    act_time_imp = (
                        (
                            results[ref_method][key]["total_time"]
                            - results["act_quant"][key]["total_time"]
                        )
                        / results[ref_method][key]["total_time"]
                        * 100
                    )
                    act_time_improvement.append(act_time_imp)

        # Plot
        plt.subplot(2, 1, 1)
        x = np.arange(len(x_labels))
        width = 0.35

        plt.bar(
            x - width / 2, rowwise_rmse_improvement, width, label="Rowwise vs Blockwise"
        )
        plt.bar(
            x + width / 2, act_rmse_improvement, width, label="Act Quant vs Blockwise"
        )

        plt.axhline(y=0, color="r", linestyle="-")
        plt.xticks(x, x_labels, rotation=45)
        plt.ylabel("RMSE Improvement (%)")
        plt.title(
            f"Relative RMSE Improvement Compared to {ref_method.capitalize()} Method"
        )
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.bar(
            x - width / 2, rowwise_time_improvement, width, label="Rowwise vs Blockwise"
        )
        plt.bar(
            x + width / 2, act_time_improvement, width, label="Act Quant vs Blockwise"
        )

        plt.axhline(y=0, color="r", linestyle="-")
        plt.xticks(x, x_labels, rotation=45)
        plt.ylabel("Time Improvement (%)")
        plt.title(
            f"Relative Time Improvement Compared to {ref_method.capitalize()} Method"
        )
        plt.legend()

        plt.tight_layout()
        plt.show()


def visualize_comparison_results_old(
    results: Dict, test_cases: List[str], dtypes: List[torch.dtype]
) -> None:
    """
    Visualize the comparison results between quantization methods.

    Args:
        results: Dictionary containing test results
        test_cases: List of test case types evaluated
        dtypes: Data types tested
    """
    import matplotlib.pyplot as plt

    # Prepare data for plotting
    metrics = ["rmse", "psnr", "mean_rel_error", "total_time"]
    metric_titles = {
        "rmse": "Root Mean Square Error (lower is better)",
        "psnr": "Peak Signal-to-Noise Ratio in dB (higher is better)",
        "mean_rel_error": "Mean Relative Error (lower is better)",
        "total_time": "Total Execution Time in seconds (lower is better)",
    }

    for metric in metrics:
        plt.figure(figsize=(15, 6))

        # Prepare data
        x_labels = []
        rowwise_values = []
        blockwise_values = []
        act_values = []

        for dtype in dtypes:
            dtype_str = dtype.__str__().split(".")[-1]
            for case in test_cases:
                key = f"{case}_{dtype_str}"
                if (
                    key in results["rowwise"]
                    and key in results["blockwise"]
                    and key in results["act_quant"]
                ):
                    x_labels.append(f"{case}\n({dtype_str})")
                    rowwise_values.append(results["rowwise"][key][metric])
                    blockwise_values.append(results["blockwise"][key][metric])
                    act_values.append(results["act_quant"][key][metric])

        # Plot
        x = np.arange(len(x_labels))
        width = 0.25

        plt.bar(x - width, rowwise_values, width, label="Rowwise")
        plt.bar(x, blockwise_values, width, label="Blockwise")
        plt.bar(x + width, act_values, width, label="Act Quant")

        plt.xticks(x, x_labels, rotation=45)
        plt.ylabel(metric_titles[metric])
        plt.title(f"Comparison of {metric_titles[metric]} between Quantization Methods")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Plot relative improvement/difference between methods
    for ref_method in ["blockwise"]:
        plt.figure(figsize=(15, 8))

        # Calculate relative improvement for RMSE
        x_labels = []
        rowwise_rmse_improvement = []
        act_rmse_improvement = []
        rowwise_time_improvement = []
        act_time_improvement = []

        for dtype in dtypes:
            dtype_str = dtype.__str__().split(".")[-1]
            for case in test_cases:
                key = f"{case}_{dtype_str}"
                if (
                    key in results["rowwise"]
                    and key in results["blockwise"]
                    and key in results["act_quant"]
                ):
                    x_labels.append(f"{case}\n({dtype_str})")

                    # For RMSE, lower is better so (ref-method)/ref
                    rowwise_rel_imp = (
                        (
                            results[ref_method][key]["rmse"]
                            - results["rowwise"][key]["rmse"]
                        )
                        / results[ref_method][key]["rmse"]
                        * 100
                    )
                    rowwise_rmse_improvement.append(rowwise_rel_imp)

                    act_rel_imp = (
                        (
                            results[ref_method][key]["rmse"]
                            - results["act_quant"][key]["rmse"]
                        )
                        / results[ref_method][key]["rmse"]
                        * 100
                    )
                    act_rmse_improvement.append(act_rel_imp)

                    # For time, lower is better so (ref-method)/ref
                    rowwise_time_imp = (
                        (
                            results[ref_method][key]["total_time"]
                            - results["rowwise"][key]["total_time"]
                        )
                        / results[ref_method][key]["total_time"]
                        * 100
                    )
                    rowwise_time_improvement.append(rowwise_time_imp)

                    act_time_imp = (
                        (
                            results[ref_method][key]["total_time"]
                            - results["act_quant"][key]["total_time"]
                        )
                        / results[ref_method][key]["total_time"]
                        * 100
                    )
                    act_time_improvement.append(act_time_imp)

        # Plot
        plt.subplot(2, 1, 1)
        x = np.arange(len(x_labels))
        width = 0.35

        plt.bar(
            x - width / 2, rowwise_rmse_improvement, width, label="Rowwise vs Blockwise"
        )
        plt.bar(
            x + width / 2, act_rmse_improvement, width, label="Act Quant vs Blockwise"
        )

        plt.axhline(y=0, color="r", linestyle="-")
        plt.xticks(x, x_labels, rotation=45)
        plt.ylabel("RMSE Improvement (%)")
        plt.title(
            f"Relative RMSE Improvement Compared to {ref_method.capitalize()} Method"
        )
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.bar(
            x - width / 2, rowwise_time_improvement, width, label="Rowwise vs Blockwise"
        )
        plt.bar(
            x + width / 2, act_time_improvement, width, label="Act Quant vs Blockwise"
        )

        plt.axhline(y=0, color="r", linestyle="-")
        plt.xticks(x, x_labels, rotation=45)
        plt.ylabel("Time Improvement (%)")
        plt.title(
            f"Relative Time Improvement Compared to {ref_method.capitalize()} Method"
        )
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Add a detailed view of quantization vs dequantization time
    plt.figure(figsize=(15, 12))

    # Prepare data
    x_labels = []
    rowwise_quant_times = []
    rowwise_dequant_times = []
    blockwise_quant_times = []
    blockwise_dequant_times = []
    act_quant_times = []
    act_dequant_times = []

    for dtype in dtypes:
        dtype_str = dtype.__str__().split(".")[-1]
        for case in test_cases:
            key = f"{case}_{dtype_str}"
            if (
                key in results["rowwise"]
                and key in results["blockwise"]
                and key in results["act_quant"]
            ):
                x_labels.append(f"{case}\n({dtype_str})")

                rowwise_quant_times.append(results["rowwise"][key]["quantization_time"])
                rowwise_dequant_times.append(
                    results["rowwise"][key]["dequantization_time"]
                )

                blockwise_quant_times.append(
                    results["blockwise"][key]["quantization_time"]
                )
                blockwise_dequant_times.append(
                    results["blockwise"][key]["dequantization_time"]
                )

                act_quant_times.append(results["act_quant"][key]["quantization_time"])
                act_dequant_times.append(
                    results["act_quant"][key]["dequantization_time"]
                )

    # Plot quantization times
    plt.subplot(2, 1, 1)
    x = np.arange(len(x_labels))
    width = 0.25

    plt.bar(x - width, rowwise_quant_times, width, label="Rowwise")
    plt.bar(x, blockwise_quant_times, width, label="Blockwise")
    plt.bar(x + width, act_quant_times, width, label="Act Quant")

    plt.xticks(x, x_labels, rotation=45)
    plt.ylabel("Time (seconds)")
    plt.title("Quantization Time Comparison")
    plt.legend()

    # Plot dequantization times
    plt.subplot(2, 1, 2)
    plt.bar(x - width, rowwise_dequant_times, width, label="Rowwise")
    plt.bar(x, blockwise_dequant_times, width, label="Blockwise")
    plt.bar(x + width, act_dequant_times, width, label="Act Quant")

    plt.xticks(x, x_labels, rotation=45)
    plt.ylabel("Time (seconds)")
    plt.title("Dequantization Time Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()


def run_extended_precision_tests():
    """
    Run the extended precision tests including all three quantization methods.
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Define test cases
    test_cases = [
        "normal",
        "uniform_small",
        "uniform_large",
        "sparse",
        "extreme_range",
        "bipolar",
        "ill_conditioned",
    ]

    # Define matrix sizes to test (larger sizes)
    sizes = [(2048, 2048), (4096, 4096), (2048, 8192)]

    # Define data types to test
    dtypes = [
        torch.bfloat16,
    ]

    # Run tests for each size
    for size in sizes:
        print(
            f"\n=== Running Extended Precision Tests with Three Methods (Size: {size}) ==="
        )
        results = compare_quantization_methods(
            test_cases=test_cases,
            size=size,
            dtypes=dtypes,
            device=device,
            visualize=True,
        )

        # Print summary table grouped by test case and dtype
        print(f"\n=== Extended Precision Test Results (Size: {size}) ===")

        for dtype in dtypes:
            dtype_str = dtype.__str__().split(".")[-1]
            print(f"\n--- Data Type: {dtype_str} ---")

            for case in test_cases:
                key = f"{case}_{dtype_str}"
                print(f"\nTest Case: {case}")
                print(
                    "Method    | RMSE      | PSNR (dB) | Mean Rel Error | Total Time (s)"
                )
                print("-" * 75)

                for method in ["rowwise", "blockwise", "act_quant_128", "act_quant_32"]:
                    if key in results[method]:
                        rmse = results[method][key]["rmse"]
                        psnr = results[method][key]["psnr"]
                        mean_rel = results[method][key]["mean_rel_error"]
                        total_time = results[method][key]["total_time"]

                        print(
                            f"{method:<10}| {rmse:<10.6f}| {psnr:<10.2f}| {mean_rel:<15.6f}| {total_time:<12.6f}"
                        )


if __name__ == "__main__":
    run_extended_precision_tests()
