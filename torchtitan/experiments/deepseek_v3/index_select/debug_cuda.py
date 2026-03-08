"""
Simple benchmark comparing CUDA permute implementation vs PyTorch with result verification.
"""

import time

import numpy as np
import torch


def verify_results_match(result1, result2, rtol=1e-5, atol=1e-8):
    """
    Verify that two tensors match within a tolerance.

    Args:
        result1: First tensor
        result2: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        bool: True if results match, False otherwise
    """
    # Check shapes match
    if result1.shape != result2.shape:
        print(f"Shape mismatch: {result1.shape} vs {result2.shape}")
        return False

    # Check values match within tolerance
    if not torch.allclose(result1, result2, rtol=rtol, atol=atol):
        # Find max absolute difference
        max_diff = torch.max(torch.abs(result1 - result2)).item()
        print(f"Values don't match. Max difference: {max_diff}")
        return False

    return True


def benchmark_implementation(
    cuda_impl, batch_size=4096, hidden_dim=4096, n_indices=2048, warmup=10, repeat=50
):
    """
    Benchmark the CUDA implementation against PyTorch and verify results match.

    Args:
        cuda_impl: The CUDA implementation function
        batch_size: Size of input tensor's first dimension
        hidden_dim: Size of input tensor's second dimension
        n_indices: Number of indices to use for permutation
        warmup: Number of warmup iterations
        repeat: Number of benchmark iterations
    """
    print(
        f"Benchmarking with batch_size={batch_size}, hidden_dim={hidden_dim}, n_indices={n_indices}"
    )

    # Create test data
    input_tensor = torch.randn(
        batch_size, hidden_dim, device="cuda", dtype=torch.float16
    )
    indices = torch.randint(0, batch_size, (n_indices,), device="cuda")

    # -----------------------
    # Verify results match
    # -----------------------
    print("Verifying CUDA implementation matches PyTorch...")

    # Run both implementations
    pytorch_result = input_tensor[indices]
    cuda_result = cuda_impl(input_tensor, indices)

    # Verify results match
    if verify_results_match(pytorch_result, cuda_result):
        print("✓ Results match!")
    else:
        print("✗ Results don't match!")
        return

    # -----------------------
    # Benchmark implementations
    # -----------------------

    # Warm up PyTorch
    for _ in range(warmup):
        _ = input_tensor[indices]

    # Benchmark PyTorch
    torch.cuda.synchronize()
    pytorch_times = []
    for _ in range(repeat):
        start = time.time()
        _ = input_tensor[indices]
        torch.cuda.synchronize()
        pytorch_times.append((time.time() - start) * 1000)  # ms

    pytorch_mean = np.mean(pytorch_times)
    pytorch_std = np.std(pytorch_times)

    # Warm up CUDA implementation
    for _ in range(warmup):
        _ = cuda_impl(input_tensor, indices)

    # Benchmark CUDA implementation
    torch.cuda.synchronize()
    cuda_times = []
    for _ in range(repeat):
        start = time.time()
        _ = cuda_impl(input_tensor, indices)
        torch.cuda.synchronize()
        cuda_times.append((time.time() - start) * 1000)  # ms

    cuda_mean = np.mean(cuda_times)
    cuda_std = np.std(cuda_times)

    # Print results
    print(f"PyTorch:  {pytorch_mean:.3f} ± {pytorch_std:.3f} ms")
    print(f"CUDA:     {cuda_mean:.3f} ± {cuda_std:.3f} ms")
    print(f"Speedup:  {pytorch_mean / cuda_mean:.2f}x")


if __name__ == "__main__":
    try:
        import fast_permute_tokens_cuda

        # Define wrapper function
        def fast_permute(input_tensor, indices):
            return fast_permute_tokens_cuda.fast_permute_tokens(  # fast_permute_tokens_triton(
                input_tensor,
                indices,
            )

        # Run benchmark with different configurations
        print("\n=== Small Configuration ===")
        benchmark_implementation(
            fast_permute, batch_size=1024, hidden_dim=4096, n_indices=512
        )

        print("\n=== Medium Configuration ===")
        benchmark_implementation(
            fast_permute, batch_size=4096, hidden_dim=4096, n_indices=4096
        )

        print("\n=== Large Configuration ===")
        benchmark_implementation(
            fast_permute, batch_size=8192, hidden_dim=4096, n_indices=8192
        )

    except ImportError:
        print("CUDA cpp extension not available. .")
        print("Run: python fast_permute_tokens_setup.py install")
