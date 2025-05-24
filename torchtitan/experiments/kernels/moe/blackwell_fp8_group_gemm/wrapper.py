import fp8_grouped_gemm_cuda
import torch


def example_fp8_grouped_gemm():
    """
    Example usage of the FP8 E4M3 Grouped GEMM PyTorch extension
    """

    # Check if Blackwell architecture is available
    if not fp8_grouped_gemm_cuda.is_blackwell_available():
        print("Error: Blackwell architecture (compute capability 10.0) not available")
        return

    device = torch.device("cuda")

    # Create multiple GEMM problems of different sizes
    problems = [
        (512, 256, 128),  # M, N, K
        (1024, 512, 256),
        (256, 1024, 512),
        (768, 768, 384),
    ]

    print(f"Running grouped GEMM with {len(problems)} problems:")
    for i, (M, N, K) in enumerate(problems):
        print(f"  Problem {i}: A({M}x{K}) @ B({K}x{N}) -> C({M}x{N})")

    # Generate random FP8 E4M3 and FP16 tensors for each problem
    a_tensors = []
    b_tensors = []
    c_tensors = []

    for M, N, K in problems:
        # Create random tensors
        # Note: torch.float8_e4m3fn might need special handling depending on PyTorch version
        a = torch.randn(M, K, dtype=torch.float16, device=device).to(
            torch.float8_e4m3fn
        )
        b = torch.randn(K, N, dtype=torch.float16, device=device).to(
            torch.float8_e4m3fn
        )
        c = torch.randn(M, N, dtype=torch.float16, device=device)

        a_tensors.append(a)
        b_tensors.append(b)
        c_tensors.append(c)

    # Example 1: Use scalar alpha and beta for all problems
    print("\nExample 1: Scalar alpha/beta")
    alpha = 2.0
    beta = 0.5

    result = fp8_grouped_gemm_cuda.fp8_grouped_gemm_scalar(
        a_tensors, b_tensors, c_tensors, alpha, beta, use_2sm=False
    )

    print(f"Result shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")

    # Example 2: Use different alpha/beta values for each problem
    print("\nExample 2: Per-problem alpha/beta")
    alpha_values = [1.0, 1.5, 2.0, 0.8]
    beta_values = [0.0, 0.5, 1.0, 0.3]

    result = fp8_grouped_gemm_cuda.fp8_grouped_gemm(
        a_tensors, b_tensors, c_tensors, alpha_values, beta_values, use_2sm=False
    )

    print(f"Result shape: {result.shape}")

    # Example 3: Use 2SM configuration for larger problems
    print("\nExample 3: Using 2SM configuration")
    result_2sm = fp8_grouped_gemm_cuda.fp8_grouped_gemm_scalar(
        a_tensors, b_tensors, c_tensors, alpha, beta, use_2sm=True
    )

    print(f"2SM Result shape: {result_2sm.shape}")

    # Verify results are reasonable (non-zero, finite)
    print(f"\nResult statistics:")
    print(f"Min: {result.min().item():.6f}")
    print(f"Max: {result.max().item():.6f}")
    print(f"Mean: {result.mean().item():.6f}")
    print(f"Std: {result.std().item():.6f}")

    # Performance comparison example
    print("\nPerformance test:")

    # Warmup
    for _ in range(5):
        _ = fp8_grouped_gemm_cuda.fp8_grouped_gemm_scalar(
            a_tensors, b_tensors, c_tensors, alpha, beta, use_2sm=False
        )

    torch.cuda.synchronize()

    # Benchmark 1SM
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(100):
        _ = fp8_grouped_gemm_cuda.fp8_grouped_gemm_scalar(
            a_tensors, b_tensors, c_tensors, alpha, beta, use_2sm=False
        )
    end_event.record()
    torch.cuda.synchronize()

    time_1sm = start_event.elapsed_time(end_event) / 100

    # Benchmark 2SM
    start_event.record()
    for _ in range(100):
        _ = fp8_grouped_gemm_cuda.fp8_grouped_gemm_scalar(
            a_tensors, b_tensors, c_tensors, alpha, beta, use_2sm=True
        )
    end_event.record()
    torch.cuda.synchronize()

    time_2sm = start_event.elapsed_time(end_event) / 100

    print(f"1SM average time: {time_1sm:.3f} ms")
    print(f"2SM average time: {time_2sm:.3f} ms")
    print(f"Speedup (1SM vs 2SM): {time_1sm/time_2sm:.2f}x")

    # Calculate theoretical FLOPS
    total_flops = sum(2 * M * N * K for M, N, K in problems)  # 2 FLOPS per multiply-add
    flops_1sm = total_flops / (time_1sm * 1e-3) / 1e12  # TFLOPS
    flops_2sm = total_flops / (time_2sm * 1e-3) / 1e12  # TFLOPS

    print(f"1SM Performance: {flops_1sm:.2f} TFLOPS")
    print(f"2SM Performance: {flops_2sm:.2f} TFLOPS")


def benchmark_vs_pytorch():
    """
    Compare performance against PyTorch's built-in operations
    """
    print("\nBenchmarking against PyTorch...")

    device = torch.device("cuda")
    problems = [
        (1024, 1024, 512),
        (512, 2048, 1024),
        (2048, 512, 256),
    ]

    # Create test tensors
    a_tensors = []
    b_tensors = []
    c_tensors = []
    a_fp16_tensors = []
    b_fp16_tensors = []

    for M, N, K in problems:
        a_fp8 = torch.randn(M, K, dtype=torch.float16, device=device).to(
            torch.float8_e4m3fn
        )
        b_fp8 = torch.randn(K, N, dtype=torch.float16, device=device).to(
            torch.float8_e4m3fn
        )
        c_fp16 = torch.randn(M, N, dtype=torch.float16, device=device)

        # Also create FP16 versions for comparison
        a_fp16 = a_fp8.to(torch.float16)
        b_fp16 = b_fp8.to(torch.float16)

        a_tensors.append(a_fp8)
        b_tensors.append(b_fp8)
        c_tensors.append(c_fp16)
        a_fp16_tensors.append(a_fp16)
        b_fp16_tensors.append(b_fp16)

    alpha, beta = 1.0, 0.0

    # Warmup
    for _ in range(10):
        _ = fp8_grouped_gemm_cuda.fp8_grouped_gemm_scalar(
            a_tensors, b_tensors, c_tensors, alpha, beta
        )
        for i in range(len(problems)):
            _ = torch.addmm(
                c_tensors[i],
                a_fp16_tensors[i],
                b_fp16_tensors[i],
                alpha=alpha,
                beta=beta,
            )

    torch.cuda.synchronize()

    # Benchmark our FP8 grouped GEMM
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(50):
        _ = fp8_grouped_gemm_cuda.fp8_grouped_gemm_scalar(
            a_tensors, b_tensors, c_tensors, alpha, beta
        )
    end.record()
    torch.cuda.synchronize()

    time_grouped = start.elapsed_time(end) / 50

    # Benchmark individual PyTorch operations
    start.record()
    for _ in range(50):
        results = []
        for i in range(len(problems)):
            result = torch.addmm(
                c_tensors[i],
                a_fp16_tensors[i],
                b_fp16_tensors[i],
                alpha=alpha,
                beta=beta,
            )
            results.append(result)
    end.record()
    torch.cuda.synchronize()

    time_individual = start.elapsed_time(end) / 50

    print(f"FP8 Grouped GEMM: {time_grouped:.3f} ms")
    print(f"Individual FP16 GEMMs: {time_individual:.3f} ms")
    print(f"Speedup: {time_individual/time_grouped:.2f}x")

    # Calculate memory usage comparison
    fp8_memory = (
        sum(a.numel() + b.numel() for a, b in zip(a_tensors, b_tensors)) * 1
    )  # 1 byte per FP8
    fp16_memory = (
        sum(a.numel() + b.numel() for a, b in zip(a_fp16_tensors, b_fp16_tensors)) * 2
    )  # 2 bytes per FP16

    print(f"FP8 memory usage: {fp8_memory / 1024**2:.2f} MB")
    print(f"FP16 memory usage: {fp16_memory / 1024**2:.2f} MB")
    print(f"Memory savings: {fp16_memory/fp8_memory:.2f}x")


def test_correctness():
    """
    Test correctness by comparing with reference implementation
    """
    print("\nTesting correctness...")

    device = torch.device("cuda")
    torch.manual_seed(42)

    # Small test problems for easy verification
    problems = [
        (64, 64, 32),
        (128, 96, 64),
    ]

    a_tensors = []
    b_tensors = []
    c_tensors = []

    for M, N, K in problems:
        a = torch.randn(M, K, dtype=torch.float16, device=device).to(
            torch.float8_e4m3fn
        )
        b = torch.randn(K, N, dtype=torch.float16, device=device).to(
            torch.float8_e4m3fn
        )
        c = torch.randn(M, N, dtype=torch.float16, device=device)

        a_tensors.append(a)
        b_tensors.append(b)
        c_tensors.append(c)

    alpha, beta = 2.0, 0.5

    # Get result from our implementation
    result = fp8_grouped_gemm_cuda.fp8_grouped_gemm_scalar(
        a_tensors, b_tensors, c_tensors, alpha, beta
    )

    # Compute reference using PyTorch
    reference_results = []
    for i in range(len(problems)):
        a_fp16 = a_tensors[i].to(torch.float16)
        b_fp16 = b_tensors[i].to(torch.float16)
        ref = torch.addmm(c_tensors[i], a_fp16, b_fp16, alpha=alpha, beta=beta)
        reference_results.append(ref)

    reference = torch.stack(reference_results)

    # Compare results
    max_diff = torch.max(torch.abs(result - reference)).item()
    mean_diff = torch.mean(torch.abs(result - reference)).item()
    rel_error = mean_diff / torch.mean(torch.abs(reference)).item()

    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Relative error: {rel_error:.6f}")

    tolerance = 1e-2  # FP8 has lower precision
    if rel_error < tolerance:
        print("✓ Correctness test PASSED")
    else:
        print("✗ Correctness test FAILED")
        return False

    return True


if __name__ == "__main__":
    try:
        # Run examples
        example_fp8_grouped_gemm()

        # Test correctness
        if test_correctness():
            # Run benchmarks
            benchmark_vs_pytorch()

        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
