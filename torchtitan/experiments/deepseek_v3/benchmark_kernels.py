# benchmark quantization kernels

# sizes to benchmark:
# valid_tokens.shape=torch.Size([2688, 2048])
# hidden_states.shape=torch.Size([2688, 1408])

# benchmark quantization kernels
import time

import dsgemm_kernels
import dsgemm_utils
import numpy as np
import torch


def benchmark_quant_kernels(shapes, dtype=torch.bfloat16, warmup=10, iters=100):
    results = []

    for shape in shapes:
        m, k = shape
        print(f"Benchmarking shape: {shape}")

        # Create input tensor
        x = torch.randn(m, k, device="cuda", dtype=dtype)

        # Warmup
        for _ in range(warmup):
            _ = dsgemm_kernels.grid_stride_act_quant(x)
            torch.cuda.synchronize()

        # Benchmark grid_stride_act_quant
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            y1 = dsgemm_kernels.grid_stride_act_quant(x)
            torch.cuda.synchronize()
        grid_stride_time = (time.time() - start) / iters * 1000  # ms

        # Benchmark activation_quant_triton
        for _ in range(warmup):
            _ = dsgemm_kernels.activation_quant_triton(x)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            y2 = dsgemm_kernels.activation_quant_triton(x)
            torch.cuda.synchronize()
        triton_time = (time.time() - start) / iters * 1000  # ms

        # Benchmark per_token_cast_to_fp8
        for _ in range(warmup):

            _ = dsgemm_utils.per_token_cast_to_fp8(x)
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            y3 = dsgemm_utils.per_token_cast_to_fp8(x)
            torch.cuda.synchronize()
        per_token_time = (time.time() - start) / iters * 1000  # ms

        # Calculate speedups
        grid_vs_triton = triton_time / grid_stride_time
        grid_vs_per_token = per_token_time / grid_stride_time

        # Check correctness
        max_diff_triton = dsgemm_utils.compare_fp8_tensors(y1[0], y2[0])
        max_diff_per_token = dsgemm_utils.compare_fp8_tensors(y1[0], y3[0])

        results.append(
            {
                "shape": shape,
                "grid_stride_ms": grid_stride_time,
                "triton_ms": triton_time,
                "per_token_ms": per_token_time,
                "grid_vs_triton": grid_vs_triton,
                "grid_vs_per_token": grid_vs_per_token,
                "max_diff_triton": max_diff_triton,
                "max_diff_per_token": max_diff_per_token,
            }
        )

        print(f"  grid_stride: {grid_stride_time:.3f} ms")
        print(f"  triton: {triton_time:.3f} ms")
        print(f"  per_token: {per_token_time:.3f} ms")
        print(
            f"  grid vs triton: Grid stride is {grid_vs_triton:.2f}x faster than Triton"
        )
        print(
            f"  grid vs per_token: Grid stride is {grid_vs_per_token:.2f}x faster than per_token"
        )
        print(f"  max_diff_triton: {max_diff_triton}")
        print(f"  max_diff_per_token: {max_diff_per_token}")
        print()

    return results


def print_results_table(results):
    print("\nResults Summary:")
    print(
        f"{'Shape':>15} | {'Grid (ms)':>10} | {'Row (ms)':>10} | {'PyTorch Eager(ms)':>10} | {'Grid/Row':>10} | {'Grid/PyTorch':>8} | {'Max Diff':>5}"
    )
    print("-" * 85)

    for r in results:
        print(
            f"{str(r['shape']):>15} | {r['grid_stride_ms']:>10.3f} | {r['triton_ms']:>10.3f} | {r['per_token_ms']:>14.3f} | {r['grid_vs_triton']:>10.2f}x | {r['grid_vs_per_token']:>10.2f}x | {r['max_diff_triton']:>8.3f}"
        )


if __name__ == "__main__":
    # Test various shapes
    shapes = [
        (2048, 2048),
        (2688, 2048),
        (4096, 2048),  # Large batch
        (8192, 2048),
        (2048, 8192),
        # Different feature dimensions
        (1024, 2048),
        (1024, 4096),
        (2048, 4096),
        (2688, 1408),
        # Square matrices
        (2048, 2048),
        (4096, 4096),
    ]

    results = benchmark_quant_kernels(shapes)
    print_results_table(results)
