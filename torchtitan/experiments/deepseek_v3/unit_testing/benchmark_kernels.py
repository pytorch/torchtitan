# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# benchmark quantization kernels

# sizes to benchmark:
# valid_tokens.shape=torch.Size([2688, 2048])
# hidden_states.shape=torch.Size([2688, 1408])

# benchmark quantization kernels
import time

import dsgemm_kernels
import dsgemm_utils
import torch


def benchmark_quant_kernels(shapes, dtype=torch.bfloat16, warmup=10, iters=100):
    results = []

    for shape in shapes:
        m, k = shape
        print(f"Benchmarking shape: {shape}")

        # Create input tensor
        x = torch.randn(m, k, device="cuda", dtype=dtype)

        # Warmup groupwise_activation_quant
        for _ in range(warmup):
            _ = dsgemm_kernels.groupwise_activation_quant(x)
            torch.cuda.synchronize()

        # Benchmark groupwise_activation_quant
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            y1 = dsgemm_kernels.groupwise_activation_quant(x)
            torch.cuda.synchronize()
        groupwise_time = (time.time() - start) / iters * 1000  # ms

        # Benchmark per_token_cast_to_fp8
        for _ in range(warmup):
            _ = dsgemm_utils.per_token_cast_to_fp8(x)
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            y2 = dsgemm_utils.per_token_cast_to_fp8(x)
            torch.cuda.synchronize()
        per_token_time = (time.time() - start) / iters * 1000  # ms

        # Calculate speedup
        groupwise_vs_per_token = per_token_time / groupwise_time

        # Check correctness
        max_diff = dsgemm_utils.compare_fp8_tensors(y1[0], y2[0])

        results.append(
            {
                "shape": shape,
                "groupwise_ms": groupwise_time,
                "per_token_ms": per_token_time,
                "groupwise_vs_per_token": groupwise_vs_per_token,
                "max_diff": max_diff,
            }
        )

        print(f"  groupwise: {groupwise_time:.3f} ms")
        print(f"  per_token: {per_token_time:.3f} ms")
        print(
            f"  groupwise vs per_token: Groupwise is {groupwise_vs_per_token:.2f}x faster than per_token"
        )
        print(f"  max_diff: {max_diff}")
        print()

    return results


def print_results_table(results):
    print("\nResults Summary:")
    print(
        f"{'Shape':>15} | {'Groupwise (ms)':>15} | {'PyTorch Eager(ms)':>18} | {'Groupwise/PyTorch':>18} "
    )
    print("-" * 85)

    for r in results:
        print(
            f"{str(r['shape']):>15} | {r['groupwise_ms']:>15.3f} | {r['per_token_ms']:>18.3f} | "
            f"{r['groupwise_vs_per_token']:>18.2f}x | "
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
