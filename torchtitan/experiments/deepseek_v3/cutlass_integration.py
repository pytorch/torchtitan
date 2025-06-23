#!/usr/bin/env python3
"""
Integration testing

current errors:
============================================================
Initializing CUTLASSGroupedGemmStrategy for Blackwell
cute hardware - device_id 0
cute hardware - driver_version 12080
max_dynamic_shared_memory: 232448
max_active_blocks: 1
max_active_blocks: 33
Initialized CUTLASSGroupedGemmStrategy for Blackwell with:
  - 2 CTA instructions: True
  - MMA tiler (M, N): (256, 128)
  - Cluster shape (M, N): (2, 2)
  - Cluster size: 4
🔍 Testing Forward Pass...
max_dynamic_shared_memory: 232448
max_active_blocks: 1
Compiling CUTLASS grouped GEMM kernel: 8 groups, 2CTA=True, cluster=(2, 2)
Kernel compilation successful
   Forward max difference: 0.00e+00
   Forward outputs close: ✓
🔍 Testing Backward Pass...
Compiling CUTLASS grouped GEMM kernel: 8 groups, 2CTA=True, cluster=(2, 2)
Kernel compilation successful
Compiling CUTLASS grouped GEMM kernel: 8 groups, 2CTA=True, cluster=(2, 2)
Kernel compilation successful
   Input grad max difference: 2.34e+02
   Input gradients close: ❌
   Weight grad max difference: 9.10e+01
   Weight gradients close: ❌


"""

import os
import sys

import torch
import torch.nn as nn


try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils

    from cutlass.cute.runtime import from_dlpack
    from cutlass_backwards import (
        CUTLASSBackwardGroupGemm,
        CUTLASSGroupedGemmStrategy,
        CUTLASSGroupedLinear,
    )

    CUTLASS_AVAILABLE = True
except ImportError:
    print("CUTLASS modules not found. Please update the import paths.")
    CUTLASS_AVAILABLE = False


from cutlass_test_driver import GroupGemmTestDriver, PyTorchManualGroupedLinear


def create_cutlass_strategy(
    use_2cta_instrs=True, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 2)
):
    """Create a CUTLASS strategy with specified configuration."""
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    strategy = CUTLASSGroupedGemmStrategy(
        custom_activation=nn.SiLU(),  # Identity for linear layers
        use_2cta_instrs=use_2cta_instrs,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )

    return strategy


def test_cutlass_vs_manual():
    """Test CUTLASS implementation against manual PyTorch implementation."""
    print("🧪 Testing CUTLASS vs Manual PyTorch Implementation")
    print("=" * 60)

    # Configuration
    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_experts = 8
    total_tokens = 1024
    in_features = 2048
    out_features = 4096

    # Generate test data
    input_tokens = torch.randn(
        total_tokens, in_features, dtype=dtype, device=device, requires_grad=True
    )
    expert_assignments = torch.randint(0, num_experts, (total_tokens,), device=device)

    # Create strategy and layers
    if CUTLASS_AVAILABLE:
        strategy = create_cutlass_strategy()
        cutlass_layer = CUTLASSGroupedLinear(
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            strategy=strategy,
            bias=False,
            dtype=dtype,
        ).to(device)
    else:
        print("❌ CUTLASS not available, skipping CUTLASS tests")
        return

    manual_layer = PyTorchManualGroupedLinear(
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        dtype=dtype,
    ).to(device)

    # Copy weights to ensure fair comparison
    cutlass_layer.weight.data.copy_(manual_layer.weight.data)

    print("🔍 Testing Forward Pass...")

    # Forward pass
    input_manual = input_tokens.clone().detach().requires_grad_(True)
    input_cutlass = input_tokens.clone().detach().requires_grad_(True)

    output_manual = manual_layer(input_manual, expert_assignments)
    output_cutlass = cutlass_layer(input_cutlass, expert_assignments)

    # Check forward pass
    forward_diff = torch.abs(output_manual - output_cutlass).max().item()
    forward_close = torch.allclose(output_manual, output_cutlass, rtol=1e-3, atol=1e-3)

    print(f"   Forward max difference: {forward_diff:.2e}")
    print(f"   Forward outputs close: {'✓' if forward_close else '❌'}")

    print("🔍 Testing Backward Pass...")

    # Backward pass
    loss_manual = output_manual.sum()
    loss_cutlass = output_cutlass.sum()

    loss_manual.backward()
    loss_cutlass.backward()

    # Check gradients
    input_grad_diff = torch.abs(input_manual.grad - input_cutlass.grad).max().item()
    input_grad_close = torch.allclose(
        input_manual.grad, input_cutlass.grad, rtol=1e-3, atol=1e-3
    )

    weight_grad_diff = (
        torch.abs(manual_layer.weight.grad - cutlass_layer.weight.grad).max().item()
    )
    weight_grad_close = torch.allclose(
        manual_layer.weight.grad, cutlass_layer.weight.grad, rtol=1e-3, atol=1e-3
    )

    print(f"   Input grad max difference: {input_grad_diff:.2e}")
    print(f"   Input gradients close: {'✓' if input_grad_close else '❌'}")
    print(f"   Weight grad max difference: {weight_grad_diff:.2e}")
    print(f"   Weight gradients close: {'✓' if weight_grad_close else '❌'}")

    # Overall result
    all_correct = forward_close and input_grad_close and weight_grad_close
    print(f"\n🎯 Overall Result: {'✅ PASS' if all_correct else '❌ FAIL'}")

    return all_correct


def benchmark_cutlass_vs_manual():
    """Benchmark CUTLASS vs manual implementation."""
    if not CUTLASS_AVAILABLE:
        print("❌ CUTLASS not available, cannot run benchmarks")
        return

    print("\n🚀 Benchmarking CUTLASS vs Manual Implementation")
    print("=" * 60)

    # Import triton for benchmarking
    try:
        from triton.testing import do_bench
    except ImportError:
        print("❌ Triton not available, using basic timing")
        do_bench = None

    # Test configurations
    configs = [
        {
            "num_experts": 8,
            "total_tokens": 1024,
            "in_features": 2048,
            "out_features": 4096,
            "name": "Medium",
        },
        {
            "num_experts": 8,
            "total_tokens": 2048,
            "in_features": 4096,
            "out_features": 11008,
            "name": "MoE-7B",
        },
        {
            "num_experts": 64,
            "total_tokens": 4096,
            "in_features": 4096,
            "out_features": 11008,
            "name": "MoE-Large",
        },
    ]

    device = torch.device("cuda")
    dtype = torch.bfloat16

    for config in configs:
        print(
            f"\n📊 {config['name']}: {config['num_experts']} experts, {config['total_tokens']} tokens"
        )

        # Setup
        num_experts = config["num_experts"]
        total_tokens = config["total_tokens"]
        in_features = config["in_features"]
        out_features = config["out_features"]

        # Create test data
        input_tokens = torch.randn(
            total_tokens, in_features, dtype=dtype, device=device, requires_grad=True
        )
        expert_assignments = torch.randint(
            0, num_experts, (total_tokens,), device=device
        )

        # Create layers
        strategy = create_cutlass_strategy()
        cutlass_layer = CUTLASSGroupedLinear(
            num_experts, in_features, out_features, strategy, dtype
        ).to(device)
        manual_layer = PyTorchManualGroupedLinear(
            num_experts, in_features, out_features, dtype
        ).to(device)

        # Copy weights
        cutlass_layer.weight.data.copy_(manual_layer.weight.data)

        # Benchmark functions
        def manual_forward():
            return manual_layer(input_tokens, expert_assignments)

        def cutlass_forward():
            return cutlass_layer(input_tokens, expert_assignments)

        def manual_backward():
            input_clone = input_tokens.clone().detach().requires_grad_(True)
            manual_layer.zero_grad()
            output = manual_layer(input_clone, expert_assignments)
            loss = output.sum()
            loss.backward()
            return loss

        def cutlass_backward():
            input_clone = input_tokens.clone().detach().requires_grad_(True)
            cutlass_layer.zero_grad()
            output = cutlass_layer(input_clone, expert_assignments)
            loss = output.sum()
            loss.backward()
            return loss

        # Run benchmarks
        if do_bench:
            manual_fwd_time = do_bench(manual_forward, warmup=5, rep=10)
            cutlass_fwd_time = do_bench(cutlass_forward, warmup=5, rep=10)
            manual_bwd_time = do_bench(manual_backward, warmup=5, rep=10)
            cutlass_bwd_time = do_bench(cutlass_backward, warmup=5, rep=10)
        else:
            # Basic timing fallback
            import time

            # Forward timing
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                manual_forward()
            torch.cuda.synchronize()
            manual_fwd_time = (time.time() - start) / 10 * 1000

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                cutlass_forward()
            torch.cuda.synchronize()
            cutlass_fwd_time = (time.time() - start) / 10 * 1000

            # Backward timing
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                manual_backward()
            torch.cuda.synchronize()
            manual_bwd_time = (time.time() - start) / 10 * 1000

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                cutlass_backward()
            torch.cuda.synchronize()
            cutlass_bwd_time = (time.time() - start) / 10 * 1000

        # Calculate speedups
        fwd_speedup = (
            manual_fwd_time / cutlass_fwd_time if cutlass_fwd_time > 0 else float("inf")
        )
        bwd_speedup = (
            manual_bwd_time / cutlass_bwd_time if cutlass_bwd_time > 0 else float("inf")
        )

        print(
            f"   Forward:  Manual={manual_fwd_time:.2f}ms, CUTLASS={cutlass_fwd_time:.2f}ms, Speedup={fwd_speedup:.2f}x"
        )
        print(
            f"   Backward: Manual={manual_bwd_time:.2f}ms, CUTLASS={cutlass_bwd_time:.2f}ms, Speedup={bwd_speedup:.2f}x"
        )


def main():
    """Main integration test."""
    print("🎯 CUTLASS Group GEMM Integration Test")

    # Test numerical correctness
    if CUTLASS_AVAILABLE:
        test_cutlass_vs_manual()

        # Benchmark performance
        benchmark_cutlass_vs_manual()
    else:
        print("❌ CUTLASS not available. Please ensure:")
        print("   1. CUTLASS Python bindings are installed")
        print("   2. cutlass_backward_group_gemm.py is available")
        print("   3. cutlass_strategy.py is available")
        print("   4. Update import paths in this script")


if __name__ == "__main__":
    main()
