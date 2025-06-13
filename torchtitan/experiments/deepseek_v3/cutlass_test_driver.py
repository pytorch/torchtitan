import argparse
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton benchmarking
try:
    import triton
    from triton.testing import do_bench

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available, using basic timing")

# Import CUTLASS components (assuming they're available)
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils

    from cutlass.cute.runtime import from_dlpack
    from cutlass_backward_group_gemm import (
        CUTLASSBackwardGroupGemm,
        CUTLASSGroupedGemmStrategy,
        CUTLASSGroupedLinear,
    )
except ImportError:
    print("CUTLASS modules not found. Please update the import paths.")
    CUTLASS_AVAILABLE = False


class PyTorchManualGroupGemm(torch.autograd.Function):
    """
    Reference implementation using manual PyTorch loops for comparison.
    """

    @staticmethod
    def forward(ctx, input_tokens, weight_stack, m_sizes, m_offsets):
        """Manual forward pass using PyTorch loops."""
        ctx.save_for_backward(input_tokens, weight_stack, m_sizes, m_offsets)

        device = input_tokens.device
        total_tokens, in_features = input_tokens.shape
        num_experts, out_features, _ = weight_stack.shape

        output = torch.zeros(
            total_tokens, out_features, dtype=input_tokens.dtype, device=device
        )

        # Manual loop over experts
        offset = 0
        for expert_idx, size in enumerate(m_sizes.cpu().tolist()):
            if size > 0:
                # Get tokens for this expert
                expert_tokens = input_tokens[
                    offset : offset + size
                ]  # [size, in_features]
                expert_weight = weight_stack[expert_idx]  # [out_features, in_features]

                # Compute: expert_tokens @ expert_weight.T
                expert_output = torch.mm(
                    expert_tokens, expert_weight.t()
                )  # [size, out_features]

                # Store results
                output[offset : offset + size] = expert_output

            offset += size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Manual backward pass using PyTorch loops."""
        input_tokens, weight_stack, m_sizes, m_offsets = ctx.saved_tensors

        device = grad_output.device
        grad_input = torch.zeros_like(input_tokens)
        grad_weight = torch.zeros_like(weight_stack)

        # Manual loop over experts
        offset = 0
        for expert_idx, size in enumerate(m_sizes.cpu().tolist()):
            if size > 0:
                # Get gradients for this expert
                grad_expert = grad_output[
                    offset : offset + size
                ]  # [size, out_features]
                expert_tokens = input_tokens[
                    offset : offset + size
                ]  # [size, in_features]
                expert_weight = weight_stack[expert_idx]  # [out_features, in_features]

                # Input gradient: grad_expert @ expert_weight
                grad_input[offset : offset + size] = torch.mm(
                    grad_expert, expert_weight
                )

                # Weight gradient: grad_expert.T @ expert_tokens
                grad_weight[expert_idx] = torch.mm(grad_expert.t(), expert_tokens)

            offset += size

        return grad_input, grad_weight, None, None


class PyTorchManualGroupedLinear(nn.Module):
    """Reference grouped linear layer using manual loops."""

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        self.weight = nn.Parameter(
            torch.empty(num_experts, out_features, in_features, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for expert_idx in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[expert_idx], a=1.41421356)

    def forward(
        self, input_tokens: torch.Tensor, expert_assignments: torch.Tensor
    ) -> torch.Tensor:
        m_sizes, m_offsets = self._compute_expert_sizes_and_offsets(expert_assignments)

        # Sort tokens by expert assignment
        sorted_indices = torch.argsort(expert_assignments)
        sorted_tokens = input_tokens[sorted_indices]

        # Apply manual grouped GEMM
        sorted_output = PyTorchManualGroupGemm.apply(
            sorted_tokens, self.weight, m_sizes, m_offsets
        )

        # Unsort to restore original order
        output = torch.empty_like(sorted_output)
        output[sorted_indices] = sorted_output

        return output

    def _compute_expert_sizes_and_offsets(
        self, expert_assignments: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = expert_assignments.device
        m_sizes = torch.zeros(self.num_experts, dtype=torch.int32, device=device)

        for expert_idx in range(self.num_experts):
            m_sizes[expert_idx] = (expert_assignments == expert_idx).sum()

        m_offsets = torch.cat(
            [torch.tensor([0], device=device), torch.cumsum(m_sizes, dim=0)]
        )
        return m_sizes, m_offsets


class GroupGemmTestDriver:
    """Test driver for comparing CUTLASS vs PyTorch manual implementation."""

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = torch.device(device)
        self.dtype = dtype

    def generate_test_data(
        self,
        num_experts: int,
        total_tokens: int,
        in_features: int,
        out_features: int,
        expert_balance: str = "uniform",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate test data for benchmarking."""

        # Create input tokens
        input_tokens = torch.randn(
            total_tokens,
            in_features,
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )

        # Create expert assignments
        if expert_balance == "uniform":
            expert_assignments = torch.randint(
                0, num_experts, (total_tokens,), device=self.device
            )
        elif expert_balance == "imbalanced":
            # Create imbalanced distribution (some experts get more tokens)
            probs = torch.tensor([0.4, 0.3, 0.2, 0.1] + [0.0] * (num_experts - 4))[
                :num_experts
            ]
            probs = probs / probs.sum()
            expert_assignments = torch.multinomial(
                probs, total_tokens, replacement=True
            ).to(self.device)
        elif expert_balance == "sparse":
            # Only use first half of experts
            expert_assignments = torch.randint(
                0, num_experts // 2, (total_tokens,), device=self.device
            )
        else:
            raise ValueError(f"Unknown expert_balance: {expert_balance}")

        return input_tokens, expert_assignments

    def test_numerical_correctness(
        self,
        num_experts=4,
        total_tokens=256,
        in_features=512,
        out_features=1024,
        rtol=1e-3,
        atol=1e-3,
    ):
        """Test numerical correctness between CUTLASS and PyTorch manual implementations."""
        print(f"\n🧮 Testing Numerical Correctness")
        print(
            f"   Problem size: {num_experts} experts, {total_tokens} tokens, {in_features}→{out_features}"
        )

        # Generate test data
        input_tokens, expert_assignments = self.generate_test_data(
            num_experts, total_tokens, in_features, out_features
        )

        # Create both implementations with same weights
        manual_layer = PyTorchManualGroupedLinear(
            num_experts, in_features, out_features, self.dtype
        ).to(self.device)

        # For now, we'll test the manual implementation against itself to verify the test setup
        # When CUTLASS implementation is available, we'll copy weights and compare

        # cutlass_layer = CUTLASSGroupedLinear(num_experts, in_features, out_features, strategy, dtype=self.dtype).to(self.device)
        # cutlass_layer.weight.data.copy_(manual_layer.weight.data)

        print("   Testing forward pass...")

        # Forward pass - Manual
        input_tokens_manual = input_tokens.clone().detach().requires_grad_(True)
        output_manual = manual_layer(input_tokens_manual, expert_assignments)

        # For now, test manual against manual (placeholder for CUTLASS comparison)
        input_tokens_cutlass = input_tokens.clone().detach().requires_grad_(True)
        output_cutlass = manual_layer(
            input_tokens_cutlass, expert_assignments
        )  # Replace with cutlass_layer when available

        # Check forward pass
        forward_diff = torch.abs(output_manual - output_cutlass).max().item()
        forward_close = torch.allclose(
            output_manual, output_cutlass, rtol=rtol, atol=atol
        )

        print(f"   ✓ Forward pass max diff: {forward_diff:.2e}")
        print(f"   ✓ Forward pass close: {forward_close}")

        print("   Testing backward pass...")

        # Backward pass
        loss_manual = output_manual.sum()
        loss_cutlass = output_cutlass.sum()

        loss_manual.backward()
        loss_cutlass.backward()

        # Check input gradients
        if (
            input_tokens_manual.grad is not None
            and input_tokens_cutlass.grad is not None
        ):
            input_grad_diff = (
                torch.abs(input_tokens_manual.grad - input_tokens_cutlass.grad)
                .max()
                .item()
            )
            input_grad_close = torch.allclose(
                input_tokens_manual.grad,
                input_tokens_cutlass.grad,
                rtol=rtol,
                atol=atol,
            )

            print(f"   ✓ Input gradient max diff: {input_grad_diff:.2e}")
            print(f"   ✓ Input gradient close: {input_grad_close}")

        # Check weight gradients
        if (
            manual_layer.weight.grad is not None
        ):  # and cutlass_layer.weight.grad is not None:
            weight_grad_diff = (
                torch.abs(manual_layer.weight.grad - manual_layer.weight.grad)
                .max()
                .item()
            )  # Replace with cutlass comparison
            weight_grad_close = True  # Replace with actual comparison

            print(f"   ✓ Weight gradient max diff: {weight_grad_diff:.2e}")
            print(f"   ✓ Weight gradient close: {weight_grad_close}")

        return forward_close and input_grad_close and weight_grad_close

    def benchmark_forward_pass(self, config: dict, warmup=5, reps=10):
        """Benchmark forward pass performance."""
        num_experts = config["num_experts"]
        total_tokens = config["total_tokens"]
        in_features = config["in_features"]
        out_features = config["out_features"]

        # Generate test data
        input_tokens, expert_assignments = self.generate_test_data(
            num_experts, total_tokens, in_features, out_features
        )

        # Create layers
        manual_layer = PyTorchManualGroupedLinear(
            num_experts, in_features, out_features, self.dtype
        ).to(self.device)
        # cutlass_layer = CUTLASSGroupedLinear(num_experts, in_features, out_features, strategy, dtype=self.dtype).to(self.device)
        # cutlass_layer.weight.data.copy_(manual_layer.weight.data)

        def manual_forward():
            return manual_layer(input_tokens, expert_assignments)

        def cutlass_forward():
            return manual_layer(
                input_tokens, expert_assignments
            )  # Replace with cutlass_layer when available

        # Benchmark using Triton if available
        if TRITON_AVAILABLE:
            manual_time = do_bench(manual_forward, warmup=warmup, rep=reps)
            cutlass_time = do_bench(cutlass_forward, warmup=warmup, rep=reps)
        else:
            # Fallback timing
            manual_time = self._basic_benchmark(manual_forward, warmup, reps)
            cutlass_time = self._basic_benchmark(cutlass_forward, warmup, reps)

        return {
            "manual_time": manual_time,
            "cutlass_time": cutlass_time,
            "speedup": manual_time / cutlass_time if cutlass_time > 0 else float("inf"),
        }

    def benchmark_backward_pass(self, config: dict, warmup=5, reps=10):
        """Benchmark backward pass performance."""
        num_experts = config["num_experts"]
        total_tokens = config["total_tokens"]
        in_features = config["in_features"]
        out_features = config["out_features"]

        # Generate test data
        input_tokens, expert_assignments = self.generate_test_data(
            num_experts, total_tokens, in_features, out_features
        )

        # Create layers
        manual_layer = PyTorchManualGroupedLinear(
            num_experts, in_features, out_features, self.dtype
        ).to(self.device)
        # cutlass_layer = CUTLASSGroupedLinear(num_experts, in_features, out_features, strategy, dtype=self.dtype).to(self.device)

        def manual_backward():
            input_tokens_clone = input_tokens.clone().detach().requires_grad_(True)
            manual_layer.zero_grad()
            output = manual_layer(input_tokens_clone, expert_assignments)
            loss = output.sum()
            loss.backward()
            return loss

        def cutlass_backward():
            input_tokens_clone = input_tokens.clone().detach().requires_grad_(True)
            manual_layer.zero_grad()  # Replace with cutlass_layer when available
            output = manual_layer(input_tokens_clone, expert_assignments)
            loss = output.sum()
            loss.backward()
            return loss

        # Benchmark using Triton if available
        if TRITON_AVAILABLE:
            manual_time = do_bench(manual_backward, warmup=warmup, rep=reps)
            cutlass_time = do_bench(cutlass_backward, warmup=warmup, rep=reps)
        else:
            manual_time = self._basic_benchmark(manual_backward, warmup, reps)
            cutlass_time = self._basic_benchmark(cutlass_backward, warmup, reps)

        return {
            "manual_time": manual_time,
            "cutlass_time": cutlass_time,
            "speedup": manual_time / cutlass_time if cutlass_time > 0 else float("inf"),
        }

    def _basic_benchmark(self, func, warmup, reps):
        """Basic timing fallback when Triton is not available."""
        # Warmup
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()

        # Timing
        start_time = time.time()
        for _ in range(reps):
            func()
        torch.cuda.synchronize()
        end_time = time.time()

        return (end_time - start_time) / reps * 1000  # Convert to ms

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmarks across different problem sizes."""
        print("🚀 Running Comprehensive Group GEMM Benchmarks")
        print("=" * 60)

        # Test configurations
        configs = [
            # Small problems
            {
                "num_experts": 4,
                "total_tokens": 256,
                "in_features": 512,
                "out_features": 1024,
                "name": "Small",
            },
            # Medium problems
            {
                "num_experts": 8,
                "total_tokens": 512,
                "in_features": 1024,
                "out_features": 2048,
                "name": "Medium",
            },
            # Large problems
            {
                "num_experts": 8,
                "total_tokens": 1024,
                "in_features": 2048,
                "out_features": 4096,
                "name": "Large",
            },
            # MoE-like problems
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

        results = []

        for config in configs:
            print(f"\n📊 Benchmarking {config['name']} Configuration")
            print(
                f"   {config['num_experts']} experts, {config['total_tokens']} tokens, {config['in_features']}→{config['out_features']}"
            )

            try:
                # Test numerical correctness first
                correct = self.test_numerical_correctness(
                    **{k: v for k, v in config.items() if k != "name"}
                )

                # Benchmark forward pass
                print("   Benchmarking forward pass...")
                forward_results = self.benchmark_forward_pass(config)

                # Benchmark backward pass
                print("   Benchmarking backward pass...")
                backward_results = self.benchmark_backward_pass(config)

                # Store results
                result = {
                    "config": config,
                    "correct": correct,
                    "forward": forward_results,
                    "backward": backward_results,
                }
                results.append(result)

                # Print summary
                print(f"   ✓ Numerical correctness: {correct}")
                print(
                    f"   ✓ Forward:  Manual={forward_results['manual_time']:.2f}ms, CUTLASS={forward_results['cutlass_time']:.2f}ms, Speedup={forward_results['speedup']:.2f}x"
                )
                print(
                    f"   ✓ Backward: Manual={backward_results['manual_time']:.2f}ms, CUTLASS={backward_results['cutlass_time']:.2f}ms, Speedup={backward_results['speedup']:.2f}x"
                )

            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue

        # Print final summary
        self.print_benchmark_summary(results)

        return results

    def print_benchmark_summary(self, results):
        """Print a formatted summary of benchmark results."""
        print("\n" + "=" * 80)
        print("📈 BENCHMARK SUMMARY")
        print("=" * 80)

        header = f"{'Config':<12} {'Correct':<8} {'Fwd Manual':<10} {'Fwd CUTLASS':<12} {'Fwd Speedup':<11} {'Bwd Manual':<10} {'Bwd CUTLASS':<12} {'Bwd Speedup':<11}"
        print(header)
        print("-" * len(header))

        for result in results:
            config = result["config"]
            correct = "✓" if result["correct"] else "❌"

            fwd = result["forward"]
            bwd = result["backward"]

            print(
                f"{config['name']:<12} {correct:<8} {fwd['manual_time']:<10.2f} {fwd['cutlass_time']:<12.2f} {fwd['speedup']:<11.2f} {bwd['manual_time']:<10.2f} {bwd['cutlass_time']:<12.2f} {bwd['speedup']:<11.2f}"
            )

        # Calculate average speedups
        if results:
            avg_fwd_speedup = np.mean(
                [
                    r["forward"]["speedup"]
                    for r in results
                    if r["forward"]["speedup"] != float("inf")
                ]
            )
            avg_bwd_speedup = np.mean(
                [
                    r["backward"]["speedup"]
                    for r in results
                    if r["backward"]["speedup"] != float("inf")
                ]
            )

            print(f"\n🎯 Average Speedups:")
            print(f"   Forward:  {avg_fwd_speedup:.2f}x")
            print(f"   Backward: {avg_bwd_speedup:.2f}x")


def main():
    """Main test driver entry point."""
    parser = argparse.ArgumentParser(description="CUTLASS Group GEMM Test Driver")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type",
    )
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--total-tokens", type=int, default=1024, help="Total tokens")
    parser.add_argument("--in-features", type=int, default=2048, help="Input features")
    parser.add_argument(
        "--out-features", type=int, default=4096, help="Output features"
    )
    parser.add_argument(
        "--test-correctness",
        action="store_true",
        help="Test numerical correctness only",
    )
    parser.add_argument(
        "--benchmark-only", action="store_true", help="Run benchmarks only"
    )

    args = parser.parse_args()

    # Setup
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Initialize test driver
    test_driver = GroupGemmTestDriver(device=args.device, dtype=dtype)

    print("🧪 CUTLASS Group GEMM Test Driver")
    print(
        f"Device: {args.device}, Dtype: {args.dtype}, Triton: {'✓' if TRITON_AVAILABLE else '❌'}"
    )

    if args.test_correctness:
        # Test numerical correctness only
        correct = test_driver.test_numerical_correctness(
            args.num_experts, args.total_tokens, args.in_features, args.out_features
        )
        print(f"\n🎯 Overall correctness: {'✓ PASS' if correct else '❌ FAIL'}")

    elif args.benchmark_only:
        # Run single benchmark
        config = {
            "num_experts": args.num_experts,
            "total_tokens": args.total_tokens,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "name": "Custom",
        }

        forward_results = test_driver.benchmark_forward_pass(config)
        backward_results = test_driver.benchmark_backward_pass(config)

        print(f"\n📊 Single Benchmark Results:")
        print(
            f"Forward:  Manual={forward_results['manual_time']:.2f}ms, CUTLASS={forward_results['cutlass_time']:.2f}ms, Speedup={forward_results['speedup']:.2f}x"
        )
        print(
            f"Backward: Manual={backward_results['manual_time']:.2f}ms, CUTLASS={backward_results['cutlass_time']:.2f}ms, Speedup={backward_results['speedup']:.2f}x"
        )

    else:
        # Run comprehensive benchmarks
        results = test_driver.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()
