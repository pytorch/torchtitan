#!/usr/bin/env python3
"""
Comprehensive benchmark: CUTLASS Group GEMM vs PyTorch Manual Looping
Tests realistic MoE workloads with ~2048 feature dimensions.
"""

import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True

# Import triton for benchmarking
try:
    from triton.testing import do_bench

    HAS_TRITON = True
    print("✅ Triton available for benchmarking")
except ImportError:
    HAS_TRITON = False
    print("❌ Triton not available, using basic timing")


try:
    from cute_group_gemm import (
        create_cutlass_strategy,
        # CUTLASSGroupedLinear,
        # CUTLASSGroupGemmStrategy,
        StrideOptimizedCUTLASSStrategy as CUTLASSGroupGemmStrategy,
        StrideOptimizedGroupedLinear as CUTLASSGroupedLinear,
    )

    HAS_CUTLASS = True
    print("✅ CUTLASS Group GEMM available")
except ImportError:
    HAS_CUTLASS = False
    print("❌ CUTLASS Group GEMM not available")
    raise ImportError("CUTLASS modules not found. Please update the import paths.")


class PyTorchManualGroupedLinear(nn.Module):
    """Reference PyTorch implementation using manual loops for comparison"""

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

        # Same weight initialization as CUTLASS version
        self.weight = nn.Parameter(
            torch.empty(num_experts, out_features, in_features, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters to match CUTLASS version"""
        for expert_idx in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[expert_idx], a=1.41421356)

    def forward(
        self, input_tokens: torch.Tensor, expert_assignments: torch.Tensor
    ) -> torch.Tensor:
        """Manual PyTorch forward pass with explicit loops"""
        device = input_tokens.device
        total_tokens, in_features = input_tokens.shape
        out_features = self.out_features

        # Compute expert sizes and offsets
        m_sizes, m_offsets = self._compute_expert_sizes_and_offsets(expert_assignments)

        # Sort tokens by expert
        sorted_indices = torch.argsort(expert_assignments)
        sorted_tokens = input_tokens[sorted_indices]

        # Initialize output
        sorted_output = torch.zeros(
            total_tokens, out_features, dtype=self.dtype, device=device
        )

        # Manual loop over experts
        valid_sizes_cpu = m_sizes.cpu().tolist()
        valid_offsets_cpu = m_offsets.cpu().tolist()

        for expert_idx in range(self.num_experts):
            size = valid_sizes_cpu[expert_idx]
            offset = valid_offsets_cpu[expert_idx]

            if size > 0:
                # Get expert data
                expert_tokens = sorted_tokens[
                    offset : offset + size
                ]  # [size, in_features]
                expert_weight = self.weight[expert_idx]  # [out_features, in_features]

                # Forward: Y = X @ W^T
                expert_output = torch.mm(
                    expert_tokens, expert_weight.t()
                )  # [size, out_features]
                sorted_output[offset : offset + size] = expert_output

        # Restore original order
        output = torch.empty_like(sorted_output)
        output[sorted_indices] = sorted_output

        return output

    def _compute_expert_sizes_and_offsets(
        self, expert_assignments: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute expert sizes and offsets"""
        device = expert_assignments.device
        m_sizes = torch.zeros(self.num_experts, dtype=torch.int32, device=device)

        for expert_idx in range(self.num_experts):
            m_sizes[expert_idx] = (expert_assignments == expert_idx).sum()

        m_offsets = torch.cat(
            [torch.tensor([0], device=device), torch.cumsum(m_sizes, dim=0)]
        )

        return m_sizes, m_offsets


class GroupGemmBenchmark:
    """Comprehensive benchmark suite for Group GEMM implementations"""

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = torch.device(device)
        self.dtype = dtype
        self.results = []

    def create_test_data(
        self, num_experts: int, total_tokens: int, in_features: int, out_features: int
    ):
        """Create test data for benchmarking"""
        # Create input tokens
        input_tokens = torch.randn(
            total_tokens,
            in_features,
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )

        # Create expert assignments (uniform distribution)
        expert_assignments = torch.randint(
            0, num_experts, (total_tokens,), device=self.device
        )

        return input_tokens, expert_assignments

    def benchmark_forward_pass(self, config: dict, warmup: int = 5, rep: int = 20):
        """Benchmark forward pass performance"""
        print(f"\n🔍 Benchmarking Forward Pass: {config['name']}")
        print(
            f"   {config['num_experts']} experts, {config['total_tokens']} tokens, {config['in_features']}→{config['out_features']}"
        )

        # Create test data
        input_tokens, expert_assignments = self.create_test_data(
            config["num_experts"],
            config["total_tokens"],
            config["in_features"],
            config["out_features"],
        )

        # Create PyTorch manual implementation
        pytorch_layer = PyTorchManualGroupedLinear(
            config["num_experts"],
            config["in_features"],
            config["out_features"],
            self.dtype,
        ).to(self.device)

        # Create CUTLASS implementation (if available)
        cutlass_layer = None
        if HAS_CUTLASS:
            strategy = create_cutlass_strategy(
                use_2cta_instrs=True,
                mma_tiler_mn=(256, 128),
                cluster_shape_mn=(2, 2),
            )
            cutlass_layer = CUTLASSGroupedLinear(
                config["num_experts"],
                config["in_features"],
                config["out_features"],
                strategy,
                dtype=self.dtype,
            ).to(self.device)

            # Copy weights to ensure fair comparison
            cutlass_layer.weight.data.copy_(pytorch_layer.weight.data)

        # Define benchmark functions
        def pytorch_forward():
            return pytorch_layer(input_tokens, expert_assignments)

        def cutlass_forward():
            if cutlass_layer is not None:
                return cutlass_layer(input_tokens, expert_assignments)
            else:
                return pytorch_forward()  # Fallback

        # Benchmark using triton if available
        if HAS_TRITON:
            pytorch_time = do_bench(pytorch_forward, warmup=warmup, rep=rep)
            cutlass_time = (
                do_bench(cutlass_forward, warmup=warmup, rep=rep)
                if HAS_CUTLASS
                else float("inf")
            )
        else:
            pytorch_time = self._basic_benchmark(pytorch_forward, warmup, rep)
            cutlass_time = (
                self._basic_benchmark(cutlass_forward, warmup, rep)
                if HAS_CUTLASS
                else float("inf")
            )

        # Verify numerical correctness
        if HAS_CUTLASS:
            with torch.no_grad():
                pytorch_out = pytorch_forward()
                cutlass_out = cutlass_forward()
                max_diff = torch.abs(pytorch_out - cutlass_out).max().item()
                rel_diff = max_diff / pytorch_out.abs().max().item()
                correctness = rel_diff < 1e-3
        else:
            correctness = False
            max_diff = float("inf")

        speedup = pytorch_time / cutlass_time if cutlass_time < float("inf") else 0

        result = {
            "config": config["name"],
            "operation": "forward",
            "pytorch_time": pytorch_time,
            "cutlass_time": cutlass_time,
            "speedup": speedup,
            "correctness": correctness,
            "max_diff": max_diff,
        }

        print(f"   PyTorch:  {pytorch_time:.2f} ms")
        print(f"   CUTLASS:  {cutlass_time:.2f} ms")
        print(f"   Speedup:  {speedup:.2f}x")
        print(
            f"   Correct:  {'✅' if correctness else '❌'} (max diff: {max_diff:.2e})"
        )

        return result

    def benchmark_backward_pass(self, config: dict, warmup: int = 5, rep: int = 20):
        """Benchmark backward pass performance"""
        print(f"\n🔍 Benchmarking Backward Pass: {config['name']}")
        print(
            f"   {config['num_experts']} experts, {config['total_tokens']} tokens, {config['in_features']}→{config['out_features']}"
        )

        # Create test data
        input_tokens, expert_assignments = self.create_test_data(
            config["num_experts"],
            config["total_tokens"],
            config["in_features"],
            config["out_features"],
        )

        # Create PyTorch manual implementation
        pytorch_layer = PyTorchManualGroupedLinear(
            config["num_experts"],
            config["in_features"],
            config["out_features"],
            self.dtype,
        ).to(self.device)

        # Create CUTLASS implementation (if available)
        cutlass_layer = None
        if HAS_CUTLASS:
            strategy = create_cutlass_strategy(
                use_2cta_instrs=True,
                mma_tiler_mn=(256, 128),
                cluster_shape_mn=(2, 2),
            )
            cutlass_layer = CUTLASSGroupedLinear(
                config["num_experts"],
                config["in_features"],
                config["out_features"],
                strategy,
                dtype=self.dtype,
            ).to(self.device)

            # Copy weights to ensure fair comparison
            cutlass_layer.weight.data.copy_(pytorch_layer.weight.data)

        # Define benchmark functions
        def pytorch_backward():
            input_clone = input_tokens.clone().detach().requires_grad_(True)
            pytorch_layer.zero_grad()
            output = pytorch_layer(input_clone, expert_assignments)
            loss = output.sum()
            loss.backward()
            return loss

        def cutlass_backward():
            if cutlass_layer is not None:
                input_clone = input_tokens.clone().detach().requires_grad_(True)
                cutlass_layer.zero_grad()
                output = cutlass_layer(input_clone, expert_assignments)
                loss = output.sum()
                loss.backward()
                return loss
            else:
                return pytorch_backward()  # Fallback

        # Benchmark using triton if available
        if HAS_TRITON:
            pytorch_time = do_bench(pytorch_backward, warmup=warmup, rep=rep)
            cutlass_time = (
                do_bench(cutlass_backward, warmup=warmup, rep=rep)
                if HAS_CUTLASS
                else float("inf")
            )
        else:
            pytorch_time = self._basic_benchmark(pytorch_backward, warmup, rep)
            cutlass_time = (
                self._basic_benchmark(cutlass_backward, warmup, rep)
                if HAS_CUTLASS
                else float("inf")
            )

        # Verify gradient correctness
        if HAS_CUTLASS:
            # Test gradient correctness
            input_pytorch = input_tokens.clone().detach().requires_grad_(True)
            input_cutlass = input_tokens.clone().detach().requires_grad_(True)

            pytorch_layer.zero_grad()
            cutlass_layer.zero_grad()

            pytorch_out = pytorch_layer(input_pytorch, expert_assignments)
            cutlass_out = cutlass_layer(input_cutlass, expert_assignments)

            pytorch_out.sum().backward()
            cutlass_out.sum().backward()

            input_grad_diff = (
                torch.abs(input_pytorch.grad - input_cutlass.grad).max().item()
            )
            weight_grad_diff = (
                torch.abs(pytorch_layer.weight.grad - cutlass_layer.weight.grad)
                .max()
                .item()
            )

            input_rel_diff = input_grad_diff / input_pytorch.grad.abs().max().item()
            weight_rel_diff = (
                weight_grad_diff / pytorch_layer.weight.grad.abs().max().item()
            )

            correctness = input_rel_diff < 1e-2 and weight_rel_diff < 1e-2
            max_diff = max(input_grad_diff, weight_grad_diff)
        else:
            correctness = False
            max_diff = float("inf")

        speedup = pytorch_time / cutlass_time if cutlass_time < float("inf") else 0

        result = {
            "config": config["name"],
            "operation": "backward",
            "pytorch_time": pytorch_time,
            "cutlass_time": cutlass_time,
            "speedup": speedup,
            "correctness": correctness,
            "max_diff": max_diff,
        }

        print(f"   PyTorch:  {pytorch_time:.2f} ms")
        print(f"   CUTLASS:  {cutlass_time:.2f} ms")
        print(f"   Speedup:  {speedup:.2f}x")
        print(
            f"   Correct:  {'✅' if correctness else '❌'} (max diff: {max_diff:.2e})"
        )

        return result

    def _basic_benchmark(self, func, warmup: int, rep: int):
        """Basic timing fallback when triton is not available"""
        # Warmup
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()

        # Timing
        start_time = time.time()
        for _ in range(rep):
            func()
        torch.cuda.synchronize()
        end_time = time.time()

        return (end_time - start_time) / rep * 1000  # Convert to ms

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmarks across different problem sizes"""
        print("🚀 Comprehensive Group GEMM Benchmark")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Data type: {self.dtype}")
        print(f"Triton benchmarking: {'✅' if HAS_TRITON else '❌'}")
        print(f"CUTLASS available: {'✅' if HAS_CUTLASS else '❌'}")

        # Test configurations focused on ~2048 feature dimensions
        configs = [
            # Small MoE setups
            {
                "name": "Small-4E",
                "num_experts": 4,
                "total_tokens": 512,
                "in_features": 2048,
                "out_features": 2048,
            },
            {
                "name": "Small-8E",
                "num_experts": 8,
                "total_tokens": 1024,
                "in_features": 2048,
                "out_features": 2048,
            },
            # Medium MoE setups (typical 7B model dimensions)
            {
                "name": "MoE-7B-Gate",
                "num_experts": 8,
                "total_tokens": 2048,
                "in_features": 4096,
                "out_features": 11008,  # Typical MoE up_proj dimension
            },
            {
                "name": "MoE-7B-Down",
                "num_experts": 8,
                "total_tokens": 2048,
                "in_features": 11008,
                "out_features": 4096,  # Typical MoE down_proj dimension
            },
            # Large MoE setups
            {
                "name": "Large-16E",
                "num_experts": 16,
                "total_tokens": 4096,
                "in_features": 4096,
                "out_features": 11008,
            },
            {
                "name": "XLarge-32E",
                "num_experts": 32,
                "total_tokens": 4096,
                "in_features": 4096,
                "out_features": 11008,
            },
            # Very large (DeepSeek-V3 scale)
            {
                "name": "DeepSeek-64E",
                "num_experts": 64,
                "total_tokens": 8192,
                "in_features": 7168,  # DeepSeek-V3 dimensions
                "out_features": 18944,
            },
        ]

        all_results = []

        for config in configs:
            print(f"\n" + "=" * 70)
            print(f"📊 Configuration: {config['name']}")
            print(
                f"   Experts: {config['num_experts']}, Tokens: {config['total_tokens']}"
            )
            print(f"   Dimensions: {config['in_features']} → {config['out_features']}")
            print(
                f"   Problem size: ~{config['total_tokens'] * config['in_features'] * config['out_features'] / 1e6:.1f}M operations"
            )

            try:
                # Benchmark forward pass
                forward_result = self.benchmark_forward_pass(config)
                all_results.append(forward_result)

                # Benchmark backward pass
                backward_result = self.benchmark_backward_pass(config)
                all_results.append(backward_result)

            except Exception as e:
                print(f"❌ Error benchmarking {config['name']}: {e}")
                continue

        # Print summary
        self.print_benchmark_summary(all_results)

        return all_results

    def print_benchmark_summary(self, results: List[dict]):
        """Print formatted summary of benchmark results"""
        print(f"\n" + "=" * 90)
        print("📈 BENCHMARK SUMMARY")
        print("=" * 90)

        # Group results by operation type
        forward_results = [r for r in results if r["operation"] == "forward"]
        backward_results = [r for r in results if r["operation"] == "backward"]

        def print_operation_summary(op_results: List[dict], operation_name: str):
            print(f"\n🔍 {operation_name.upper()} PASS RESULTS:")
            print("-" * 90)

            header = f"{'Config':<15} {'PyTorch (ms)':<12} {'CUTLASS (ms)':<12} {'Speedup':<8} {'Correct':<8} {'Max Diff':<10}"
            print(header)
            print("-" * len(header))

            speedups = []
            for result in op_results:
                config = result["config"]
                pytorch_time = result["pytorch_time"]
                cutlass_time = result["cutlass_time"]
                speedup = result["speedup"]
                correctness = "✅" if result["correctness"] else "❌"
                max_diff = result["max_diff"]

                cutlass_str = (
                    f"{cutlass_time:.2f}" if cutlass_time < float("inf") else "N/A"
                )
                speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
                max_diff_str = f"{max_diff:.1e}" if max_diff < float("inf") else "N/A"

                print(
                    f"{config:<15} {pytorch_time:<12.2f} {cutlass_str:<12} {speedup_str:<8} {correctness:<8} {max_diff_str:<10}"
                )

                if speedup > 0:
                    speedups.append(speedup)

            if speedups:
                avg_speedup = np.mean(speedups)
                min_speedup = np.min(speedups)
                max_speedup = np.max(speedups)
                print(f"\n📊 {operation_name.title()} Speedup Summary:")
                print(f"   Average: {avg_speedup:.2f}x")
                print(f"   Range: {min_speedup:.2f}x - {max_speedup:.2f}x")

        # Print summaries for each operation
        if forward_results:
            print_operation_summary(forward_results, "forward")

        if backward_results:
            print_operation_summary(backward_results, "backward")

        # Overall summary
        print(f"\n" + "=" * 90)
        print("OVERALL PERFORMANCE ANALYSIS")
        print("=" * 90)

        all_speedups = [r["speedup"] for r in results if r["speedup"] > 0]
        if all_speedups:
            overall_avg = np.mean(all_speedups)
            print(f"📈 Average speedup across all operations: {overall_avg:.2f}x")

            if overall_avg > 2.0:
                print(
                    "🚀 Excellent performance! CUTLASS provides significant acceleration."
                )
            elif overall_avg > 1.5:
                print("✅ Good performance! CUTLASS provides solid acceleration.")
            elif overall_avg > 1.0:
                print("⚡ Moderate performance! CUTLASS provides some acceleration.")
            else:
                print("⚠️  Limited acceleration. Consider optimizing configuration.")
        else:
            print("❌ No speedup data available.")

        # Correctness summary
        correct_results = [r for r in results if r["correctness"]]
        total_results = len([r for r in results if r["cutlass_time"] < float("inf")])

        if total_results > 0:
            correctness_rate = len(correct_results) / total_results * 100
            print(
                f"✅ Numerical correctness: {len(correct_results)}/{total_results} ({correctness_rate:.1f}%)"
            )


def main():
    """Main benchmark entry point"""
    print("🧪 CUTLASS vs PyTorch Group GEMM Benchmark")
    print("Focused on realistic MoE workloads with ~2048 feature dimensions")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    # Run comprehensive benchmark
    benchmark = GroupGemmBenchmark(device="cuda", dtype=torch.bfloat16)
    results = benchmark.run_comprehensive_benchmark()

    print(f"\n🎉 Benchmark completed! Tested {len(results)} configurations.")


if __name__ == "__main__":
    main()
