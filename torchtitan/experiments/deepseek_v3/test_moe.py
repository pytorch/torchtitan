#!/usr/bin/env python3
"""
Simple MoE Benchmark using CUTLASSGroupedGemmStrategy

This benchmark creates a realistic MoE layer and compares:
1. CUTLASSGroupedGemmStrategy (our optimized approach)
2. Manual looping through experts (baseline)
3. PyTorch grouped_mm (if available)

Uses Triton's do_bench for accurate GPU timing, with CUDA events as fallback.
"""

import gc
import math
import time
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import timing utilities
try:
    import triton.testing

    HAS_TRITON_BENCH = True
    print("✓ Triton do_bench available for accurate timing")
except ImportError:
    HAS_TRITON_BENCH = False
    print("⚠️  Triton do_bench not available, using CUDA events")

# Import CUTLASS components
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils

    from cutlass.cute.runtime import from_dlpack
    from group_gemms import (
        CUTLASSGroupedGemmStrategy,
        GroupGEMMStrategy,
        ManualLoopGroupGEMM,
    )

    HAS_CUTLASS = True
    print("✓ CUTLASS and strategies imported successfully")
except ImportError as e:
    HAS_CUTLASS = False
    print(f"✗ Import failed: {e}")
    print("Using PyTorch fallback implementations only")


class SimpleMoELayer(nn.Module):
    """
    Simplified MoE layer for benchmarking

    Architecture:
    - Router: Linear layer that outputs expert probabilities
    - Experts: Each expert has gate_proj -> activation -> up_proj -> down_proj
    - Top-K routing: Each token is assigned to top_k experts
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.dtype = dtype

        # Router network
        self.router = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)

        # Expert weights - stored as [num_experts, out_dim, in_dim] for our strategy
        self.gate_weights = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size, dtype=dtype)
            * math.sqrt(2.0 / (hidden_size + intermediate_size))
        )

        self.up_weights = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size, dtype=dtype)
            * math.sqrt(2.0 / (hidden_size + intermediate_size))
        )

        self.down_weights = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype)
            * math.sqrt(2.0 / (hidden_size + intermediate_size))
        )

        # Mock parameter access for strategies
        self._expert_params = {
            "gate_proj_weight": self.gate_weights,
            "up_proj_weight": self.up_weights,
            "down_proj_weight": self.down_weights,
        }

    def get_parameter(self, name: str):
        """Strategy interface for accessing parameters"""
        return self._expert_params.get(name)

    def silu_activation(self, x):
        """SiLU activation function"""
        return x * torch.sigmoid(x)

    def route_tokens(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts using top-k selection

        Returns:
            contig_tokens: Tokens arranged contiguously by expert assignment
            m_sizes: Number of tokens assigned to each expert
            m_offsets: Cumulative token offsets for each expert (as torch.Tensor)
            routing_weights: Weights for combining expert outputs
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(
            -1, hidden_size
        )  # [total_tokens, hidden_size]

        # Get routing scores
        router_logits = self.router(hidden_states)  # [total_tokens, num_experts]
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1)

        # For simplicity, assign each token to its top-1 expert
        # In practice, you'd handle top-k routing with load balancing
        top_expert = selected_experts[:, 0]  # [total_tokens]
        token_weights = routing_weights[:, 0]  # [total_tokens]

        # Count tokens per expert
        m_sizes = [0] * self.num_experts
        expert_tokens = [[] for _ in range(self.num_experts)]
        expert_weights = [[] for _ in range(self.num_experts)]

        for token_idx, expert_idx in enumerate(top_expert):
            expert_idx = expert_idx.item()
            m_sizes[expert_idx] += 1
            expert_tokens[expert_idx].append(token_idx)
            expert_weights[expert_idx].append(token_weights[token_idx])

        # Create contiguous token arrangement
        contig_tokens = []
        token_to_output_pos = {}
        current_pos = 0

        for expert_idx in range(self.num_experts):
            if m_sizes[expert_idx] > 0:
                expert_token_indices = expert_tokens[expert_idx]
                expert_hidden_states = hidden_states[expert_token_indices]
                contig_tokens.append(expert_hidden_states)

                # Track where each token should go in the output
                for local_pos, global_token_idx in enumerate(expert_token_indices):
                    token_to_output_pos[global_token_idx] = current_pos + local_pos

                current_pos += m_sizes[expert_idx]

        if contig_tokens:
            contig_tokens = torch.cat(contig_tokens, dim=0)
        else:
            contig_tokens = torch.empty(
                0, hidden_size, dtype=self.dtype, device=hidden_states.device
            )

        # Create offsets as torch.Tensor (required for PyTorch grouped_mm)
        m_offsets_list = []
        cumsum = 0
        for size in m_sizes:
            cumsum += size
            m_offsets_list.append(cumsum)

        m_offsets = torch.tensor(
            m_offsets_list, dtype=torch.int32, device=hidden_states.device
        )

        # Store routing info for output reconstruction
        self._routing_info = {
            "token_to_output_pos": token_to_output_pos,
            "expert_weights": expert_weights,
            "original_shape": (batch_size, seq_len, hidden_size),
        }

        return contig_tokens, m_sizes, m_offsets, token_weights

    def reconstruct_output(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        """Reconstruct output tensor from expert results"""
        routing_info = self._routing_info
        batch_size, seq_len, hidden_size = routing_info["original_shape"]
        total_tokens = batch_size * seq_len

        # Initialize output
        output = torch.zeros(
            total_tokens, hidden_size, dtype=self.dtype, device=expert_outputs.device
        )

        # Place expert outputs back in original token positions
        current_pos = 0
        for expert_idx in range(self.num_experts):
            if len(routing_info["expert_weights"][expert_idx]) > 0:
                expert_size = len(routing_info["expert_weights"][expert_idx])
                expert_output = expert_outputs[current_pos : current_pos + expert_size]
                expert_weight_list = routing_info["expert_weights"][expert_idx]

                # Apply expert-specific routing weights
                for local_pos, (global_token_idx, weight) in enumerate(
                    zip(
                        [
                            k
                            for k, v in routing_info["token_to_output_pos"].items()
                            if current_pos <= v < current_pos + expert_size
                        ],
                        expert_weight_list,
                    )
                ):
                    output[global_token_idx] = expert_output[local_pos] * weight

                current_pos += expert_size

        return output.view(batch_size, seq_len, hidden_size)


class MoEBenchmark:
    """Benchmark harness for MoE implementations"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        # Set timing method
        if HAS_TRITON_BENCH:
            self.timing_method = "triton"
        else:
            self.timing_method = "cuda_events"

        # Get GPU architecture info
        self.gpu_arch_info = self._get_gpu_arch_info()

    def _get_gpu_arch_info(self):
        """Get GPU architecture information"""
        if not torch.cuda.is_available():
            return {
                "compute_capability": None,
                "is_hopper": False,
                "is_blackwell": False,
            }

        cap = torch.cuda.get_device_capability()
        compute_capability = f"{cap[0]}.{cap[1]}"

        return {
            "compute_capability": compute_capability,
            "is_hopper": cap[0] == 9,
            "is_blackwell": cap[0] == 10,
        }

    def _is_pytorch_grouped_mm_available(self):
        """Check if PyTorch grouped_mm is available and supported on this architecture"""
        if not hasattr(torch, "_grouped_mm"):
            return False

        # Currently grouped_mm is only optimized for Hopper
        if self.gpu_arch_info["is_blackwell"]:
            return False

        return True

    def create_test_data(
        self, batch_size: int, seq_len: int, hidden_size: int
    ) -> torch.Tensor:
        """Create test input data"""
        return (
            torch.randn(
                batch_size, seq_len, hidden_size, dtype=self.dtype, device=self.device
            )
            * 0.02
        )

    def benchmark_cutlass_strategy(
        self,
        moe_layer: SimpleMoELayer,
        hidden_states: torch.Tensor,
        iterations: int = 10,
    ) -> Tuple[torch.Tensor, float]:
        """Benchmark CUTLASS grouped GEMM strategy"""
        strategy = CUTLASSGroupedGemmStrategy(moe_layer.silu_activation)

        # Warmup
        for _ in range(3):
            contig_tokens, m_sizes, m_offsets, _ = moe_layer.route_tokens(hidden_states)
            if sum(m_sizes) > 0:
                expert_outputs = strategy.execute(
                    contig_tokens, m_sizes, m_offsets, moe_layer
                )

        # Use triton.do_bench if available
        if self.timing_method == "triton":

            def bench_fn():
                contig_tokens, m_sizes, m_offsets, _ = moe_layer.route_tokens(
                    hidden_states
                )
                if sum(m_sizes) > 0:
                    expert_outputs = strategy.execute(
                        contig_tokens, m_sizes, m_offsets, moe_layer
                    )
                else:
                    expert_outputs = torch.empty(
                        0, moe_layer.hidden_size, dtype=self.dtype, device=self.device
                    )
                torch.cuda.synchronize()

            ms_times = triton.testing.do_bench(bench_fn, warmup=3, rep=iterations)
            avg_time = ms_times
        else:
            # Fall back to CUDA events
            torch.cuda.synchronize()
            start_time = time.time()

            for _ in range(iterations):
                contig_tokens, m_sizes, m_offsets, _ = moe_layer.route_tokens(
                    hidden_states
                )
                if sum(m_sizes) > 0:
                    expert_outputs = strategy.execute(
                        contig_tokens, m_sizes, m_offsets, moe_layer
                    )
                else:
                    expert_outputs = torch.empty(
                        0, moe_layer.hidden_size, dtype=self.dtype, device=self.device
                    )

            torch.cuda.synchronize()
            end_time = time.time()
            avg_time = (end_time - start_time) / iterations * 1000  # ms

        # Final forward pass for output
        contig_tokens, m_sizes, m_offsets, _ = moe_layer.route_tokens(hidden_states)
        if sum(m_sizes) > 0:
            expert_outputs = strategy.execute(
                contig_tokens, m_sizes, m_offsets, moe_layer
            )
        else:
            expert_outputs = torch.empty(
                0, moe_layer.hidden_size, dtype=self.dtype, device=self.device
            )

        final_output = moe_layer.reconstruct_output(expert_outputs)

        return final_output, avg_time

    def benchmark_manual_strategy(
        self,
        moe_layer: SimpleMoELayer,
        hidden_states: torch.Tensor,
        iterations: int = 10,
    ) -> Tuple[torch.Tensor, float]:
        """Benchmark manual loop strategy"""
        strategy = ManualLoopGroupGEMM(moe_layer.silu_activation)

        # Warmup
        for _ in range(3):
            contig_tokens, m_sizes, m_offsets, _ = moe_layer.route_tokens(hidden_states)
            if sum(m_sizes) > 0:
                expert_outputs = strategy.execute(
                    contig_tokens, m_sizes, m_offsets, moe_layer
                )

        # Use triton.do_bench if available
        if self.timing_method == "triton":

            def bench_fn():
                contig_tokens, m_sizes, m_offsets, _ = moe_layer.route_tokens(
                    hidden_states
                )
                if sum(m_sizes) > 0:
                    expert_outputs = strategy.execute(
                        contig_tokens, m_sizes, m_offsets, moe_layer
                    )
                else:
                    expert_outputs = torch.empty(
                        0, moe_layer.hidden_size, dtype=self.dtype, device=self.device
                    )
                torch.cuda.synchronize()

            ms_times = triton.testing.do_bench(bench_fn, warmup=3, rep=iterations)
            avg_time = ms_times
        else:
            # Fall back to CUDA events
            torch.cuda.synchronize()
            start_time = time.time()

            for _ in range(iterations):
                contig_tokens, m_sizes, m_offsets, _ = moe_layer.route_tokens(
                    hidden_states
                )
                if sum(m_sizes) > 0:
                    expert_outputs = strategy.execute(
                        contig_tokens, m_sizes, m_offsets, moe_layer
                    )
                else:
                    expert_outputs = torch.empty(
                        0, moe_layer.hidden_size, dtype=self.dtype, device=self.device
                    )

            torch.cuda.synchronize()
            end_time = time.time()
            avg_time = (end_time - start_time) / iterations * 1000  # ms

        # Final forward pass for output
        contig_tokens, m_sizes, m_offsets, _ = moe_layer.route_tokens(hidden_states)
        if sum(m_sizes) > 0:
            expert_outputs = strategy.execute(
                contig_tokens, m_sizes, m_offsets, moe_layer
            )
        else:
            expert_outputs = torch.empty(
                0, moe_layer.hidden_size, dtype=self.dtype, device=self.device
            )

        final_output = moe_layer.reconstruct_output(expert_outputs)

        return final_output, avg_time

    def benchmark_pytorch_grouped_mm(
        self,
        moe_layer: SimpleMoELayer,
        hidden_states: torch.Tensor,
        iterations: int = 10,
    ) -> Tuple[torch.Tensor, float]:
        """Benchmark PyTorch _grouped_mm if available"""
        if not hasattr(torch, "_grouped_mm"):
            return None, float("inf")

        # Warmup and benchmark similar to other methods
        for _ in range(3):
            contig_tokens, m_sizes, m_offsets, _ = moe_layer.route_tokens(hidden_states)
            if sum(m_sizes) > 0:
                # Simulate grouped_mm operations
                self._pytorch_grouped_mm_forward(moe_layer, contig_tokens, m_offsets)

        torch.cuda.synchronize()

        start_time = time.time()

        for _ in range(iterations):
            contig_tokens, m_sizes, m_offsets, _ = moe_layer.route_tokens(hidden_states)
            if sum(m_sizes) > 0:
                expert_outputs = self._pytorch_grouped_mm_forward(
                    moe_layer, contig_tokens, m_offsets
                )
            else:
                expert_outputs = torch.empty(
                    0, moe_layer.hidden_size, dtype=self.dtype, device=self.device
                )

        torch.cuda.synchronize()
        end_time = time.time()

        # Final forward pass
        contig_tokens, m_sizes, m_offsets, _ = moe_layer.route_tokens(hidden_states)
        if sum(m_sizes) > 0:
            expert_outputs = self._pytorch_grouped_mm_forward(
                moe_layer, contig_tokens, m_offsets
            )
        else:
            expert_outputs = torch.empty(
                0, moe_layer.hidden_size, dtype=self.dtype, device=self.device
            )

        final_output = moe_layer.reconstruct_output(expert_outputs)
        avg_time = (end_time - start_time) / iterations * 1000  # ms

        return final_output, avg_time

    def _pytorch_grouped_mm_forward(
        self, moe_layer: SimpleMoELayer, tokens: torch.Tensor, m_offsets: List[int]
    ) -> torch.Tensor:
        """Simulate PyTorch grouped_mm forward pass"""
        # Gate and up projections
        gate_proj = torch._grouped_mm(
            tokens,
            moe_layer.gate_weights.transpose(-2, -1),
            m_offsets,
            out_dtype=self.dtype,
        )
        up_proj = torch._grouped_mm(
            tokens,
            moe_layer.up_weights.transpose(-2, -1),
            m_offsets,
            out_dtype=self.dtype,
        )

        # Apply activation and combine
        hidden_outputs = moe_layer.silu_activation(gate_proj) * up_proj

        # Down projection
        final_outputs = torch._grouped_mm(
            hidden_outputs,
            moe_layer.down_weights.transpose(-2, -1),
            m_offsets,
            out_dtype=self.dtype,
        )

        return final_outputs

    def validate_outputs(
        self, output1: torch.Tensor, output2: torch.Tensor, tolerance: float = 1e-1
    ) -> bool:
        """Validate that two outputs are close"""
        if output1.shape != output2.shape:
            print(f"Shape mismatch: {output1.shape} vs {output2.shape}")
            return False

        diff = torch.abs(output1 - output2)
        max_diff = torch.max(diff).item()
        rel_error = torch.norm(diff).item() / torch.norm(output1).item()

        print(f"    Max diff: {max_diff:.6f}, Rel error: {rel_error:.6f}")

        return max_diff < tolerance and rel_error < tolerance

    def run_benchmark_suite(self):
        """Run comprehensive benchmark suite"""
        print("=" * 80)
        print("MoE Benchmark Suite")
        print("=" * 80)

        # Test configurations
        configs = [
            {
                "name": "Small MoE",
                "batch_size": 4,
                "seq_len": 512,
                "hidden_size": 512,
                "intermediate_size": 1024,
                "num_experts": 8,
                "top_k": 2,
            },
            {
                "name": "Medium MoE",
                "batch_size": 8,
                "seq_len": 1024,
                "hidden_size": 1024,
                "intermediate_size": 2048,
                "num_experts": 16,
                "top_k": 2,
            },
            {
                "name": "Large MoE",
                "batch_size": 16,
                "seq_len": 2048,
                "hidden_size": 2048,
                "intermediate_size": 4096,
                "num_experts": 32,
                "top_k": 2,
            },
            {
                "name": "X-Large MoE",
                "batch_size": 8,
                "seq_len": 4096,
                "hidden_size": 2048,
                "intermediate_size": 4096,
                "num_experts": 256,
                "top_k": 6,
            },
        ]

        all_passed = True

        for config in configs:
            print(f"\n" + "=" * 60)
            print(f"Benchmarking: {config['name']}")
            print(
                f"  Shape: [{config['batch_size']}, {config['seq_len']}, {config['hidden_size']}]"
            )
            print(f"  Experts: {config['num_experts']}, Top-K: {config['top_k']}")
            print(f"  Intermediate size: {config['intermediate_size']}")
            print("=" * 60)

            try:
                # Create test data
                hidden_states = self.create_test_data(
                    config["batch_size"], config["seq_len"], config["hidden_size"]
                )

                # Create MoE layer
                moe_layer = SimpleMoELayer(
                    hidden_size=config["hidden_size"],
                    intermediate_size=config["intermediate_size"],
                    num_experts=config["num_experts"],
                    top_k=config["top_k"],
                    dtype=self.dtype,
                ).to(self.device)

                # Show routing statistics
                with torch.no_grad():
                    contig_tokens, m_sizes, m_offsets, _ = moe_layer.route_tokens(
                        hidden_states
                    )
                    active_experts = sum(1 for s in m_sizes if s > 0)
                    total_tokens = sum(m_sizes)
                    print(
                        f"  Routing: {total_tokens} tokens across {active_experts}/{config['num_experts']} experts"
                    )
                    print(f"  Expert loads: {m_sizes}")
                    print(f"  Offsets (tensor): {m_offsets.tolist()}")

                results = {}

                # Benchmark manual strategy
                print(f"\n  Benchmarking Manual Loop Strategy...")
                manual_output, manual_time = self.benchmark_manual_strategy(
                    moe_layer, hidden_states
                )
                results["manual"] = (manual_output, manual_time)
                print(f"    Time: {manual_time:.3f} ms")

                # Benchmark CUTLASS strategy
                if HAS_CUTLASS:
                    print(f"\n  Benchmarking CUTLASS Strategy...")
                    cutlass_output, cutlass_time = self.benchmark_cutlass_strategy(
                        moe_layer, hidden_states
                    )
                    results["cutlass"] = (cutlass_output, cutlass_time)
                    print(f"    Time: {cutlass_time:.3f} ms")

                    # Validate
                    print(f"\n  Validating CUTLASS vs Manual...")
                    if self.validate_outputs(manual_output, cutlass_output):
                        print(f"    ✓ Validation passed")
                        speedup = manual_time / cutlass_time
                        print(f"    Speedup: {speedup:.2f}x")

                        if speedup > 1.1:
                            print(f"    🚀 CUTLASS is faster!")
                        elif speedup < 0.9:
                            print(f"    ⚠️  CUTLASS is slower - may indicate an issue")
                        else:
                            print(f"    ≈ Performance is similar")
                    else:
                        print(f"    ✗ Validation failed")
                        all_passed = False

                # Benchmark PyTorch grouped_mm (if available and supported)
                if self._is_pytorch_grouped_mm_available():
                    print(f"\n  Benchmarking PyTorch grouped_mm...")
                    pytorch_output, pytorch_time = self.benchmark_pytorch_grouped_mm(
                        moe_layer, hidden_states
                    )
                    if pytorch_output is not None:
                        results["pytorch"] = (pytorch_output, pytorch_time)
                        print(f"    Time: {pytorch_time:.3f} ms")

                        print(f"\n  Validating PyTorch vs Manual...")
                        if self.validate_outputs(manual_output, pytorch_output):
                            print(f"    ✓ Validation passed")
                            speedup_pytorch = manual_time / pytorch_time
                            print(f"    Speedup vs Manual: {speedup_pytorch:.2f}x")
                        else:
                            print(f"    ✗ Validation failed")
                else:
                    # Explain why PyTorch grouped_mm is not available
                    if not hasattr(torch, "_grouped_mm"):
                        print(
                            f"\n  PyTorch grouped_mm not available (requires PyTorch 2.4+)"
                        )
                    elif self.gpu_arch_info["is_blackwell"]:
                        print(
                            f"\n  PyTorch grouped_mm disabled on Blackwell (Hopper-only currently)"
                        )
                    else:
                        print(
                            f"\n  PyTorch grouped_mm not available on this architecture"
                        )

                # Performance summary with timing method info
                print(f"\n  Performance Summary ({self.timing_method} timing):")
                print(f"    Manual Loop:     {results['manual'][1]:.3f} ms")
                if "cutlass" in results:
                    speedup = results["manual"][1] / results["cutlass"][1]
                    print(
                        f"    CUTLASS Grouped: {results['cutlass'][1]:.3f} ms ({speedup:.2f}x)"
                    )
                if "pytorch" in results:
                    speedup_pytorch = results["manual"][1] / results["pytorch"][1]
                    print(
                        f"    PyTorch grouped: {results['pytorch'][1]:.3f} ms ({speedup_pytorch:.2f}x)"
                    )
                else:
                    print(f"    PyTorch grouped: Not available")

                # Calculate FLOPS (more detailed)
                total_tokens = config["batch_size"] * config["seq_len"]
                # More accurate FLOP counting:
                # Gate: tokens * hidden * intermediate * 2 (FMA)
                # Up: tokens * hidden * intermediate * 2 (FMA)
                # Down: tokens * intermediate * hidden * 2 (FMA)
                flops_per_token = 2 * (
                    config["hidden_size"] * config["intermediate_size"]  # gate
                    + config["hidden_size"] * config["intermediate_size"]  # up
                    + config["intermediate_size"] * config["hidden_size"]  # down
                )
                total_flops = (
                    total_tokens
                    * flops_per_token
                    * sum(1 for s in m_sizes if s > 0)
                    / config["num_experts"]
                )  # Adjust for active experts

                print(f"\n  FLOPS Analysis:")
                print(f"    Total FLOPs: {total_flops/1e9:.2f} GFLOP")

                manual_tflops = total_flops / (results["manual"][1] * 1e-3) / 1e12
                print(f"    Manual TFLOPS:   {manual_tflops:.3f}")

                if "cutlass" in results:
                    cutlass_tflops = total_flops / (results["cutlass"][1] * 1e-3) / 1e12
                    efficiency = cutlass_tflops / manual_tflops * 100
                    print(
                        f"    CUTLASS TFLOPS:  {cutlass_tflops:.3f} ({efficiency:.1f}% of manual)"
                    )

                if "pytorch" in results:
                    pytorch_tflops = total_flops / (results["pytorch"][1] * 1e-3) / 1e12
                    print(f"    PyTorch TFLOPS:  {pytorch_tflops:.3f}")

                # Memory bandwidth analysis
                total_params = (
                    config["num_experts"]
                    * config["hidden_size"]
                    * config["intermediate_size"]
                    * 2  # gate + up
                    + config["num_experts"]
                    * config["intermediate_size"]
                    * config["hidden_size"]  # down
                )
                param_size_gb = total_params * 2 / 1e9  # BF16 = 2 bytes

                print(f"\n  Memory Analysis:")
                print(
                    f"    Total parameters: {total_params/1e6:.1f}M ({param_size_gb:.2f} GB)"
                )

                manual_bandwidth = param_size_gb / (results["manual"][1] * 1e-3)
                print(f"    Manual bandwidth: {manual_bandwidth:.1f} GB/s")

                if "cutlass" in results:
                    cutlass_bandwidth = param_size_gb / (results["cutlass"][1] * 1e-3)
                    print(f"    CUTLASS bandwidth: {cutlass_bandwidth:.1f} GB/s")

            except Exception as e:
                print(f"✗ {config['name']} failed: {e}")
                import traceback

                traceback.print_exc()
                all_passed = False

            finally:
                # Cleanup
                torch.cuda.empty_cache()
                gc.collect()

        print(f"\n" + "=" * 80)
        if all_passed:
            print("🎉 All benchmarks completed successfully!")
        else:
            print("⚠️  Some benchmarks failed")
        print("=" * 80)

        return all_passed


def main():
    """Run the MoE benchmark"""
    benchmark = MoEBenchmark()

    print("MoE Performance Benchmark with Accurate GPU Timing")
    print(f"Device: {benchmark.device}")
    print(f"Dtype: {benchmark.dtype}")
    print(f"Timing Method: {benchmark.timing_method}")
    print(f"CUTLASS Available: {HAS_CUTLASS}")
    print(
        f"PyTorch grouped_mm Available: {benchmark._is_pytorch_grouped_mm_available()}"
    )

    # Show architecture-specific information
    if torch.cuda.is_available():
        arch_info = benchmark.gpu_arch_info
        if arch_info["is_blackwell"]:
            print(
                f"🚫 PyTorch grouped_mm disabled on Blackwell (compute capability {arch_info['compute_capability']})"
            )
        elif arch_info["is_hopper"]:
            print(
                f" PyTorch grouped_mm available on Hopper (compute capability {arch_info['compute_capability']})"
            )

    if benchmark.timing_method == "triton":
        print("Using Triton do_bench for GPU timing")
    elif benchmark.timing_method == "cuda_events":
        print("⏱️  Using CUDA events for accurate GPU timing")
    else:
        print("⚠️  Using CPU timing - results may be less accurate")

    success = benchmark.run_benchmark_suite()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
