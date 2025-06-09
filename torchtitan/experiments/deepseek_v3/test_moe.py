#!/usr/bin/env python3
"""
Simple MoE Benchmark using CUTLASSGroupedGemmStrategy

This benchmark creates a realistic MoE layer and compares:
1. CUTLASSGroupedGemmStrategy (our optimized approach)
2. Manual looping through experts (baseline)
3. PyTorch grouped_mm (if available)

Measures performance across different scales and expert utilization patterns.
"""

import gc
import math
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import CUTLASS components
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils

    # Import our strategy - UPDATE PATH AS NEEDED

    from cutlass.cute.runtime import from_dlpack
    from group_gemms import CUTLASSGroupedGemmStrategy, ManualLoopGroupGEMM

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
    ) -> Tuple[torch.Tensor, List[int], List[int], torch.Tensor]:
        """
        Route tokens to experts using top-k selection

        Returns:
            contig_tokens: Tokens arranged contiguously by expert assignment
            m_sizes: Number of tokens assigned to each expert
            m_offsets: Cumulative token offsets for each expert
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

        # Create offsets
        m_offsets = []
        cumsum = 0
        for size in m_sizes:
            cumsum += size
            m_offsets.append(cumsum)

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

        torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()

        for _ in range(iterations):
            contig_tokens, m_sizes, m_offsets, _ = moe_layer.route_tokens(hidden_states)
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
        avg_time = (end_time - start_time) / iterations * 1000  # ms

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

        torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()

        for _ in range(iterations):
            contig_tokens, m_sizes, m_offsets, _ = moe_layer.route_tokens(hidden_states)
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
        avg_time = (end_time - start_time) / iterations * 1000  # ms

        return final_output, avg_time

    def benchmark_pytorch_grouped_mm(
        self,
        moe_layer: SimpleMoELayer,
        hidden_states: torch.Tensor,
        iterations: int = 10,
    ) -> Tuple[torch.Tensor, float]:
        """Benchmark PyTorch _grouped_mm if available"""
        # Check if _grouped_mm exists and we're on a supported device
        if not hasattr(torch, "_grouped_mm"):
            return None, float("inf")

        # Check if we're on Blackwell (compute capability 9.0)
        device_props = torch.cuda.get_device_properties(hidden_states.device)
        if device_props.major != 9:
            print("  Skipping torch._grouped_mm: requires compute capability 9.0")
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
        # Convert m_offsets list to tensor
        m_offsets_tensor = torch.tensor(m_offsets, device=tokens.device)

        # Gate and up projections
        gate_proj = torch._grouped_mm(
            tokens,
            moe_layer.gate_weights.transpose(-2, -1),
            m_offsets_tensor,  # Use tensor instead of list
            out_dtype=self.dtype,
        )
        up_proj = torch._grouped_mm(
            tokens,
            moe_layer.up_weights.transpose(-2, -1),
            m_offsets_tensor,  # Use tensor instead of list
            out_dtype=self.dtype,
        )

        # Apply activation and combine
        hidden_outputs = moe_layer.silu_activation(gate_proj) * up_proj

        # Down projection
        final_outputs = torch._grouped_mm(
            hidden_outputs,
            moe_layer.down_weights.transpose(-2, -1),
            m_offsets_tensor,  # Use tensor instead of list
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

                results = {}

                # Benchmark manual strategy
                print(f"\n  Benchmarking Manual Loop Strategy...")
                manual_output, manual_time = self.benchmark_manual_strategy(
                    moe_layer, hidden_states
                )
                results["manual"] = (manual_output, manual_time)
                print(f"    Time: {manual_time:.2f} ms")

                # Benchmark CUTLASS strategy
                if HAS_CUTLASS:
                    print(f"\n  Benchmarking CUTLASS Strategy...")
                    cutlass_output, cutlass_time = self.benchmark_cutlass_strategy(
                        moe_layer, hidden_states
                    )
                    results["cutlass"] = (cutlass_output, cutlass_time)
                    print(f"    Time: {cutlass_time:.2f} ms")

                    # Validate
                    print(f"\n  Validating CUTLASS vs Manual...")
                    if self.validate_outputs(manual_output, cutlass_output):
                        print(f"    ✓ Validation passed")
                        speedup = manual_time / cutlass_time
                        print(f"    Speedup: {speedup:.2f}x")
                    else:
                        print(f"    ✗ Validation failed")
                        all_passed = False

                # Benchmark PyTorch grouped_mm
                if hasattr(torch, "_grouped_mm"):
                    print(f"\n  Benchmarking PyTorch grouped_mm...")
                    pytorch_output, pytorch_time = self.benchmark_pytorch_grouped_mm(
                        moe_layer, hidden_states
                    )
                    results["pytorch"] = (pytorch_output, pytorch_time)
                    print(f"    Time: {pytorch_time:.2f} ms")

                    if pytorch_output is not None:
                        print(f"\n  Validating PyTorch vs Manual...")
                        if self.validate_outputs(manual_output, pytorch_output):
                            print(f"    ✓ Validation passed")
                        else:
                            print(f"    ✗ Validation failed")

                # Summary
                print(f"\n  Performance Summary:")
                print(f"    Manual Loop:     {results['manual'][1]:.2f} ms")
                if "cutlass" in results:
                    print(f"    CUTLASS Grouped: {results['cutlass'][1]:.2f} ms")
                if "pytorch" in results:
                    print(f"    PyTorch grouped: {results['pytorch'][1]:.2f} ms")

                # Calculate FLOPS
                total_tokens = config["batch_size"] * config["seq_len"]
                # Approximate FLOPs: 2 * (gate + up + down projections)
                flops_per_token = 2 * (
                    config["hidden_size"] * config["intermediate_size"] * 2  # gate + up
                    + config["intermediate_size"] * config["hidden_size"]
                )  # down
                total_flops = total_tokens * flops_per_token

                manual_tflops = total_flops / (results["manual"][1] * 1e-3) / 1e12
                print(f"    Manual TFLOPS:   {manual_tflops:.2f}")

                if "cutlass" in results:
                    cutlass_tflops = total_flops / (results["cutlass"][1] * 1e-3) / 1e12
                    print(f"    CUTLASS TFLOPS:  {cutlass_tflops:.2f}")

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

    print("MoE Performance Benchmark")
    print(f"Device: {benchmark.device}")
    print(f"Dtype: {benchmark.dtype}")
    print(f"CUTLASS Available: {HAS_CUTLASS}")
    print(f"PyTorch grouped_mm Available: {hasattr(torch, '_grouped_mm')}")

    success = benchmark.run_benchmark_suite()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
