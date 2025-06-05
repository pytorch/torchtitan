#!/usr/bin/env python3
"""
Test script for validating Group GEMM integration
"""

import time
from typing import Dict, List

import torch
import torch.nn as nn

# Import the strategies
from group_gemms import CUTLASSGroupGEMM, ManualLoopGroupGEMM, TritonCGBF16GroupGEMM


class MockMoEModule(nn.Module):
    """Mock MoE module for testing group GEMM strategies"""

    def __init__(self, num_experts=8, hidden_size=512, intermediate_size=1024):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Create expert weights
        self.gate_weights = []
        self.up_weights = []
        self.down_weights = []

        for i in range(num_experts):
            gate_weight = torch.randn(
                intermediate_size, hidden_size, dtype=torch.bfloat16
            )
            up_weight = torch.randn(
                intermediate_size, hidden_size, dtype=torch.bfloat16
            )
            down_weight = torch.randn(
                hidden_size, intermediate_size, dtype=torch.bfloat16
            )

            self.gate_weights.append(gate_weight)
            self.up_weights.append(up_weight)
            self.down_weights.append(down_weight)

        # Activation function
        self.activation = nn.SiLU()

    def setup_strategy(self, strategy_name):
        """Setup a specific group GEMM strategy"""
        strategies = {
            "manual": ManualLoopGroupGEMM(self.activation),
            "tritoncg": (
                TritonCGBF16GroupGEMM(self.activation)
                if TritonCGBF16GroupGEMM.is_available()
                else None
            ),
            "cutlass": (
                CUTLASSGroupGEMM(self.activation)
                if CUTLASSGroupGEMM.is_available()
                else None
            ),
        }

        strategy = strategies.get(strategy_name)
        if strategy is None:
            raise ValueError(f"Strategy {strategy_name} not available")

        # Arrange weights
        gate_combined = strategy.arrange_expert_weights(
            self.gate_weights, "gate_proj", self
        )
        up_combined = strategy.arrange_expert_weights(self.up_weights, "up_proj", self)
        down_combined = strategy.arrange_expert_weights(
            self.down_weights, "down_proj", self
        )

        # Register as parameters
        self.register_parameter("gate_proj_weight", nn.Parameter(gate_combined))
        self.register_parameter("up_proj_weight", nn.Parameter(up_combined))
        self.register_parameter("down_proj_weight", nn.Parameter(down_combined))

        self.strategy = strategy

    def forward(self, tokens, m_sizes, m_offsets):
        """Forward pass using the configured strategy"""
        return self.strategy.execute(tokens, m_sizes, m_offsets, self)


def create_test_data(seq_len=1024, hidden_size=512, num_experts=8, device="cuda"):
    """Create test data for group GEMM validation"""
    # Create random tokens
    tokens = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16, device=device)

    # Create realistic expert assignment (some experts get more tokens)
    expert_probs = torch.softmax(torch.randn(num_experts), dim=0)
    expert_assignments = torch.multinomial(expert_probs, seq_len, replacement=True)

    # Compute m_sizes (tokens per expert)
    m_sizes = []
    for i in range(num_experts):
        count = (expert_assignments == i).sum().item()
        m_sizes.append(count)

    # Compute m_offsets
    m_offsets = [0]
    for size in m_sizes:
        m_offsets.append(m_offsets[-1] + size)

    # Sort tokens by expert assignment
    sorted_indices = expert_assignments.argsort()
    sorted_tokens = tokens[sorted_indices]

    return sorted_tokens, m_sizes, m_offsets[:-1]  # Remove last offset


def validate_correctness(strategies: List[str], num_tests=3):
    """Validate that all strategies produce the same results"""
    print("=" * 60)
    print("CORRECTNESS VALIDATION")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: Running on CPU, some strategies may not be available")

    results_by_strategy = {}

    for test_idx in range(num_tests):
        print(f"\nTest {test_idx + 1}/{num_tests}")

        # Create test data
        tokens, m_sizes, m_offsets = create_test_data(device=device)
        print(f"  Tokens shape: {tokens.shape}")
        print(f"  Expert sizes: {m_sizes}")
        print(f"  Expert offsets: {m_offsets}")

        # Test each strategy
        for strategy_name in strategies:
            try:
                # Create fresh module for each strategy
                module = MockMoEModule().to(device)
                module.setup_strategy(strategy_name)

                print(f"    Testing {strategy_name}...")

                # Run forward pass
                with torch.no_grad():
                    start_time = time.time()
                    output = module(tokens, m_sizes, m_offsets)
                    end_time = time.time()

                if strategy_name not in results_by_strategy:
                    results_by_strategy[strategy_name] = []
                results_by_strategy[strategy_name].append(output.cpu())

                print(
                    f"    ✓ {strategy_name}: output shape {output.shape}, time: {(end_time-start_time)*1000:.2f}ms"
                )

                # Additional validation for CUTLASS
                if strategy_name == "cutlass":
                    print(
                        f"      CUTLASS output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}"
                    )
                    if torch.isnan(output).any():
                        print(f"      ⚠ WARNING: CUTLASS output contains NaN values!")
                    if torch.isinf(output).any():
                        print(
                            f"      ⚠ WARNING: CUTLASS output contains infinite values!"
                        )

            except Exception as e:
                print(f"    ✗ {strategy_name}: {e}")
                import traceback

                if strategy_name == "cutlass":  # More detailed error info for CUTLASS
                    print(f"      CUTLASS error details:")
                    traceback.print_exc()

    # Compare results across strategies
    print("\n" + "=" * 40)
    print("CROSS-STRATEGY COMPARISON")
    print("=" * 40)

    if len(results_by_strategy) < 2:
        print("Need at least 2 working strategies for comparison")
        return

    strategy_names = list(results_by_strategy.keys())
    reference_strategy = strategy_names[0]
    reference_results = results_by_strategy[reference_strategy]

    print(f"Using {reference_strategy} as reference")

    for strategy_name in strategy_names[1:]:
        strategy_results = results_by_strategy[strategy_name]

        max_diff = 0.0
        all_close = True

        for test_idx in range(len(reference_results)):
            try:
                ref = reference_results[test_idx]
                test = strategy_results[test_idx]

                diff = torch.abs(ref - test).max().item()
                max_diff = max(max_diff, diff)

                # Use looser tolerance for CUTLASS due to potential precision differences
                tolerance = 1e-1 if strategy_name == "cutlass" else 1e-2
                torch.testing.assert_close(ref, test, atol=tolerance, rtol=tolerance)

            except (AssertionError, IndexError) as e:
                all_close = False
                print(
                    f"  ✗ {strategy_name} vs {reference_strategy}: Test {test_idx + 1} failed"
                )
                print(f"    Error: {e}")
                break

        if all_close:
            print(
                f"  ✓ {strategy_name} vs {reference_strategy}: All tests passed (max_diff: {max_diff:.2e})"
            )
        else:
            print(f"  ✗ {strategy_name} vs {reference_strategy}: Some tests failed")


def benchmark_performance(strategies: List[str], warmup=3, iterations=10):
    """Benchmark performance of different strategies"""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKING")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test different problem sizes
    test_configs = [
        {
            "seq_len": 512,
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_experts": 8,
        },
        {
            "seq_len": 1024,
            "hidden_size": 1024,
            "intermediate_size": 2048,
            "num_experts": 16,
        },
        {
            "seq_len": 2048,
            "hidden_size": 2048,
            "intermediate_size": 4096,
            "num_experts": 32,
        },
    ]

    results = {}

    for config in test_configs:
        print(
            f"\nConfig: seq_len={config['seq_len']}, hidden_size={config['hidden_size']}, experts={config['num_experts']}"
        )

        # Create test data
        tokens = torch.randn(
            config["seq_len"],
            config["hidden_size"],
            dtype=torch.bfloat16,
            device=device,
        )

        # Simple uniform distribution for consistent benchmarking
        tokens_per_expert = config["seq_len"] // config["num_experts"]
        m_sizes = [tokens_per_expert] * config["num_experts"]

        # Handle remainder
        remainder = config["seq_len"] % config["num_experts"]
        for i in range(remainder):
            m_sizes[i] += 1

        m_offsets = [sum(m_sizes[:i]) for i in range(len(m_sizes))]

        config_key = (
            f"{config['seq_len']}x{config['hidden_size']}x{config['num_experts']}"
        )
        results[config_key] = {}

        for strategy_name in strategies:
            try:
                # Setup module
                module = MockMoEModule(
                    num_experts=config["num_experts"],
                    hidden_size=config["hidden_size"],
                    intermediate_size=config["intermediate_size"],
                ).to(device)
                module.setup_strategy(strategy_name)

                # Warmup
                with torch.no_grad():
                    for _ in range(warmup):
                        _ = module(tokens, m_sizes, m_offsets)

                if device == "cuda":
                    torch.cuda.synchronize()

                # Benchmark
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(iterations):
                        _ = module(tokens, m_sizes, m_offsets)

                if device == "cuda":
                    torch.cuda.synchronize()

                end_time = time.time()
                avg_time = (end_time - start_time) / iterations * 1000  # ms

                results[config_key][strategy_name] = avg_time
                print(f"  {strategy_name:12}: {avg_time:8.3f} ms")

            except Exception as e:
                print(f"  {strategy_name:12}: FAILED ({e})")
                results[config_key][strategy_name] = None

    # Summary
    print("\n" + "=" * 40)
    print("PERFORMANCE SUMMARY")
    print("=" * 40)

    for config_key, config_results in results.items():
        print(f"\n{config_key}:")

        # Find baseline (manual) time
        baseline_time = config_results.get("manual")
        if baseline_time is None:
            baseline_time = min(t for t in config_results.values() if t is not None)

        for strategy, time_ms in config_results.items():
            if time_ms is not None:
                speedup = baseline_time / time_ms if time_ms > 0 else 0
                print(f"  {strategy:12}: {time_ms:8.3f} ms ({speedup:5.2f}x)")
            else:
                print(f"  {strategy:12}: FAILED")


def debug_cutlass_integration():
    """Debug CUTLASS integration with detailed logging"""
    print("\n" + "=" * 60)
    print("CUTLASS INTEGRATION DEBUG")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available - skipping CUTLASS debug")
        return

    try:
        from group_gemms import CUTLASSGroupGEMM

        if not CUTLASSGroupGEMM.is_available():
            print("CUTLASS not available")
            return

        print("CUTLASS available, testing basic functionality...")

        # Create simple test case
        device = "cuda"
        tokens = torch.randn(128, 512, dtype=torch.bfloat16, device=device)
        m_sizes = [64, 64]  # Two experts with equal tokens
        m_offsets = [0, 64]

        print(f"Test data: {tokens.shape}, expert sizes: {m_sizes}")

        # Create module and setup CUTLASS
        module = MockMoEModule(
            num_experts=2, hidden_size=512, intermediate_size=1024
        ).to(device)

        print("Setting up CUTLASS strategy...")
        module.setup_strategy("cutlass")

        print("Executing CUTLASS forward pass...")
        with torch.no_grad():
            try:
                output = module(tokens, m_sizes, m_offsets)
                print(f"✓ CUTLASS execution successful!")
                print(f"  Output shape: {output.shape}")
                print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
                print(f"  Output mean: {output.mean():.4f}")

                # Check for common issues
                if torch.isnan(output).any():
                    print("  ⚠ WARNING: Output contains NaN")
                if torch.isinf(output).any():
                    print("  ⚠ WARNING: Output contains Inf")
                if output.abs().max() > 1000:
                    print("  ⚠ WARNING: Very large output values")

            except Exception as e:
                print(f"✗ CUTLASS execution failed: {e}")
                import traceback

                traceback.print_exc()

        print("CUTLASS debug completed.")

    except Exception as e:
        print(f"CUTLASS debug setup failed: {e}")


def main():
    """Main test function"""
    print("Group GEMM Integration Test")
    print("Testing available strategies...")

    # Check which strategies are available
    available_strategies = []

    strategies_to_test = ["manual", "tritoncg", "cutlass"]

    for strategy_name in strategies_to_test:
        try:
            if strategy_name == "manual":
                ManualLoopGroupGEMM(nn.SiLU())
                available_strategies.append(strategy_name)
            elif strategy_name == "tritoncg" and TritonCGBF16GroupGEMM.is_available():
                TritonCGBF16GroupGEMM(nn.SiLU())
                available_strategies.append(strategy_name)
            elif strategy_name == "cutlass" and CUTLASSGroupGEMM.is_available():
                CUTLASSGroupGEMM(nn.SiLU())
                available_strategies.append(strategy_name)
        except Exception as e:
            print(f"Strategy {strategy_name} not available: {e}")

    print(f"Available strategies: {available_strategies}")

    if len(available_strategies) == 0:
        print("No strategies available!")
        return

    # Debug CUTLASS specifically if available
    if "cutlass" in available_strategies:
        debug_cutlass_integration()

    # Run validation
    validate_correctness(available_strategies)

    # Run benchmarks
    if torch.cuda.is_available():
        benchmark_performance(available_strategies)
    else:
        print("\nSkipping performance benchmarks (CUDA not available)")


if __name__ == "__main__":
    main()
