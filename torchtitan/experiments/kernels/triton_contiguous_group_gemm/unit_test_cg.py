# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import time
import unittest

import torch
from cg_backward import cg_grouped_gemm

# Import implementations to test
from cg_forward import cg_grouped_gemm_forward


def run_tests(run_benchmarks=False):
    """Run unit tests with optional benchmarks"""
    # Create a test loader
    loader = unittest.TestLoader()

    # Create a test suite
    suite = unittest.TestSuite()

    # Add the test classes
    suite.addTest(loader.loadTestsFromTestCase(TestCGGEMMDeepSeekShapes))

    # Only add benchmarks if requested
    if run_benchmarks:
        print("running benchmarks...")
        suite.addTest(loader.loadTestsFromTestCase(TestCGGEMMPerformanceDeepSeek))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite).wasSuccessful()


class CGGEMMTestCase(unittest.TestCase):
    """Base test case for contiguous grouped GEMM tests."""

    def verify_forward(
        self, M_total, K, N, num_experts, group_size_m, print_stats=False
    ):
        """Verify forward pass correctness."""
        # Create test data
        inputs = torch.randn((M_total, K), dtype=torch.bfloat16, device="cuda")
        expert_weights = torch.randn(
            (num_experts, N, K), dtype=torch.bfloat16, device="cuda"
        )

        # Create expert indices with proper group alignment
        expert_indices = torch.zeros(M_total, dtype=torch.int32, device="cuda")
        num_groups = M_total // group_size_m

        for group_idx in range(num_groups):
            start_idx = group_idx * group_size_m
            end_idx = start_idx + group_size_m
            expert_idx = group_idx % num_experts
            expert_indices[start_idx:end_idx] = expert_idx

        # Run our implementation
        output_cg = cg_grouped_gemm_forward(
            inputs, expert_weights, expert_indices, group_size_m=group_size_m
        )

        # Run reference implementation
        output_ref = torch.empty((M_total, N), device="cuda", dtype=torch.bfloat16)
        for i in range(0, M_total, group_size_m):
            end_idx = min(i + group_size_m, M_total)
            expert_idx = expert_indices[i].item()
            expert_weight = expert_weights[expert_idx]
            output_ref[i:end_idx] = torch.matmul(inputs[i:end_idx], expert_weight.t())

        # Verify results
        is_close = torch.allclose(output_cg, output_ref, rtol=1e-2, atol=1e-2)

        if print_stats and not is_close:
            abs_diff = torch.abs(output_cg - output_ref)
            max_diff = torch.max(abs_diff).item()
            mean_diff = torch.mean(abs_diff).item()
            print(f"Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")

        return is_close

    def verify_backward(
        self, M_total, K, N, num_experts, group_size_m, print_stats=False
    ):
        """Verify backward pass correctness."""
        # Create test data with gradients
        inputs = torch.randn(
            (M_total, K), dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        expert_weights = torch.randn(
            (num_experts, N, K), dtype=torch.bfloat16, device="cuda", requires_grad=True
        )

        # Create copies for reference implementation
        inputs_ref = inputs.detach().clone().requires_grad_(True)
        expert_weights_ref = expert_weights.detach().clone().requires_grad_(True)

        # Create expert indices
        expert_indices = torch.zeros(M_total, dtype=torch.int32, device="cuda")
        num_groups = M_total // group_size_m

        for group_idx in range(num_groups):
            start_idx = group_idx * group_size_m
            end_idx = start_idx + group_size_m
            expert_idx = group_idx % num_experts
            expert_indices[start_idx:end_idx] = expert_idx

        # Forward pass - our implementation
        output_cg = cg_grouped_gemm(
            inputs, expert_weights, expert_indices, group_size_m=group_size_m
        )

        # Forward pass - reference implementation
        output_ref = torch.empty((M_total, N), device="cuda", dtype=torch.bfloat16)
        for i in range(0, M_total, group_size_m):
            end_idx = min(i + group_size_m, M_total)
            expert_idx = expert_indices[i].item()
            expert_weight = expert_weights_ref[expert_idx]
            output_ref[i:end_idx] = torch.matmul(
                inputs_ref[i:end_idx], expert_weight.t()
            )

        # Create gradient for backward pass
        grad_output = torch.randn_like(output_cg)
        grad_output_ref = grad_output.detach().clone()

        # Backward pass - our implementation
        output_cg.backward(grad_output)

        # Backward pass - reference implementation
        output_ref.backward(grad_output_ref)

        # Check input gradients
        inputs_match = torch.allclose(
            inputs.grad, inputs_ref.grad, rtol=1e-2, atol=1e-2
        )

        # Check weight gradients
        weights_match = torch.allclose(
            expert_weights.grad, expert_weights_ref.grad, rtol=1e-2, atol=1e-2
        )

        if print_stats and not (inputs_match and weights_match):
            if not inputs_match:
                abs_diff = torch.abs(inputs.grad - inputs_ref.grad)
                max_diff = torch.max(abs_diff).item()
                mean_diff = torch.mean(abs_diff).item()
                print(
                    f"Input grads - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}"
                )

            if not weights_match:
                abs_diff = torch.abs(expert_weights.grad - expert_weights_ref.grad)
                max_diff = torch.max(abs_diff).item()
                mean_diff = torch.mean(abs_diff).item()
                print(
                    f"Weight grads - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}"
                )

        return inputs_match, weights_match

    def benchmark_forward(self, M_total, K, N, num_experts, group_size_m, num_runs=10):
        """Benchmark forward pass performance."""
        # Create test data
        inputs = torch.randn((M_total, K), dtype=torch.bfloat16, device="cuda")
        expert_weights = torch.randn(
            (num_experts, N, K), dtype=torch.bfloat16, device="cuda"
        )

        # Create expert indices
        expert_indices = torch.zeros(M_total, dtype=torch.int32, device="cuda")
        num_groups = M_total // group_size_m

        for group_idx in range(num_groups):
            start_idx = group_idx * group_size_m
            end_idx = start_idx + group_size_m
            expert_idx = group_idx % num_experts
            expert_indices[start_idx:end_idx] = expert_idx

        # Warmup
        for _ in range(5):
            cg_grouped_gemm_forward(
                inputs, expert_weights, expert_indices, group_size_m=group_size_m
            )
            torch.cuda.synchronize()

        # Benchmark our implementation
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            cg_grouped_gemm_forward(
                inputs, expert_weights, expert_indices, group_size_m=group_size_m
            )
            torch.cuda.synchronize()
        cg_time = (time.time() - start) / num_runs

        # Reference implementation for comparison
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            output = torch.empty((M_total, N), device="cuda", dtype=torch.bfloat16)
            for i in range(0, M_total, group_size_m):
                end_idx = min(i + group_size_m, M_total)
                expert_idx = expert_indices[i].item()
                expert_weight = expert_weights[expert_idx]
                output[i:end_idx] = torch.matmul(inputs[i:end_idx], expert_weight.t())
            torch.cuda.synchronize()
        ref_time = (time.time() - start) / num_runs

        # Calculate TFLOPS
        flops = 2 * M_total * K * N  # Multiply-adds
        cg_tflops = flops / cg_time / 1e12
        ref_tflops = flops / ref_time / 1e12

        # Calculate speedup
        speedup = ref_time / cg_time

        # Return results
        shape_str = f"M={M_total}, K={K}, N={N}"
        return {
            "shape": shape_str,
            "cg_time": cg_time,
            "ref_time": ref_time,
            "speedup": speedup,
            "cg_tflops": cg_tflops,
            "ref_tflops": ref_tflops,
        }

    def benchmark_backward(self, M_total, K, N, num_experts, group_size_m, num_runs=5):
        """Benchmark backward pass performance."""
        # Create test data with gradients
        inputs = torch.randn(
            (M_total, K), dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        expert_weights = torch.randn(
            (num_experts, N, K), dtype=torch.bfloat16, device="cuda", requires_grad=True
        )

        # Create expert indices
        expert_indices = torch.zeros(M_total, dtype=torch.int32, device="cuda")
        num_groups = M_total // group_size_m

        for group_idx in range(num_groups):
            start_idx = group_idx * group_size_m
            end_idx = start_idx + group_size_m
            expert_idx = group_idx % num_experts
            expert_indices[start_idx:end_idx] = expert_idx

        # Create gradient for backward pass
        grad_output = torch.randn((M_total, N), dtype=torch.bfloat16, device="cuda")

        # Warmup
        for _ in range(3):
            # Our implementation
            inputs_cg = inputs.detach().clone().requires_grad_(True)
            weights_cg = expert_weights.detach().clone().requires_grad_(True)
            output_cg = cg_grouped_gemm(
                inputs_cg, weights_cg, expert_indices, group_size_m=group_size_m
            )
            output_cg.backward(grad_output)
            torch.cuda.synchronize()

        # Benchmark our implementation
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            inputs_cg = inputs.detach().clone().requires_grad_(True)
            weights_cg = expert_weights.detach().clone().requires_grad_(True)
            output_cg = cg_grouped_gemm(
                inputs_cg, weights_cg, expert_indices, group_size_m=group_size_m
            )
            output_cg.backward(grad_output)
            torch.cuda.synchronize()
        cg_time = (time.time() - start) / num_runs

        # Benchmark reference implementation
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            inputs_ref = inputs.detach().clone().requires_grad_(True)
            weights_ref = expert_weights.detach().clone().requires_grad_(True)
            output_ref = torch.empty((M_total, N), device="cuda", dtype=torch.bfloat16)
            for i in range(0, M_total, group_size_m):
                end_idx = min(i + group_size_m, M_total)
                expert_idx = expert_indices[i].item()
                expert_weight = weights_ref[expert_idx]
                output_ref[i:end_idx] = torch.matmul(
                    inputs_ref[i:end_idx], expert_weight.t()
                )
            output_ref.backward(grad_output)
            torch.cuda.synchronize()
        ref_time = (time.time() - start) / num_runs

        # Calculate TFLOPS (forward + backward)
        flops = 4 * M_total * K * N  # Forward + backward multiply-adds
        cg_tflops = flops / cg_time / 1e12
        ref_tflops = flops / ref_time / 1e12

        # Calculate speedup
        speedup = ref_time / cg_time

        # Return results
        shape_str = f"M={M_total}, K={K}, N={N}"
        return {
            "shape": shape_str,
            "cg_time": cg_time,
            "ref_time": ref_time,
            "speedup": speedup,
            "cg_tflops": cg_tflops,
            "ref_tflops": ref_tflops,
        }


class TestCGGEMMDeepSeekShapes(CGGEMMTestCase):
    """Tests for contiguous grouped GEMM with DeepSeek shapes."""

    def test_forward_deepseek_shapes(self):
        """Test forward pass with DeepSeek shapes."""
        # Updated shapes based on debug.py
        test_configs = [
            # M_total, K, N, num_experts, group_size_m
            (32 * 128, 1024, 1024, 8, 128),  # Similar to debug.py medium test
            (32 * 128, 4096, 4096, 8, 128),  # Smaller version of DeepSeek
            (16 * 128, 4096, 7168, 8, 128),  # Reduced DeepSeek shape
            (16 * 128, 7168, 2048, 8, 128),  # Reduced DeepSeek shape
        ]

        print("\n===== Testing Forward Pass: DeepSeek Shapes =====")
        for M_total, K, N, num_experts, group_size_m in test_configs:
            print(
                f"Testing shape: M={M_total}, K={K}, N={N}, group_size={group_size_m}"
            )
            match = self.verify_forward(
                M_total, K, N, num_experts, group_size_m, print_stats=True
            )
            self.assertTrue(
                match,
                f"Forward pass failed for DeepSeek shape: M={M_total}, K={K}, N={N}",
            )

    def test_backward_deepseek_shapes(self):
        """Test backward pass with DeepSeek shapes."""
        # Using smaller shapes based on debug.py
        test_configs = [
            # M_total, K, N, num_experts, group_size_m
            (4 * 128, 128, 128, 4, 128),  # Small test
            (8 * 128, 1024, 1024, 8, 128),  # Medium test
        ]

        print("\n===== Testing Backward Pass: DeepSeek Shapes   =====")
        for M_total, K, N, num_experts, group_size_m in test_configs:
            print(
                f"Testing shape: M={M_total:,}, K={K:,}, N={N:,}, group_size={group_size_m}"
            )
            inputs_match, weights_match = self.verify_backward(
                M_total, K, N, num_experts, group_size_m, print_stats=True
            )
            self.assertTrue(
                inputs_match,
                f"Input gradients failed for DeepSeek shape: M={M_total}, K={K}, N={N}",
            )
            self.assertTrue(
                weights_match,
                f"Weight gradients failed for DeepSeek shape: M={M_total}, K={K}, N={N}",
            )


class TestCGGEMMPerformanceDeepSeek(CGGEMMTestCase):
    """Performance benchmarks specifically for DeepSeek shapes."""

    def test_forward_performance_deepseek(self):
        """Benchmark forward pass performance with DeepSeek shapes."""
        # Updated shapes based on debug.py
        test_configs = [
            # M_total, K, N, num_experts, group_size_m
            (16 * 128, 1024, 1024, 8, 128),  # Medium test from debug.py
            (32 * 128, 4096, 4096, 8, 128),  # Smaller version of DeepSeek
        ]

        print("\n===== Benchmarking Forward Pass with DeepSeek Shapes =====")
        results = []
        for M_total, K, N, num_experts, group_size_m in test_configs:
            print(
                f"Benchmarking shape: M={M_total}, K={K}, N={N}, group_size={group_size_m}"
            )
            try:
                result = self.benchmark_forward(
                    M_total, K, N, num_experts, group_size_m, num_runs=5
                )
                results.append(result)
            except Exception as e:
                print(f"Error benchmarking shape M={M_total}, K={K}, N={N}: {e}")

        # Print summary table
        if results:
            print("\nDeepSeek Forward Performance Summary:")
            print(
                f"{'Shape':<30} {'CG-GEMM (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10} {'TFLOPS':<10}"
            )
            print("-" * 80)
            for result in results:
                print(
                    f"{result['shape']:<30} {result['cg_time'] * 1000:<15.3f} {result['ref_time'] * 1000:<15.3f} "
                    f"{result['speedup']:<10.2f} {result['cg_tflops']:<10.2f}"
                )
        else:
            print("No benchmark results collected. Check for errors above.")

    def test_backward_performance_deepseek(self):
        """Benchmark backward pass performance with reduced DeepSeek shapes."""
        # Use reduced sizes similar to debug.py
        test_configs = [
            # M_total, K, N, num_experts, group_size_m
            (4 * 128, 128, 128, 4, 128),  # Small test
            (8 * 128, 1024, 1024, 8, 128),  # Medium test
        ]

        print("\n===== Benchmarking Backward Pass with DeepSeek Shapes   =====")
        results = []
        for M_total, K, N, num_experts, group_size_m in test_configs:
            print(
                f"Benchmarking shape: M={M_total}, K={K}, N={N}, group_size={group_size_m}"
            )
            try:
                result = self.benchmark_backward(
                    M_total, K, N, num_experts, group_size_m, num_runs=3
                )
                results.append(result)
            except Exception as e:
                print(f"Error benchmarking shape M={M_total}, K={K}, N={N}: {e}")

        # Print summary table
        if results:
            print("\nDeepSeek Backward Performance Summary:")
            print(
                f"{'Shape':<30} {'CG-GEMM (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10} {'TFLOPS':<10}"
            )
            print("-" * 80)
            for result in results:
                print(
                    f"{result['shape']:<30} {result['cg_time'] * 1000:<15.3f} {result['ref_time'] * 1000:<15.3f} "
                    f"{result['speedup']:<10.2f} {result['cg_tflops']:<10.2f}"
                )
        else:
            print("No benchmark results collected. Check for errors above.")


if __name__ == "__main__":
    # Run tests
    run_benchmarks = False

    success = run_tests(run_benchmarks)
