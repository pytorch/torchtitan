import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from tabulate import tabulate

try:
    import token_sorting_cuda
except ImportError:
    print(f"unable to import token_sorting_cuda extension...")
    raise

# temp verify

print(f"Main Function signature: {token_sorting_cuda.sort_tokens_by_expert.__doc__}")

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import token_sorting_cuda
import torch
from tabulate import tabulate


def pytorch_sort_tokens(topk_ids, x, n_experts):
    """Original PyTorch implementation for comparison"""
    with torch.no_grad():
        # [seq_len, n_experts]
        cnts = topk_ids.new_zeros((topk_ids.shape[0], n_experts))
        # Fill 1 to the selected experts
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        # Token indices for each expert
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens_shape = idxs.shape + x.shape[1:]
    sorted_tokens = x[idxs // topk_ids.shape[1]]

    return sorted_tokens, idxs, tokens_per_expert


def cuda_sort_tokens(topk_ids, x, n_experts, use_parallel_scan=False):
    """CUDA optimized implementation"""
    # Ensure tensor types are compatible with CUDA implementation
    # The CUDA implementation expects int32 tensors internally but handles int64 conversion
    sorted_tokens, sorted_indices, tokens_per_expert = (
        token_sorting_cuda.sort_tokens_by_expert(
            topk_ids, x, n_experts, use_parallel_scan
        )
    )
    return sorted_tokens, sorted_indices, tokens_per_expert


def verify_implementations(seq_len, hidden_dim, n_experts, k=1):
    """Verify that PyTorch and CUDA implementations produce identical results"""
    print(f"\nVerifying implementations for {n_experts} experts:")

    # Create random input data
    torch.manual_seed(42)
    device = torch.device("cuda")

    # Generate expert IDs, ensuring they're valid indices
    topk_ids = torch.randint(
        0, n_experts, (seq_len, k), device=device, dtype=torch.int64
    )
    x = torch.randn(seq_len, hidden_dim, device=device)

    # Run implementations
    pt_sorted, pt_indices, pt_counts = pytorch_sort_tokens(topk_ids, x, n_experts)
    cuda_sorted, cuda_indices, cuda_counts = cuda_sort_tokens(topk_ids, x, n_experts)

    # Verify tokens per expert counts
    counts_match = torch.allclose(pt_counts, cuda_counts)

    # Verify sorted tokens
    # Note: For k>1, the shape of sorted tokens may differ between implementations
    # So we need to check if the content matches when reshaping
    if pt_sorted.shape[0] == cuda_sorted.shape[0]:
        tokens_match = torch.allclose(pt_sorted, cuda_sorted, rtol=1e-5, atol=1e-5)
    else:
        print(
            f"  Warning: Shape mismatch - PyTorch: {pt_sorted.shape}, CUDA: {cuda_sorted.shape}"
        )
        tokens_match = False

    # Check if indices map correctly - regenerate the original features
    # and see if they match the input
    if pt_indices.shape[0] == cuda_indices.shape[0]:
        # Map back to original token features
        pt_original = torch.zeros_like(x)
        cuda_original = torch.zeros_like(x)

        # Create mask for valid indices (to handle k>1 case)
        valid_pt_indices = pt_indices // k < seq_len
        valid_cuda_indices = cuda_indices < seq_len

        # Reconstruct using valid indices only
        pt_reconstructed = x[pt_indices[valid_pt_indices] // k]
        cuda_reconstructed = x[cuda_indices[valid_cuda_indices]]

        # Check if reconstructed features are close
        reconstruction_match = torch.allclose(
            pt_reconstructed, cuda_reconstructed, rtol=1e-5, atol=1e-5
        )
    else:
        print(
            f"  Warning: Indices shape mismatch - PyTorch: {pt_indices.shape}, CUDA: {cuda_indices.shape}"
        )
        reconstruction_match = False

    # Print results
    print(f"  Token counts match: {counts_match}")
    print(f"  Sorted tokens match: {tokens_match}")
    print(f"  Reconstruction match: {reconstruction_match}")

    overall_match = counts_match and tokens_match and reconstruction_match

    # For deeper verification, output the first few tokens and indices
    if not overall_match:
        print("\nDetailed debugging info:")
        print("  PyTorch tokens per expert:", pt_counts.cpu().numpy())
        print("  CUDA tokens per expert:", cuda_counts.cpu().numpy())

        print("\n  First 5 PyTorch sorted tokens:")
        print(pt_sorted[:5, :3].cpu().numpy())
        print("  First 5 CUDA sorted tokens:")
        print(cuda_sorted[:5, :3].cpu().numpy())

        print("\n  First 10 PyTorch indices:")
        print(pt_indices[:10].cpu().numpy())
        print("  First 10 CUDA indices:")
        print(cuda_indices[:10].cpu().numpy())

    return overall_match


def benchmark_implementations(
    seq_len, hidden_dim, n_experts, k=1, num_runs=10, verify=True
):
    """Benchmark PyTorch vs CUDA implementations"""
    # Create random input data
    torch.manual_seed(2020)
    device = torch.device("cuda")

    topk_ids = torch.randint(
        0, n_experts, (seq_len, k), device=device, dtype=torch.int64
    )
    x = torch.randn(seq_len, hidden_dim, device=device)

    # Verify if requested
    if verify:
        match = verify_implementations(seq_len, hidden_dim, n_experts, k)
        if not match:
            print(f"  WARNING: Verification failed for {n_experts} experts with k={k}!")

    # Warmup
    for _ in range(3):
        pytorch_sort_tokens(topk_ids, x, n_experts)
        cuda_sort_tokens(topk_ids, x, n_experts)

    results = {}

    # Benchmark PyTorch
    torch.cuda.synchronize()
    pt_times = []

    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        pytorch_sort_tokens(topk_ids, x, n_experts)
        end_event.record()

        torch.cuda.synchronize()
        pt_times.append(start_event.elapsed_time(end_event))

    pt_avg_time = sum(pt_times) / len(pt_times)
    pt_std_time = torch.tensor(pt_times).std().item()

    # Benchmark CUDA
    torch.cuda.synchronize()
    cuda_times = []

    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        cuda_sort_tokens(topk_ids, x, n_experts)
        end_event.record()

        torch.cuda.synchronize()
        cuda_times.append(start_event.elapsed_time(end_event))

    cuda_avg_time = sum(cuda_times) / len(cuda_times)
    cuda_std_time = torch.tensor(cuda_times).std().item()

    # Calculate speedup
    speedup = pt_avg_time / cuda_avg_time

    results = {
        "n_experts": n_experts,
        "k": k,
        "pytorch_time": pt_avg_time,
        "pytorch_std": pt_std_time,
        "cuda_time": cuda_avg_time,
        "cuda_std": cuda_std_time,
        "speedup": speedup,
    }

    print(f"\nBenchmark Results for {n_experts} experts, k={k}:")
    print(f"  PyTorch: {pt_avg_time:.3f} ± {pt_std_time:.3f} ms")
    print(f"  CUDA:    {cuda_avg_time:.3f} ± {cuda_std_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    return results


def plot_results(results_list):
    """Generate plots from benchmark results"""
    results_df = pd.DataFrame(results_list)

    # Create directory for plots
    import os

    os.makedirs("benchmark_results", exist_ok=True)

    # Group results by k value
    k_values = results_df["k"].unique()

    for k in k_values:
        k_results = results_df[results_df["k"] == k]

        # 1. Execution Time vs Number of Experts
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            k_results["n_experts"],
            k_results["pytorch_time"],
            yerr=k_results["pytorch_std"],
            marker="o",
            label="PyTorch",
        )
        plt.errorbar(
            k_results["n_experts"],
            k_results["cuda_time"],
            yerr=k_results["cuda_std"],
            marker="s",
            label="CUDA",
        )
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("Number of Experts")
        plt.ylabel("Execution Time (ms)")
        plt.title(f"Execution Time vs Number of Experts (k={k})")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"benchmark_results/execution_time_k{k}.png", dpi=300)

        # 2. Speedup vs Number of Experts
        plt.figure(figsize=(10, 6))
        plt.plot(k_results["n_experts"], k_results["speedup"], marker="o")
        plt.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
        plt.xscale("log", base=2)
        plt.xlabel("Number of Experts")
        plt.ylabel("Speedup Factor (PyTorch / CUDA)")
        plt.title(f"CUDA Speedup vs Number of Experts (k={k})")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"benchmark_results/speedup_k{k}.png", dpi=300)

    # 3. Impact of k on speedup
    if len(k_values) > 1:
        plt.figure(figsize=(10, 6))
        for n_exp in results_df["n_experts"].unique():
            n_exp_results = results_df[results_df["n_experts"] == n_exp]
            plt.plot(
                n_exp_results["k"],
                n_exp_results["speedup"],
                marker="o",
                label=f"{n_exp} experts",
            )

        plt.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
        plt.xlabel("k value (experts per token)")
        plt.ylabel("Speedup Factor (PyTorch / CUDA)")
        plt.title("Impact of k on CUDA Speedup")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("benchmark_results/k_impact.png", dpi=300)

    return results_df


def verify_k_values(seq_len, hidden_dim, n_experts):
    """Verify the implementation for different k values"""
    print("\n" + "=" * 80)
    print(f"Verifying implementation for different k values (experts per token)")
    print("=" * 80)

    k_values = [1, 2, 4, 8]
    results = []

    for k in k_values:
        print(f"\nTesting k={k}:")
        match = verify_implementations(seq_len, hidden_dim, n_experts, k)
        results.append({"k": k, "match": match})

    print("\nSummary results for different k values:")
    for res in results:
        status = "✓ PASS" if res["match"] else "✗ FAIL"
        print(f"k={res['k']}: {status}")

    return all(res["match"] for res in results)


def main():
    parser = argparse.ArgumentParser(
        description="Verify and benchmark token sorting implementations"
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument(
        "--hidden-dim", type=int, default=1024, help="Hidden dimension size"
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of runs for timing"
    )
    parser.add_argument(
        "--skip-verify", action="store_true", help="Skip verification step"
    )
    parser.add_argument(
        "--k", type=int, default=1, help="Number of expert assignments per token"
    )
    parser.add_argument(
        "--verify-k", action="store_true", help="Verify different k values"
    )
    parser.add_argument(
        "--experts",
        type=str,
        default="16,64,128,256,512",
        help="Comma-separated list of expert counts to test",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"Token Sorting Benchmark")
    print(
        f"Sequence Length: {args.seq_len}, Hidden Dimension: {args.hidden_dim}, k: {args.k}"
    )
    print("=" * 80)

    # Verify different k values if requested
    if args.verify_k:
        all_k_pass = verify_k_values(
            args.seq_len, args.hidden_dim, 64
        )  # Use 64 experts for k verification
        if not all_k_pass:
            print("\nWARNING: Some k values failed verification!")

    # Parse expert counts
    expert_counts = [int(x) for x in args.experts.split(",")]

    # Run benchmarks
    results = []
    for n_experts in expert_counts:
        result = benchmark_implementations(
            args.seq_len,
            args.hidden_dim,
            n_experts,
            k=args.k,
            num_runs=args.runs,
            verify=not args.skip_verify,
        )
        results.append(result)

    # Generate plots
    results_df = plot_results(results)

    # Print summary table
    print("\n" + "=" * 100)
    print("Summary Results:")

    summary_data = []
    for _, row in results_df.iterrows():
        summary_data.append(
            [
                row["n_experts"],
                row["k"],
                f"{row['pytorch_time']:.2f} ± {row['pytorch_std']:.2f}",
                f"{row['cuda_time']:.2f} ± {row['cuda_std']:.2f}",
                f"{row['speedup']:.2f}x",
            ]
        )

    headers = ["Experts", "k", "PyTorch (ms)", "CUDA (ms)", "Speedup"]
    print(tabulate(summary_data, headers=headers, tablefmt="grid"))
    print("=" * 100)

    print("\nBenchmark complete! Plots saved to benchmark_results/ directory.")


if __name__ == "__main__":

    seq_len = 8
    hidden_dim = 4
    n_experts = 4
    k = 2
    topk_ids = torch.randint(0, n_experts, (seq_len, k), device="cuda")
    x = torch.randn(seq_len, hidden_dim, device="cuda")

    # Compare results
    pt_result = pytorch_sort_tokens(topk_ids, x, n_experts)
    cuda_result = cuda_sort_tokens(topk_ids, x, n_experts)

    print(f"{pt_result=}")
    print(f"{cuda_result=}")

    # same = torch.allclose(pt_result, cuda_result)
    # print(f"{same=}")

    # main()
