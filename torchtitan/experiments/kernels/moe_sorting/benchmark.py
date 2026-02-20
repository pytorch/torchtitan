import argparse
import os
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from tabulate import tabulate

try:
    import token_sorting_cuda
except ImportError:
    print(f"Unable to import token_sorting extension")
    raise


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
    sorted_tokens = x[idxs // topk_ids.shape[1]]

    return sorted_tokens, idxs, tokens_per_expert


def cuda_sort_tokens(topk_ids, x, n_experts):
    """CUDA optimized implementation"""
    sorted_tokens, sorted_indices, tokens_per_expert = (
        token_sorting_cuda.sort_tokens_by_expert(topk_ids, x, n_experts)
    )
    return sorted_tokens, sorted_indices, tokens_per_expert


def verify_implementations(
    seq_len: int, hidden_dim: int, n_experts: int, k: int
) -> bool:
    """Verify that PyTorch and CUDA implementations produce identical results"""
    print(
        f"\nVerifying implementations for {n_experts} experts, {hidden_dim} features:"
    )

    # Create random input data
    torch.manual_seed(2020)
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
    tokens_match = torch.allclose(pt_sorted, cuda_sorted, rtol=1e-5, atol=1e-5)

    # Verify indices match
    indices_match = torch.equal(pt_indices, cuda_indices)

    # Print results
    print(f"  Token counts match: {counts_match}")
    print(f"  Sorted tokens match: {tokens_match}")
    print(f"  Indices match: {indices_match}")

    overall_match = counts_match and tokens_match and indices_match

    if not overall_match:
        print("\nDetailed diagnostics:")
        if not counts_match:
            diff = torch.abs(pt_counts - cuda_counts)
            max_diff = torch.max(diff).item()
            print(f"  Max count difference: {max_diff}")
            print(f"  First few PyTorch counts: {pt_counts[:5]}")
            print(f"  First few CUDA counts: {cuda_counts[:5]}")

        if not tokens_match:
            diff = torch.abs(pt_sorted - cuda_sorted)
            max_diff = torch.max(diff).item()
            max_diff_idx = torch.argmax(diff.view(-1)).item()
            print(f"  Max token difference: {max_diff} at index {max_diff_idx}")

        if not indices_match:
            print(f"  First few PyTorch indices: {pt_indices[:5]}")
            print(f"  First few CUDA indices: {cuda_indices[:5]}")

    return overall_match


def benchmark_implementations(
    seq_len: int, hidden_dim: int, n_experts: int, k: int, num_runs: int = 10
) -> Dict[str, Any]:
    """Benchmark PyTorch vs CUDA implementations"""
    # Create random input data
    torch.manual_seed(42)
    device = torch.device("cuda")

    topk_ids = torch.randint(
        0, n_experts, (seq_len, k), device=device, dtype=torch.int64
    )
    x = torch.randn(seq_len, hidden_dim, device=device)

    # Warmup
    for _ in range(3):
        pytorch_sort_tokens(topk_ids, x, n_experts)
        cuda_sort_tokens(topk_ids, x, n_experts)

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
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "n_experts": n_experts,
        "k": k,
        "pytorch_time": pt_avg_time,
        "pytorch_std": pt_std_time,
        "cuda_time": cuda_avg_time,
        "cuda_std": cuda_std_time,
        "speedup": speedup,
    }

    print(f"  PyTorch: {pt_avg_time:.3f} ± {pt_std_time:.3f} ms")
    print(f"  CUDA:    {cuda_avg_time:.3f} ± {cuda_std_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    return results


def run_benchmarks(
    expert_counts: List[int],
    hidden_dims: List[int],
    seq_lens: List[int],
    k_values: List[int],
    num_runs: int = 10,
    verify: bool = True,
) -> pd.DataFrame:
    """Run benchmarks for various configurations"""
    all_results = []

    # Ensure output directory exists
    os.makedirs("benchmark_results", exist_ok=True)

    for seq_len in seq_lens:
        for n_experts in expert_counts:
            for hidden_dim in hidden_dims:
                for k in k_values:
                    print(f"\n{'-'*80}")
                    print(
                        f"Benchmarking: seq_len={seq_len}, hidden_dim={hidden_dim}, experts={n_experts}, k={k}"
                    )

                    if verify:
                        verification_passed = verify_implementations(
                            seq_len, hidden_dim, n_experts, k
                        )
                        if not verification_passed:
                            print(
                                f"WARNING: Verification failed for this configuration!"
                            )

                    results = benchmark_implementations(
                        seq_len, hidden_dim, n_experts, k, num_runs
                    )
                    all_results.append(results)

                    # Save incremental results to avoid losing data if something crashes
                    pd.DataFrame(all_results).to_csv(
                        "benchmark_results/incremental_results.csv", index=False
                    )

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    results_df.to_csv("benchmark_results/benchmark_results.csv", index=False)

    return results_df


def create_plots(results_df: pd.DataFrame):
    """Create various plots from benchmark results"""
    # Create experts vs time plots for each hidden_dim
    for hidden_dim in results_df["hidden_dim"].unique():
        for k in results_df["k"].unique():
            for seq_len in results_df["seq_len"].unique():
                # Filter data
                df_filtered = results_df[
                    (results_df["hidden_dim"] == hidden_dim)
                    & (results_df["k"] == k)
                    & (results_df["seq_len"] == seq_len)
                ]

                if df_filtered.empty:
                    continue

                plt.figure(figsize=(10, 6))
                plt.errorbar(
                    df_filtered["n_experts"],
                    df_filtered["pytorch_time"],
                    yerr=df_filtered["pytorch_std"],
                    marker="o",
                    label="PyTorch",
                )
                plt.errorbar(
                    df_filtered["n_experts"],
                    df_filtered["cuda_time"],
                    yerr=df_filtered["cuda_std"],
                    marker="s",
                    label="CUDA",
                )

                plt.xscale("log", base=2)
                plt.yscale("log")
                plt.xlabel("Number of Experts")
                plt.ylabel("Time (ms)")
                plt.title(
                    f"Execution Time vs. Experts (hidden_dim={hidden_dim}, k={k}, seq_len={seq_len})"
                )
                plt.grid(True, which="both", ls="--", alpha=0.5)
                plt.legend()
                plt.tight_layout()

                plt.savefig(
                    f"benchmark_results/time_vs_experts_h{hidden_dim}_k{k}_seq{seq_len}.png",
                    dpi=300,
                )
                plt.close()

                # Create speedup plot
                plt.figure(figsize=(10, 6))
                plt.plot(df_filtered["n_experts"], df_filtered["speedup"], marker="o")
                plt.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
                plt.xscale("log", base=2)
                plt.xlabel("Number of Experts")
                plt.ylabel("Speedup (PyTorch / CUDA)")
                plt.title(
                    f"Speedup vs. Experts (hidden_dim={hidden_dim}, k={k}, seq_len={seq_len})"
                )
                plt.grid(True, which="both", ls="--", alpha=0.5)
                plt.tight_layout()

                plt.savefig(
                    f"benchmark_results/speedup_vs_experts_h{hidden_dim}_k{k}_seq{seq_len}.png",
                    dpi=300,
                )
                plt.close()

    # Create hidden_dim vs time plots for each expert count
    for n_experts in results_df["n_experts"].unique():
        for k in results_df["k"].unique():
            for seq_len in results_df["seq_len"].unique():
                # Filter data
                df_filtered = results_df[
                    (results_df["n_experts"] == n_experts)
                    & (results_df["k"] == k)
                    & (results_df["seq_len"] == seq_len)
                ]

                if df_filtered.empty:
                    continue

                plt.figure(figsize=(10, 6))
                plt.errorbar(
                    df_filtered["hidden_dim"],
                    df_filtered["pytorch_time"],
                    yerr=df_filtered["pytorch_std"],
                    marker="o",
                    label="PyTorch",
                )
                plt.errorbar(
                    df_filtered["hidden_dim"],
                    df_filtered["cuda_time"],
                    yerr=df_filtered["cuda_std"],
                    marker="s",
                    label="CUDA",
                )

                plt.xscale("log", base=2)
                plt.yscale("log")
                plt.xlabel("Hidden Dimension")
                plt.ylabel("Time (ms)")
                plt.title(
                    f"Execution Time vs. Hidden Dim (experts={n_experts}, k={k}, seq_len={seq_len})"
                )
                plt.grid(True, which="both", ls="--", alpha=0.5)
                plt.legend()
                plt.tight_layout()

                plt.savefig(
                    f"benchmark_results/time_vs_hidden_e{n_experts}_k{k}_seq{seq_len}.png",
                    dpi=300,
                )
                plt.close()

    # Create a summary heatmap of speedups
    plt.figure(figsize=(12, 8))

    # For simplicity, fix k=2 and seq_len=2048 for the heatmap
    if 2 in results_df["k"].values and 2048 in results_df["seq_len"].values:
        df_heatmap = results_df[
            (results_df["k"] == 1) & (results_df["seq_len"] == 4096)
        ]

        # Create pivot table for heatmap
        heatmap_data = df_heatmap.pivot(
            index="n_experts", columns="hidden_dim", values="speedup"
        )

        # Plot heatmap
        plt.imshow(heatmap_data, cmap="viridis", aspect="auto", interpolation="nearest")
        plt.colorbar(label="Speedup (PyTorch / CUDA)")

        # Set labels
        plt.xlabel("Hidden Dimension")
        plt.ylabel("Number of Experts")
        plt.title("Speedup Heatmap (k=1, seq_len=4096)")

        # Set ticks
        plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
        plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)

        plt.tight_layout()
        plt.savefig("benchmark_results/speedup_heatmap.png", dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and verify token sorting implementations"
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="2048,4096,8192",
        help="Comma-separated list of sequence lengths to test",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="1024,4096,8192",
        help="Comma-separated list of hidden dimensions to test",
    )
    parser.add_argument(
        "--expert-counts",
        type=str,
        default="16,64,128,256,512",
        help="Comma-separated list of expert counts to test",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,2,6",
        help="Comma-separated list of k values (experts per token) to test",
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of runs for each benchmark"
    )
    parser.add_argument(
        "--skip-verify", action="store_true", help="Skip verification step"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick benchmark with reduced parameter sets",
    )
    args = parser.parse_args()

    # Parse arguments
    if args.quick:
        # Use a reduced set of parameters for quick testing
        seq_lens = [2048]
        hidden_dims = [1024]
        expert_counts = [16, 128, 512]
        k_values = [2]
    else:
        seq_lens = [int(x) for x in args.seq_lens.split(",")]
        hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
        expert_counts = [int(x) for x in args.expert_counts.split(",")]
        k_values = [int(x) for x in args.k_values.split(",")]

    print("=" * 80)
    print("Token Sorting Benchmark")
    print("=" * 80)
    print(f"Sequence Lengths: {seq_lens}")
    print(f"Hidden Dimensions: {hidden_dims}")
    print(f"Expert Counts: {expert_counts}")
    print(f"K Values: {k_values}")
    print(f"Runs per test: {args.runs}")
    print(f"Skip verification: {args.skip_verify}")
    print("=" * 80)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        return 1

    # Print CUDA device info
    device = torch.cuda.current_device()
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(device)}")
    print(
        f"CUDA Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB"
    )
    print("=" * 80)

    # Run benchmarks
    results_df = run_benchmarks(
        expert_counts=expert_counts,
        hidden_dims=hidden_dims,
        seq_lens=seq_lens,
        k_values=k_values,
        num_runs=args.runs,
        verify=not args.skip_verify,
    )

    # Create plots
    create_plots(results_df)

    # Print summary table
    print("\n" + "=" * 100)
    print("Summary Results (k=2):")

    # For simplicity, show only k=2 in the summary table
    if 2 in results_df["k"].values:
        summary_df = results_df[results_df["k"] == 2]

        # Create a pivot table for better readability
        for seq_len in seq_lens:
            if seq_len in summary_df["seq_len"].values:
                seq_df = summary_df[summary_df["seq_len"] == seq_len]

                print(f"\nSequence Length: {seq_len}")

                summary_data = []
                for n_experts in expert_counts:
                    if n_experts not in seq_df["n_experts"].values:
                        continue

                    row = [n_experts]
                    for hidden_dim in hidden_dims:
                        if hidden_dim not in seq_df["hidden_dim"].values:
                            continue

                        # Get the speedup for this configuration
                        speedup = seq_df[
                            (seq_df["n_experts"] == n_experts)
                            & (seq_df["hidden_dim"] == hidden_dim)
                        ]["speedup"].values

                        if len(speedup) > 0:
                            row.append(f"{speedup[0]:.2f}x")
                        else:
                            row.append("N/A")

                    summary_data.append(row)

                headers = ["Experts"] + [
                    f"Hidden={dim}"
                    for dim in hidden_dims
                    if dim in seq_df["hidden_dim"].values
                ]
                print(tabulate(summary_data, headers=headers, tablefmt="grid"))

    print("\nBenchmark complete! Results saved to benchmark_results/ directory.")
    return 0


if __name__ == "__main__":
    main()
