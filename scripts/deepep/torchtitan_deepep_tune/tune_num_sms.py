#!/usr/bin/env python3
"""
DeepEP num_sms Tuner - Tests different num_sms values by recreating Buffer each time

This script properly tunes num_sms by creating a fresh Buffer for each value,
avoiding the caching issue that caused assertion errors.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import deep_ep

import torch
import torch.distributed as dist


@dataclass
class NumSmsResult:
    """Result for a single num_sms value across all NVLink configs"""

    num_sms: int
    best_dispatch_config: Tuple[int, int]  # (nvl_send, nvl_buffer)
    best_dispatch_time_us: float
    best_dispatch_bandwidth_gbps: float
    best_combine_config: Tuple[int, int]  # (nvl_send, nvl_buffer)
    best_combine_time_us: float
    best_combine_bandwidth_gbps: float
    num_configs_tested: int


def init_dist():
    """Initialize distributed environment"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    return local_rank, world_size, dist.group.WORLD, rank


def bench(fn, warmup=5, repeat=10):
    """Benchmark a function"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / repeat


def tune_for_num_sms(
    num_sms: int,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    num_ranks: int,
    rank: int,
    group: dist.ProcessGroup,
    local_rank: int,
) -> NumSmsResult:
    """Tune dispatch and combine for a specific num_sms value"""

    # Create fresh Buffer for this num_sms
    deep_ep.Buffer.set_num_sms(num_sms)
    buffer = deep_ep.Buffer(group, int(2e9), 0, explicitly_destroy=True)

    if local_rank == 0:
        print(f"\n{'='*100}")
        print(f"Testing num_sms = {num_sms}")
        print(f"{'='*100}")

    # Generate test data
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    # Get layout
    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)

    # Test configurations
    nvl_buffer_range = [256, 512, 1024]
    dispatch_chunk_range = range(4, 33, 2)
    combine_chunk_range = range(1, 17, 1)

    # Tune dispatch
    best_dispatch_time = float("inf")
    best_dispatch_config = None
    best_dispatch_bandwidth = 0.0

    for nvl_buffer in nvl_buffer_range:
        for nvl_chunk in dispatch_chunk_range:
            try:
                config = deep_ep.Config(num_sms, nvl_chunk, nvl_buffer)

                # Initial dispatch to get handle
                dispatch_args = {
                    "x": x,
                    "num_tokens_per_rank": num_tokens_per_rank,
                    "is_token_in_rank": is_token_in_rank,
                    "num_tokens_per_expert": num_tokens_per_expert,
                    "config": config,
                }
                recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)
                recv_bytes = recv_x.numel() * 2  # bfloat16

                # Benchmark with cached handle
                tune_args = {"x": x, "handle": handle, "config": config}
                t = bench(lambda: buffer.dispatch(**tune_args))
                bandwidth = recv_bytes / 1e9 / t

                if t < best_dispatch_time:
                    best_dispatch_time = t
                    best_dispatch_config = (nvl_chunk, nvl_buffer)
                    best_dispatch_bandwidth = bandwidth

            except Exception as e:
                if local_rank == 0:
                    print(
                        f"  [dispatch] FAILED: chunk={nvl_chunk}, buf={nvl_buffer} | {e}"
                    )
                continue

    if local_rank == 0 and best_dispatch_config:
        print(
            f"  [dispatch] BEST: chunk={best_dispatch_config[0]}, buf={best_dispatch_config[1]} | "
            f"{best_dispatch_bandwidth:.2f} GB/s | {best_dispatch_time * 1e6:.2f} us"
        )

    # Re-dispatch with best config for combine tuning
    if best_dispatch_config:
        best_dispatch_config_obj = deep_ep.Config(
            num_sms, best_dispatch_config[0], best_dispatch_config[1]
        )
        dispatch_args["config"] = best_dispatch_config_obj
        recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)
        send_bytes = recv_x.numel() * 2
    else:
        # Fallback if dispatch failed
        buffer.destroy()
        return NumSmsResult(
            num_sms=num_sms,
            best_dispatch_config=(0, 0),
            best_dispatch_time_us=float("inf"),
            best_dispatch_bandwidth_gbps=0.0,
            best_combine_config=(0, 0),
            best_combine_time_us=float("inf"),
            best_combine_bandwidth_gbps=0.0,
            num_configs_tested=0,
        )

    # Tune combine
    best_combine_time = float("inf")
    best_combine_config = None
    best_combine_bandwidth = 0.0

    for nvl_buffer in nvl_buffer_range:
        for nvl_chunk in combine_chunk_range:
            try:
                config = deep_ep.Config(num_sms, nvl_chunk, nvl_buffer)
                tune_args = {"x": recv_x, "handle": handle, "config": config}
                t = bench(lambda: buffer.combine(**tune_args))
                bandwidth = send_bytes / 1e9 / t

                if t < best_combine_time:
                    best_combine_time = t
                    best_combine_config = (nvl_chunk, nvl_buffer)
                    best_combine_bandwidth = bandwidth

            except Exception as e:
                if local_rank == 0:
                    print(
                        f"  [combine] FAILED: chunk={nvl_chunk}, buf={nvl_buffer} | {e}"
                    )
                continue

    if local_rank == 0 and best_combine_config:
        print(
            f"  [combine] BEST: chunk={best_combine_config[0]}, buf={best_combine_config[1]} | "
            f"{best_combine_bandwidth:.2f} GB/s | {best_combine_time * 1e6:.2f} us"
        )

    # Cleanup
    buffer.destroy()
    dist.barrier()

    num_configs = len(nvl_buffer_range) * (
        len(dispatch_chunk_range) + len(combine_chunk_range)
    )

    return NumSmsResult(
        num_sms=num_sms,
        best_dispatch_config=best_dispatch_config or (0, 0),
        best_dispatch_time_us=best_dispatch_time * 1e6,
        best_dispatch_bandwidth_gbps=best_dispatch_bandwidth,
        best_combine_config=best_combine_config or (0, 0),
        best_combine_time_us=best_combine_time * 1e6,
        best_combine_bandwidth_gbps=best_combine_bandwidth,
        num_configs_tested=num_configs,
    )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Comprehensive num_sms tuning for DeepEP"
    )
    parser.add_argument(
        "--num-tokens", type=int, default=4096, help="Tokens (default: 4096)"
    )
    parser.add_argument(
        "--hidden", type=int, default=2048, help="Hidden dim (default: 2048)"
    )
    parser.add_argument("--num-topk", type=int, default=8, help="Top-k (default: 8)")
    parser.add_argument(
        "--num-experts", type=int, default=128, help="Experts (default: 128)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory"
    )
    args = parser.parse_args()

    # Initialize distributed
    local_rank, num_ranks, group, rank = init_dist()
    torch.manual_seed(rank)

    if local_rank == 0:
        print("\n" + "=" * 100)
        print("DeepEP Comprehensive num_sms Tuner")
        print("=" * 100)
        print("Model: Qwen3-30B-A3B")
        print(f"  num_tokens = {args.num_tokens}")
        print(f"  hidden = {args.hidden}")
        print(f"  num_experts = {args.num_experts}")
        print(f"  num_topk = {args.num_topk}")
        print("Setup:")
        print(f"  num_ranks = {num_ranks}")
        print("  Hardware: B200 (~600 SMs per GPU)")
        print("=" * 100)

    # Test different num_sms values
    # Even values work best due to num_channels = num_sms / 2
    num_sms_range = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128]

    all_results = []
    for num_sms in num_sms_range:
        result = tune_for_num_sms(
            num_sms=num_sms,
            num_tokens=args.num_tokens,
            hidden=args.hidden,
            num_experts=args.num_experts,
            num_topk=args.num_topk,
            num_ranks=num_ranks,
            rank=rank,
            group=group,
            local_rank=local_rank,
        )
        all_results.append(result)

    # Find overall best
    if local_rank == 0:
        valid_results = [
            r for r in all_results if r.best_dispatch_time_us < float("inf")
        ]

        if not valid_results:
            print("\n[ERROR] No valid configurations found!")
            return

        best_dispatch = min(valid_results, key=lambda r: r.best_dispatch_time_us)
        best_combine = min(valid_results, key=lambda r: r.best_combine_time_us)
        worst_dispatch = max(valid_results, key=lambda r: r.best_dispatch_time_us)
        worst_combine = max(valid_results, key=lambda r: r.best_combine_time_us)

        # Find baseline (num_sms=24)
        baseline = next((r for r in valid_results if r.num_sms == 24), None)

        print("\n" + "=" * 100)
        print("FINAL RESULTS")
        print("=" * 100)

        print("\nOptimal Dispatch:")
        print(f"  num_sms = {best_dispatch.num_sms}")
        print(f"  nvl_send_chunk = {best_dispatch.best_dispatch_config[0]}")
        print(f"  nvl_recv_buffer = {best_dispatch.best_dispatch_config[1]}")
        disp_time = best_dispatch.best_dispatch_time_us
        disp_bw = best_dispatch.best_dispatch_bandwidth_gbps
        print(f"  Performance: {disp_time:.2f} us, {disp_bw:.2f} GB/s")
        print(f"  For utils.py: turbo_deepep_num_cus = {best_dispatch.num_sms}")
        disp_cfg_0 = best_dispatch.best_dispatch_config[0]
        disp_cfg_1 = best_dispatch.best_dispatch_config[1]
        print(
            f"               turbo_deepep_dispatch_tuned_config = "
            f"({disp_cfg_0}, {disp_cfg_1}, 8, 128)"
        )

        print("\nOptimal Combine:")
        print(f"  num_sms = {best_combine.num_sms}")
        print(f"  nvl_send_chunk = {best_combine.best_combine_config[0]}")
        print(f"  nvl_recv_buffer = {best_combine.best_combine_config[1]}")
        comb_time = best_combine.best_combine_time_us
        comb_bw = best_combine.best_combine_bandwidth_gbps
        print(f"  Performance: {comb_time:.2f} us, {comb_bw:.2f} GB/s")
        comb_cfg_0 = best_combine.best_combine_config[0]
        comb_cfg_1 = best_combine.best_combine_config[1]
        print(
            f"  For utils.py: turbo_deepep_combine_tuned_config = "
            f"({comb_cfg_0}, {comb_cfg_1}, 8, 128)"
        )

        print("\nComparison vs Worst:")
        dispatch_improvement = (
            (worst_dispatch.best_dispatch_time_us - best_dispatch.best_dispatch_time_us)
            / worst_dispatch.best_dispatch_time_us
            * 100
        )
        combine_improvement = (
            (worst_combine.best_combine_time_us - best_combine.best_combine_time_us)
            / worst_combine.best_combine_time_us
            * 100
        )
        worst_disp_time = worst_dispatch.best_dispatch_time_us
        best_disp_time = best_dispatch.best_dispatch_time_us
        print(
            f"  Dispatch: {dispatch_improvement:.1f}% faster "
            f"({worst_disp_time:.2f} us -> {best_disp_time:.2f} us)"
        )
        worst_comb_time = worst_combine.best_combine_time_us
        best_comb_time = best_combine.best_combine_time_us
        print(
            f"  Combine: {combine_improvement:.1f}% faster "
            f"({worst_comb_time:.2f} us -> {best_comb_time:.2f} us)"
        )

        if baseline:
            print("\nComparison vs Baseline (num_sms=24):")
            dispatch_vs_baseline = (
                (baseline.best_dispatch_time_us - best_dispatch.best_dispatch_time_us)
                / baseline.best_dispatch_time_us
                * 100
            )
            combine_vs_baseline = (
                (baseline.best_combine_time_us - best_combine.best_combine_time_us)
                / baseline.best_combine_time_us
                * 100
            )
            print(
                f"  Dispatch: {dispatch_vs_baseline:+.1f}% ({'faster' if dispatch_vs_baseline > 0 else 'slower'})"
            )
            print(
                f"    Baseline: {baseline.best_dispatch_time_us:.2f} us, {baseline.best_dispatch_bandwidth_gbps:.2f} GB/s"
            )
            print(
                f"    Optimal:  {best_dispatch.best_dispatch_time_us:.2f} us, {best_dispatch.best_dispatch_bandwidth_gbps:.2f} GB/s"
            )
            print(
                f"  Combine: {combine_vs_baseline:+.1f}% ({'faster' if combine_vs_baseline > 0 else 'slower'})"
            )
            print(
                f"    Baseline: {baseline.best_combine_time_us:.2f} us, {baseline.best_combine_bandwidth_gbps:.2f} GB/s"
            )
            print(
                f"    Optimal:  {best_combine.best_combine_time_us:.2f} us, {best_combine.best_combine_bandwidth_gbps:.2f} GB/s"
            )

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "model_params": {
                "num_tokens": args.num_tokens,
                "hidden": args.hidden,
                "num_experts": args.num_experts,
                "num_topk": args.num_topk,
            },
            "optimal_dispatch": {
                "num_sms": best_dispatch.num_sms,
                "nvl_send": best_dispatch.best_dispatch_config[0],
                "nvl_buffer": best_dispatch.best_dispatch_config[1],
                "time_us": best_dispatch.best_dispatch_time_us,
                "bandwidth_gbps": best_dispatch.best_dispatch_bandwidth_gbps,
                "utils_py_config": [
                    best_dispatch.best_dispatch_config[0],
                    best_dispatch.best_dispatch_config[1],
                    8,
                    128,
                ],
            },
            "optimal_combine": {
                "num_sms": best_combine.num_sms,
                "nvl_send": best_combine.best_combine_config[0],
                "nvl_buffer": best_combine.best_combine_config[1],
                "time_us": best_combine.best_combine_time_us,
                "bandwidth_gbps": best_combine.best_combine_bandwidth_gbps,
                "utils_py_config": [
                    best_combine.best_combine_config[0],
                    best_combine.best_combine_config[1],
                    8,
                    128,
                ],
            },
            "baseline_num_sms_24": {
                "dispatch_time_us": baseline.best_dispatch_time_us
                if baseline
                else None,
                "dispatch_bandwidth_gbps": baseline.best_dispatch_bandwidth_gbps
                if baseline
                else None,
                "combine_time_us": baseline.best_combine_time_us if baseline else None,
                "combine_bandwidth_gbps": baseline.best_combine_bandwidth_gbps
                if baseline
                else None,
            }
            if baseline
            else None,
            "all_results": [
                {
                    "num_sms": r.num_sms,
                    "dispatch_config": list(r.best_dispatch_config),
                    "dispatch_time_us": r.best_dispatch_time_us,
                    "dispatch_bandwidth_gbps": r.best_dispatch_bandwidth_gbps,
                    "combine_config": list(r.best_combine_config),
                    "combine_time_us": r.best_combine_time_us,
                    "combine_bandwidth_gbps": r.best_combine_bandwidth_gbps,
                }
                for r in all_results
            ],
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = output_dir / f"num_sms_tuning_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        summary_dir = output_dir.parent / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "num_sms_tuning_summary.json"

        summary = {
            "timestamp": results_dict["timestamp"],
            "optimal_dispatch": results_dict["optimal_dispatch"],
            "optimal_combine": results_dict["optimal_combine"],
            "baseline_num_sms_24": results_dict["baseline_num_sms_24"],
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\nResults saved to:")
        print(f"  Full: {results_path}")
        print(f"  Summary: {summary_path}")
        print("=" * 100)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
