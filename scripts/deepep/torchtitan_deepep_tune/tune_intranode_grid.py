#!/usr/bin/env python3
"""
DeepEP Comprehensive Grid Tuner for TorchTitan - Single Node

Grid search across ALL parameters:
- num_sms: [16, 20, 24, 28, 32]
- nvl_buffer_size: [256, 512, 1024]
- nvl_send_chunk (dispatch): range(4, 33, 2)
- nvl_send_chunk (combine): range(1, 17, 1)

Based on /home/phuc/workspace/moe/DeepEP/tests/test_intranode.py
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import deep_ep

import torch
import torch.distributed as dist


@dataclass
class BenchResult:
    """Benchmark result for a single configuration"""

    num_sms: int
    nvl_send: int
    nvl_buffer: int
    time_us: float
    bandwidth_gbps: float

    @property
    def config_tuple(self) -> Tuple[int, int, int]:
        """Return (num_sms, nvl_send, nvl_buffer)"""
        return (self.num_sms, self.nvl_send, self.nvl_buffer)


def init_dist():
    """Initialize distributed environment (torchrun already sets up env vars)"""
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


def grid_tune_dispatch(
    buffer: deep_ep.Buffer,
    x: torch.Tensor,
    handle: any,
    num_sms_range: List[int],
    nvl_buffer_range: List[int],
    nvl_chunk_range: range,
    recv_bytes: int,
    local_rank: int,
) -> Tuple[BenchResult, List[BenchResult]]:
    """Grid search for dispatch operation"""
    best_time = float("inf")
    best_result = None
    all_results = []

    total_configs = len(num_sms_range) * len(nvl_buffer_range) * len(nvl_chunk_range)
    config_num = 0

    if local_rank == 0:
        print(f"\n[dispatch] Testing {total_configs} configurations...")
        print("=" * 100)

    for num_sms in num_sms_range:
        for nvl_buffer in nvl_buffer_range:
            for nvl_chunk in nvl_chunk_range:
                config_num += 1
                config = deep_ep.Config(num_sms, nvl_chunk, nvl_buffer)
                tune_args = {"x": x, "handle": handle, "config": config}

                try:
                    t = bench(lambda: buffer.dispatch(**tune_args))
                    bandwidth_gbps = recv_bytes / 1e9 / t

                    result = BenchResult(
                        num_sms=num_sms,
                        nvl_send=nvl_chunk,
                        nvl_buffer=nvl_buffer,
                        time_us=t * 1e6,
                        bandwidth_gbps=bandwidth_gbps,
                    )
                    all_results.append(result)

                    if t < best_time:
                        best_time = t
                        best_result = result

                    if local_rank == 0:
                        progress = f"[{config_num}/{total_configs}]"
                        print(
                            f"{progress:>12} SMs={num_sms:2d} chunk={nvl_chunk:2d} buf={nvl_buffer:4d} | "
                            f"{bandwidth_gbps:6.2f} GB/s | {t * 1e6:6.2f} us"
                        )
                except Exception as e:
                    if local_rank == 0:
                        print(
                            f"[{config_num}/{total_configs}] FAILED: SMs={num_sms} chunk={nvl_chunk} buf={nvl_buffer} | Error: {e}"
                        )

    if local_rank == 0 and best_result:
        print("=" * 100)
        print(
            f"[dispatch] BEST: SMs={best_result.num_sms}, chunk={best_result.nvl_send}, buf={best_result.nvl_buffer} | "
            f"{best_result.bandwidth_gbps:.2f} GB/s | {best_result.time_us:.2f} us"
        )
        print("=" * 100)

    return best_result, all_results


def grid_tune_combine(
    buffer: deep_ep.Buffer,
    recv_x: torch.Tensor,
    handle: any,
    num_sms_range: List[int],
    nvl_buffer_range: List[int],
    nvl_chunk_range: range,
    send_bytes: int,
    local_rank: int,
) -> Tuple[BenchResult, List[BenchResult]]:
    """Grid search for combine operation"""
    best_time = float("inf")
    best_result = None
    all_results = []

    total_configs = len(num_sms_range) * len(nvl_buffer_range) * len(nvl_chunk_range)
    config_num = 0

    if local_rank == 0:
        print(f"\n[combine] Testing {total_configs} configurations...")
        print("=" * 100)

    for num_sms in num_sms_range:
        for nvl_buffer in nvl_buffer_range:
            for nvl_chunk in nvl_chunk_range:
                config_num += 1
                config = deep_ep.Config(num_sms, nvl_chunk, nvl_buffer)
                tune_args = {"x": recv_x, "handle": handle, "config": config}

                try:
                    t = bench(lambda: buffer.combine(**tune_args))
                    bandwidth_gbps = send_bytes / 1e9 / t

                    result = BenchResult(
                        num_sms=num_sms,
                        nvl_send=nvl_chunk,
                        nvl_buffer=nvl_buffer,
                        time_us=t * 1e6,
                        bandwidth_gbps=bandwidth_gbps,
                    )
                    all_results.append(result)

                    if t < best_time:
                        best_time = t
                        best_result = result

                    if local_rank == 0:
                        progress = f"[{config_num}/{total_configs}]"
                        print(
                            f"{progress:>12} SMs={num_sms:2d} chunk={nvl_chunk:2d} buf={nvl_buffer:4d} | "
                            f"{bandwidth_gbps:6.2f} GB/s | {t * 1e6:6.2f} us"
                        )
                except Exception as e:
                    if local_rank == 0:
                        print(
                            f"[{config_num}/{total_configs}] FAILED: SMs={num_sms} chunk={nvl_chunk} buf={nvl_buffer} | Error: {e}"
                        )

    if local_rank == 0 and best_result:
        print("=" * 100)
        print(
            f"[combine] BEST: SMs={best_result.num_sms}, chunk={best_result.nvl_send}, buf={best_result.nvl_buffer} | "
            f"{best_result.bandwidth_gbps:.2f} GB/s | {best_result.time_us:.2f} us"
        )
        print("=" * 100)

    return best_result, all_results


def test_main(
    args: argparse.Namespace,
    local_rank: int,
    num_ranks: int,
    rank: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
):
    """Main tuning logic with comprehensive grid search"""
    num_tokens = args.num_tokens
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts

    assert num_experts % num_ranks == 0

    if local_rank == 0:
        print("\n" + "=" * 100)
        print("DeepEP Comprehensive Grid Tuner for TorchTitan")
        print("=" * 100)
        print("Model: Qwen3-30B-A3B")
        print(f"  num_tokens = {num_tokens}")
        print(f"  hidden = {hidden}")
        print(f"  num_experts = {num_experts}")
        print(f"  num_topk = {num_topk}")
        print("Setup:")
        print(f"  num_ranks = {num_ranks}")
        print("  mode = intranode (single-node, NVLink only)")
        print("=" * 100)

    # Random data
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    # Rank layout
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)

    # Layout calculation
    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)

    # Grid search parameters
    # Note: For 4 ranks (<=8), num_sms has NO constraints per DeepEP config.hpp
    # B200 has ~600 SMs per GPU, so we can use much higher values than H100's default of 24
    # We test from conservative (16) to aggressive (96) to find optimal for B200
    num_sms_range = [16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96]
    nvl_buffer_range = [256, 512, 1024]
    dispatch_chunk_range = range(4, 33, 2)  # 4, 6, 8, ..., 32
    combine_chunk_range = range(1, 17, 1)  # 1, 2, 3, ..., 16

    # Initial dispatch with default config
    initial_config = deep_ep.Config(24, 8, 512)
    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": initial_config,
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Bytes for bandwidth calculation
    dispatch_recv_bytes = recv_x.numel() * 2  # bfloat16
    combine_send_bytes = dispatch_recv_bytes

    if local_rank == 0:
        print("\nData transfer:")
        print(f"  Dispatch recv: {dispatch_recv_bytes / 1e6:.2f} MB")
        print(f"  Combine send: {combine_send_bytes / 1e6:.2f} MB")

    # Grid tune dispatch
    best_dispatch, all_dispatch_results = grid_tune_dispatch(
        buffer,
        x,
        handle,
        num_sms_range,
        nvl_buffer_range,
        dispatch_chunk_range,
        dispatch_recv_bytes,
        local_rank,
    )

    # Re-dispatch with best config for combine tuning
    best_dispatch_config = deep_ep.Config(
        best_dispatch.num_sms, best_dispatch.nvl_send, best_dispatch.nvl_buffer
    )
    dispatch_args["config"] = best_dispatch_config
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Grid tune combine
    best_combine, all_combine_results = grid_tune_combine(
        buffer,
        recv_x,
        handle,
        num_sms_range,
        nvl_buffer_range,
        combine_chunk_range,
        combine_send_bytes,
        local_rank,
    )

    # Save results (rank 0 only)
    if local_rank == 0:
        # Convert to 4-param format for utils.py (add minimal RDMA values)
        dispatch_config_4param = (
            best_dispatch.nvl_send,
            best_dispatch.nvl_buffer,
            8,
            128,
        )
        combine_config_4param = (best_combine.nvl_send, best_combine.nvl_buffer, 8, 128)

        # Find worst results
        worst_dispatch = max(all_dispatch_results, key=lambda r: r.time_us)
        worst_combine = max(all_combine_results, key=lambda r: r.time_us)

        results = {
            "timestamp": datetime.now().isoformat(),
            "setup": {
                "ep_size": num_ranks,
                "num_ranks": num_ranks,
                "mode": "intranode_grid_search",
                "hardware": "B200",
            },
            "model_params": {
                "num_tokens": num_tokens,
                "hidden": hidden,
                "num_experts": num_experts,
                "num_topk": num_topk,
            },
            "grid_search_params": {
                "num_sms_range": num_sms_range,
                "nvl_buffer_range": nvl_buffer_range,
                "dispatch_chunk_range": [
                    dispatch_chunk_range.start,
                    dispatch_chunk_range.stop,
                    dispatch_chunk_range.step,
                ],
                "combine_chunk_range": [
                    combine_chunk_range.start,
                    combine_chunk_range.stop,
                    combine_chunk_range.step,
                ],
                "total_dispatch_configs": len(all_dispatch_results),
                "total_combine_configs": len(all_combine_results),
            },
            "optimal_configs": {
                "dispatch": {
                    "num_sms": best_dispatch.num_sms,
                    "nvl_send": best_dispatch.nvl_send,
                    "nvl_buffer": best_dispatch.nvl_buffer,
                    "utils_py_4param": list(dispatch_config_4param),
                },
                "combine": {
                    "num_sms": best_combine.num_sms,
                    "nvl_send": best_combine.nvl_send,
                    "nvl_buffer": best_combine.nvl_buffer,
                    "utils_py_4param": list(combine_config_4param),
                },
            },
            "performance": {
                "dispatch": {
                    "best_time_us": round(best_dispatch.time_us, 2),
                    "best_bandwidth_gbps": round(best_dispatch.bandwidth_gbps, 2),
                    "worst_time_us": round(worst_dispatch.time_us, 2),
                    "worst_bandwidth_gbps": round(worst_dispatch.bandwidth_gbps, 2),
                    "improvement_vs_worst_pct": round(
                        (worst_dispatch.time_us - best_dispatch.time_us)
                        / worst_dispatch.time_us
                        * 100,
                        1,
                    ),
                },
                "combine": {
                    "best_time_us": round(best_combine.time_us, 2),
                    "best_bandwidth_gbps": round(best_combine.bandwidth_gbps, 2),
                    "worst_time_us": round(worst_combine.time_us, 2),
                    "worst_bandwidth_gbps": round(worst_combine.bandwidth_gbps, 2),
                    "improvement_vs_worst_pct": round(
                        (worst_combine.time_us - best_combine.time_us)
                        / worst_combine.time_us
                        * 100,
                        1,
                    ),
                },
            },
            "all_results": {
                "dispatch": [
                    {
                        "num_sms": r.num_sms,
                        "nvl_send": r.nvl_send,
                        "nvl_buffer": r.nvl_buffer,
                        "time_us": round(r.time_us, 2),
                        "bandwidth_gbps": round(r.bandwidth_gbps, 2),
                    }
                    for r in all_dispatch_results
                ],
                "combine": [
                    {
                        "num_sms": r.num_sms,
                        "nvl_send": r.nvl_send,
                        "nvl_buffer": r.nvl_buffer,
                        "time_us": round(r.time_us, 2),
                        "bandwidth_gbps": round(r.bandwidth_gbps, 2),
                    }
                    for r in all_combine_results
                ],
            },
        }

        # Save full results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_results_path = output_dir / f"ep{num_ranks}_grid_{timestamp}.json"
        with open(full_results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save summary
        summary_dir = output_dir.parent / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"ep{num_ranks}_grid_summary.json"
        summary = {
            "timestamp": results["timestamp"],
            "optimal_configs": results["optimal_configs"],
            "performance": results["performance"],
            "grid_search_params": results["grid_search_params"],
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "=" * 100)
        print("FINAL RESULTS")
        print("=" * 100)
        print("\nOptimal Dispatch Config:")
        print(f"  num_sms = {best_dispatch.num_sms}")
        print(f"  nvl_send_chunk = {best_dispatch.nvl_send}")
        print(f"  nvl_recv_buffer = {best_dispatch.nvl_buffer}")
        print(
            f"  For utils.py: turbo_deepep_dispatch_tuned_config = {dispatch_config_4param}"
        )
        print(
            f"  Performance: {best_dispatch.time_us:.2f} us, {best_dispatch.bandwidth_gbps:.2f} GB/s"
        )
        print(
            f"  Improvement vs worst: {results['performance']['dispatch']['improvement_vs_worst_pct']:.1f}%"
        )

        print("\nOptimal Combine Config:")
        print(f"  num_sms = {best_combine.num_sms}")
        print(f"  nvl_send_chunk = {best_combine.nvl_send}")
        print(f"  nvl_recv_buffer = {best_combine.nvl_buffer}")
        print(
            f"  For utils.py: turbo_deepep_combine_tuned_config = {combine_config_4param}"
        )
        print(
            f"  Performance: {best_combine.time_us:.2f} us, {best_combine.bandwidth_gbps:.2f} GB/s"
        )
        print(
            f"  Improvement vs worst: {results['performance']['combine']['improvement_vs_worst_pct']:.1f}%"
        )

        print("\nResults saved to:")
        print(f"  Full: {full_results_path}")
        print(f"  Summary: {summary_path}")
        print("=" * 100)


def main():
    """Main entry point for each process (called by torchrun)"""
    parser = argparse.ArgumentParser(
        description="Comprehensive grid tuning for DeepEP intranode"
    )
    parser.add_argument(
        "--num-tokens", type=int, default=4096, help="Tokens (default: 4096)"
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=2048,
        help="Hidden dim - Qwen3-30B (default: 2048)",
    )
    parser.add_argument("--num-topk", type=int, default=8, help="Top-k (default: 8)")
    parser.add_argument(
        "--num-experts",
        type=int,
        default=128,
        help="Experts - Qwen3-30B-A3B (default: 128)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory"
    )
    args = parser.parse_args()

    # Initialize distributed (torchrun already set env vars)
    local_rank, num_ranks, group, rank = init_dist()
    buffer = deep_ep.Buffer(group, int(2e9), 0, explicitly_destroy=True)
    torch.manual_seed(rank)

    test_main(args, local_rank, num_ranks, rank, buffer, group)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
