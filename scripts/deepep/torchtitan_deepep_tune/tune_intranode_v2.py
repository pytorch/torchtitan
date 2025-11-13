#!/usr/bin/env python3
"""
DeepEP Intranode Tuner for TorchTitan - Single Node (NVLink only)

Based on /home/phuc/workspace/moe/DeepEP/tests/test_intranode.py

For 4 GPUs in the same node with NVLink, we tune only NVLink parameters.
The Config format is: Config(num_sms, nvl_send_chunk, nvl_recv_buffer, rdma_send, rdma_recv)

For single-node (num_ranks <= NUM_MAX_NVL_PEERS), RDMA is not used, so we focus on NVLink tuning.
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

    config: Tuple[int, ...]  # (nvl_send, nvl_recv) for intranode
    time_us: float
    bandwidth_gbps: float


def init_dist(local_rank: int, num_local_ranks: int):
    """Initialize distributed environment"""
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(num_local_ranks)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(29500 + local_rank)

    dist.init_process_group(backend="nccl", rank=local_rank, world_size=num_local_ranks)
    torch.cuda.set_device(local_rank)
    return local_rank, num_local_ranks, dist.group.WORLD


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


def tune_dispatch(
    buffer: deep_ep.Buffer,
    x: torch.Tensor,
    handle: any,
    num_sms: int,
    nvl_buffer_size: int,
    recv_bytes: int,
    local_rank: int,
) -> Tuple[BenchResult, List[BenchResult]]:
    """Tune dispatch operation"""
    best_time = float("inf")
    best_config = None
    all_results = []

    # Tune nvl_chunk_size (send chunk)
    for nvl_chunk_size in tuple(range(4, 33, 2)):
        config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
        tune_args = {"x": x, "handle": handle, "config": config}
        t = bench(lambda: buffer.dispatch(**tune_args))

        bandwidth_gbps = recv_bytes / 1e9 / t
        result = BenchResult(
            config=(nvl_chunk_size, nvl_buffer_size),
            time_us=t * 1e6,
            bandwidth_gbps=bandwidth_gbps,
        )
        all_results.append(result)

        if t < best_time:
            best_time = t
            best_config = (nvl_chunk_size, nvl_buffer_size)

        if local_rank == 0:
            print(
                f"[dispatch] SMs {num_sms}, NVL chunk {nvl_chunk_size}, buffer {nvl_buffer_size}: "
                f"{bandwidth_gbps:.2f} GB/s, {t * 1e6:.2f} us"
            )

    best_result = BenchResult(
        config=best_config,
        time_us=best_time * 1e6,
        bandwidth_gbps=recv_bytes / 1e9 / best_time,
    )

    if local_rank == 0:
        print(
            f"\n[dispatch] BEST: SMs {num_sms}, NVL chunk {best_config[0]}, buffer {best_config[1]}: "
            f"{best_result.bandwidth_gbps:.2f} GB/s, {best_result.time_us:.2f} us\n"
        )

    return best_result, all_results


def tune_combine(
    buffer: deep_ep.Buffer,
    recv_x: torch.Tensor,
    handle: any,
    num_sms: int,
    nvl_buffer_size: int,
    send_bytes: int,
    local_rank: int,
) -> Tuple[BenchResult, List[BenchResult]]:
    """Tune combine operation"""
    best_time = float("inf")
    best_config = None
    all_results = []

    # Tune nvl_chunk_size (send chunk)
    for nvl_chunk_size in tuple(range(1, 17, 1)):
        config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
        tune_args = {"x": recv_x, "handle": handle, "config": config}
        t = bench(lambda: buffer.combine(**tune_args))

        bandwidth_gbps = send_bytes / 1e9 / t
        result = BenchResult(
            config=(nvl_chunk_size, nvl_buffer_size),
            time_us=t * 1e6,
            bandwidth_gbps=bandwidth_gbps,
        )
        all_results.append(result)

        if t < best_time:
            best_time = t
            best_config = (nvl_chunk_size, nvl_buffer_size)

        if local_rank == 0:
            print(
                f"[combine] SMs {num_sms}, NVL chunk {nvl_chunk_size}, buffer {nvl_buffer_size}: "
                f"{bandwidth_gbps:.2f} GB/s, {t * 1e6:.2f} us"
            )

    best_result = BenchResult(
        config=best_config,
        time_us=best_time * 1e6,
        bandwidth_gbps=send_bytes / 1e9 / best_time,
    )

    if local_rank == 0:
        print(
            f"\n[combine] BEST: SMs {num_sms}, NVL chunk {best_config[0]}, buffer {best_config[1]}: "
            f"{best_result.bandwidth_gbps:.2f} GB/s, {best_result.time_us:.2f} us\n"
        )

    return best_result, all_results


def test_main(
    args: argparse.Namespace,
    num_sms: int,
    local_rank: int,
    num_ranks: int,
    rank: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
):
    """Main tuning logic"""
    # Model parameters matching actual Qwen3-30B-A3B
    num_tokens = args.num_tokens
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts

    assert num_experts % num_ranks == 0

    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, num_experts={num_experts}"
        )
        print(f"[config] num_ranks={num_ranks}, num_sms={num_sms}")
        print()

    # Random data (matching test_intranode.py)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    # Rank layout meta
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)

    # Use DeepEP's layout calculation
    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)

    # NVL buffer size for 4 ranks (from DeepEP conventions)
    nvl_buffer_size = 512  # Larger buffer for 4 ranks

    # Initial config for dispatch setup
    initial_config = deep_ep.Config(num_sms, 8, nvl_buffer_size)

    # Do initial dispatch to get handle and recv_x
    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": initial_config,
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Calculate bytes for bandwidth measurement
    dispatch_recv_bytes = recv_x.numel() * 2  # bfloat16 = 2 bytes
    combine_send_bytes = dispatch_recv_bytes

    if local_rank == 0:
        print(f"[setup] Dispatch recv bytes: {dispatch_recv_bytes / 1e6:.2f} MB")
        print(f"[setup] Combine send bytes: {combine_send_bytes / 1e6:.2f} MB")
        print()

    # Tune dispatch
    if local_rank == 0:
        print("=" * 80)
        print("TUNING DISPATCH")
        print("=" * 80)

    best_dispatch, all_dispatch_results = tune_dispatch(
        buffer, x, handle, num_sms, nvl_buffer_size, dispatch_recv_bytes, local_rank
    )

    # Re-dispatch with best config to get proper recv_x for combine tuning
    best_dispatch_config = deep_ep.Config(
        num_sms, best_dispatch.config[0], best_dispatch.config[1]
    )
    dispatch_args["config"] = best_dispatch_config
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine
    if local_rank == 0:
        print("=" * 80)
        print("TUNING COMBINE")
        print("=" * 80)

    best_combine, all_combine_results = tune_combine(
        buffer, recv_x, handle, num_sms, nvl_buffer_size, combine_send_bytes, local_rank
    )

    # Gather results from rank 0
    if local_rank == 0:
        # Convert to 4-param format for utils.py compatibility
        # For single-node, we use minimal RDMA values since they're not used
        dispatch_config_4param = (
            best_dispatch.config[0],
            best_dispatch.config[1],
            8,
            128,
        )
        combine_config_4param = (best_combine.config[0], best_combine.config[1], 8, 128)

        results = {
            "timestamp": datetime.now().isoformat(),
            "setup": {
                "ep_size": num_ranks,
                "num_ranks": num_ranks,
                "mode": "intranode_single_node",
                "hardware": "B200",
                "num_sms": num_sms,
            },
            "model_params": {
                "num_tokens": num_tokens,
                "hidden": hidden,
                "num_experts": num_experts,
                "num_topk": num_topk,
            },
            "optimal_configs": {
                "dispatch_intranode_2param": list(best_dispatch.config),
                "combine_intranode_2param": list(best_combine.config),
                "dispatch_utils_py_4param": list(dispatch_config_4param),
                "combine_utils_py_4param": list(combine_config_4param),
            },
            "performance": {
                "dispatch": {
                    "best_time_us": round(best_dispatch.time_us, 2),
                    "best_bandwidth_gbps": round(best_dispatch.bandwidth_gbps, 2),
                    "worst_time_us": round(
                        max(r.time_us for r in all_dispatch_results), 2
                    ),
                    "improvement_vs_worst_pct": round(
                        (
                            max(r.time_us for r in all_dispatch_results)
                            - best_dispatch.time_us
                        )
                        / max(r.time_us for r in all_dispatch_results)
                        * 100,
                        1,
                    ),
                },
                "combine": {
                    "best_time_us": round(best_combine.time_us, 2),
                    "best_bandwidth_gbps": round(best_combine.bandwidth_gbps, 2),
                    "worst_time_us": round(
                        max(r.time_us for r in all_combine_results), 2
                    ),
                    "improvement_vs_worst_pct": round(
                        (
                            max(r.time_us for r in all_combine_results)
                            - best_combine.time_us
                        )
                        / max(r.time_us for r in all_combine_results)
                        * 100,
                        1,
                    ),
                },
            },
            "all_results": {
                "dispatch": [
                    {
                        "config": list(r.config),
                        "time_us": round(r.time_us, 2),
                        "bandwidth_gbps": round(r.bandwidth_gbps, 2),
                    }
                    for r in all_dispatch_results
                ],
                "combine": [
                    {
                        "config": list(r.config),
                        "time_us": round(r.time_us, 2),
                        "bandwidth_gbps": round(r.bandwidth_gbps, 2),
                    }
                    for r in all_combine_results
                ],
            },
        }

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_results_path = output_dir / f"ep{num_ranks}_intranode_{timestamp}.json"
        with open(full_results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Summary (overwrite)
        summary_dir = output_dir.parent / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"ep{num_ranks}_intranode_summary.json"

        summary = {
            "timestamp": results["timestamp"],
            "optimal_configs": results["optimal_configs"],
            "performance": results["performance"],
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Optimal dispatch config (2-param intranode): {best_dispatch.config}")
        print(f"Optimal combine config (2-param intranode): {best_combine.config}")
        print()
        print(f"For utils.py (4-param format):")
        print(f"  turbo_deepep_dispatch_tuned_config = {dispatch_config_4param}")
        print(f"  turbo_deepep_combine_tuned_config = {combine_config_4param}")
        print()
        print(
            f"Dispatch: {best_dispatch.time_us:.2f} us, {best_dispatch.bandwidth_gbps:.2f} GB/s"
        )
        print(
            f"Combine: {best_combine.time_us:.2f} us, {best_combine.bandwidth_gbps:.2f} GB/s"
        )
        print()
        print(f"Improvement vs worst:")
        print(
            f"  Dispatch: {results['performance']['dispatch']['improvement_vs_worst_pct']:.1f}%"
        )
        print(
            f"  Combine: {results['performance']['combine']['improvement_vs_worst_pct']:.1f}%"
        )
        print()
        print(f"Results saved to: {full_results_path}")
        print(f"Summary saved to: {summary_path}")
        print("=" * 80)


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    """Main entry point for each process"""
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    # Create DeepEP buffer (intranode mode - RDMA not used)
    buffer = deep_ep.Buffer(group, int(2e9), 0, explicitly_destroy=True)
    torch.manual_seed(rank)

    # Test with num_sms = 24 (standard for B200)
    test_main(args, 24, local_rank, num_ranks, rank, buffer, group)

    # Cleanup
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune DeepEP intranode configs for TorchTitan"
    )
    parser.add_argument(
        "--num-processes", type=int, default=4, help="Number of processes (default: 4)"
    )
    parser.add_argument(
        "--num-tokens", type=int, default=4096, help="Number of tokens (default: 4096)"
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=2048,
        help="Hidden dimension - Qwen3-30B (default: 2048)",
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=128,
        help="Number of experts - Qwen3-30B-A3B (default: 128)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for results"
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
