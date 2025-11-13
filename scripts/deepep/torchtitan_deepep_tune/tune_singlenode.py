#!/usr/bin/env python3
"""
Single-Node DeepEP Tuning for TorchTitan

Tunes DeepEP dispatch and combine configurations for single-node (intranode) setup.
For single node, only NVLink is used (no RDMA), so Config takes 3 params: (num_sms, nvl_chunk, nvl_buffer).

Based on: /home/phuc/workspace/moe/DeepEP/tests/test_intranode.py

Usage:
    torchrun --nproc_per_node=4 tune_singlenode.py --ep-size 4 --output results_ep4.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist

try:
    import deep_ep
except ImportError:
    print("ERROR: deep_ep not found. Install DeepEP first.")
    sys.exit(1)

# Import from DeepEP tests
sys.path.insert(0, "/home/phuc/workspace/moe/DeepEP/tests")
from utils import bench, init_dist, inplace_unique


@dataclass
class Config:
    """Single-node config: (num_sms, nvl_chunk, nvl_buffer)"""

    num_sms: int
    nvl_chunk: int
    nvl_buffer: int

    def to_deepep(self) -> deep_ep.Config:
        return deep_ep.Config(self.num_sms, self.nvl_chunk, self.nvl_buffer)

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.num_sms, self.nvl_chunk, self.nvl_buffer)


@dataclass
class BenchmarkResult:
    """Benchmark result with timing and bandwidth"""

    config: Config
    time_us: float
    bandwidth_gbps: float
    phase: str  # 'dispatch' or 'combine'


class SingleNodeTuner:
    """Tunes DeepEP for single-node (intranode) setup"""

    def __init__(
        self,
        num_tokens: int = 4096,
        hidden: int = 7168,
        num_experts: int = 256,
        num_topk: int = 8,
    ):
        self.num_tokens = num_tokens
        self.hidden = hidden
        self.num_experts = num_experts
        self.num_topk = num_topk

        # Init distributed
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", 8))
        self.rank, self.num_ranks, self.group = init_dist(
            self.local_rank, num_local_ranks
        )

        # Buffer sizes for single node
        self.nvl_buffer_size = 256  # From test_intranode.py line 75
        self.num_sms = 24  # Default SMs

        # Create buffer
        self.buffer = deep_ep.Buffer(
            self.group,
            int(2e9),  # nvl buffer
            0,  # no rdma for intranode
            low_latency_mode=False,
        )

        if self.is_rank0():
            print(f"\n{'='*80}")
            print(f"Single-Node DeepEP Tuner")
            print(f"{'='*80}")
            print(f"Ranks: {self.num_ranks}")
            print(f"Tokens: {num_tokens}, Hidden: {hidden}")
            print(f"Experts: {num_experts}, TopK: {num_topk}")
            print(f"NVL Buffer: {self.nvl_buffer_size}, SMs: {self.num_sms}")
            print(f"{'='*80}\n")

    def is_rank0(self) -> bool:
        return self.local_rank == 0

    def setup_data(self):
        """Setup test data (from test_intranode.py)"""
        # Create test tensors
        self.x = (
            torch.ones(
                (self.num_tokens, self.hidden), dtype=torch.bfloat16, device="cuda"
            )
            * self.rank
        )

        # Random scores and routing
        scores = (
            torch.randn(
                (self.num_tokens, self.num_experts), dtype=torch.float32, device="cuda"
            ).abs()
            + 1
        )
        self.topk_idx = torch.topk(
            scores, self.num_topk, dim=-1, largest=True, sorted=False
        )[1]
        self.topk_idx = self.topk_idx.to(deep_ep.topk_idx_t)
        self.topk_weights = (
            torch.ones(
                (self.num_tokens, self.num_topk), dtype=torch.float32, device="cuda"
            )
            * self.rank
        )

        # Compute rank indices
        rank_idx = self.topk_idx // (self.num_experts // self.num_ranks)
        rank_idx = rank_idx.to(torch.int64)
        rank_idx.masked_fill_(self.topk_idx == -1, -1)
        inplace_unique(rank_idx, self.num_ranks)

        # Get layout
        (
            self.num_tokens_per_rank,
            _,
            self.num_tokens_per_expert,
            self.is_token_in_rank,
            _,
        ) = self.buffer.get_dispatch_layout(self.topk_idx, self.num_experts)

        # Get dispatch handle for combine tuning
        config = Config(self.num_sms, 8, self.nvl_buffer_size).to_deepep()
        self.recv_x, _, _, _, self.handle, _ = self.buffer.dispatch(
            x=self.x,
            num_tokens_per_rank=self.num_tokens_per_rank,
            is_token_in_rank=self.is_token_in_rank,
            num_tokens_per_expert=self.num_tokens_per_expert,
            config=config,
        )

        # Calculate data size for bandwidth
        self.data_bytes = self.recv_x.numel() * 2  # BF16 = 2 bytes

    def tune_dispatch(
        self, nvl_chunk_range: List[int]
    ) -> Tuple[Config, List[BenchmarkResult]]:
        """Tune dispatch phase"""
        if self.is_rank0():
            print(f"\n{'='*80}")
            print(f"PHASE 1: Tuning Dispatch")
            print(f"Testing {len(nvl_chunk_range)} configurations...")
            print(f"{'='*80}\n")

        best_config = None
        best_time = float("inf")
        results = []

        for i, nvl_chunk in enumerate(nvl_chunk_range):
            config = Config(self.num_sms, nvl_chunk, self.nvl_buffer_size)

            try:
                # Benchmark dispatch
                t = bench(
                    lambda: self.buffer.dispatch(
                        x=self.x,
                        num_tokens_per_rank=self.num_tokens_per_rank,
                        is_token_in_rank=self.is_token_in_rank,
                        num_tokens_per_expert=self.num_tokens_per_expert,
                        config=config.to_deepep(),
                    )
                )[0]

                time_us = t * 1e6
                bw_gbps = (self.data_bytes / 1e9) / t

                result = BenchmarkResult(config, time_us, bw_gbps, "dispatch")
                results.append(result)

                if t < best_time:
                    best_time = t
                    best_config = config
                    marker = "★"
                else:
                    marker = " "

                if self.is_rank0():
                    print(
                        f"[{i+1:2d}/{len(nvl_chunk_range)}] {marker} nvl_chunk={nvl_chunk:2d} | {time_us:6.0f}µs | {bw_gbps:6.1f} GB/s"
                    )

            except Exception as e:
                if self.is_rank0():
                    print(
                        f"[{i+1:2d}/{len(nvl_chunk_range)}]   nvl_chunk={nvl_chunk:2d} | ERROR: {e}"
                    )

        if self.is_rank0():
            print(
                f"\n[BEST DISPATCH] {best_config.as_tuple()} | {best_time*1e6:.0f}µs | {(self.data_bytes/1e9)/best_time:.1f} GB/s\n"
            )

        return best_config, results

    def tune_combine(
        self, dispatch_config: Config, nvl_chunk_range: List[int]
    ) -> Tuple[Config, List[BenchmarkResult]]:
        """Tune combine phase"""
        if self.is_rank0():
            print(f"\n{'='*80}")
            print(f"PHASE 2: Tuning Combine")
            print(f"Testing {len(nvl_chunk_range)} configurations...")
            print(f"{'='*80}\n")

        best_config = None
        best_time = float("inf")
        results = []

        for i, nvl_chunk in enumerate(nvl_chunk_range):
            config = Config(self.num_sms, nvl_chunk, self.nvl_buffer_size)

            try:
                # Benchmark combine
                t = bench(
                    lambda: self.buffer.combine(
                        x=self.recv_x,
                        handle=self.handle,
                        config=config.to_deepep(),
                    )
                )[0]

                time_us = t * 1e6
                bw_gbps = (self.data_bytes / 1e9) / t

                result = BenchmarkResult(config, time_us, bw_gbps, "combine")
                results.append(result)

                if t < best_time:
                    best_time = t
                    best_config = config
                    marker = "★"
                else:
                    marker = " "

                if self.is_rank0():
                    print(
                        f"[{i+1:2d}/{len(nvl_chunk_range)}] {marker} nvl_chunk={nvl_chunk:2d} | {time_us:6.0f}µs | {bw_gbps:6.1f} GB/s"
                    )

            except Exception as e:
                if self.is_rank0():
                    print(
                        f"[{i+1:2d}/{len(nvl_chunk_range)}]   nvl_chunk={nvl_chunk:2d} | ERROR: {e}"
                    )

        if self.is_rank0():
            print(
                f"\n[BEST COMBINE] {best_config.as_tuple()} | {best_time*1e6:.0f}µs | {(self.data_bytes/1e9)/best_time:.1f} GB/s\n"
            )

        return best_config, results

    def cleanup(self):
        """Cleanup resources"""
        self.buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Tune single-node DeepEP config")
    parser.add_argument(
        "--ep-size", type=int, required=True, help="Expert parallel size (GPUs)"
    )
    parser.add_argument("--num-tokens", type=int, default=4096, help="Tokens per batch")
    parser.add_argument("--hidden", type=int, default=7168, help="Hidden dimension")
    parser.add_argument("--num-experts", type=int, default=256, help="Total experts")
    parser.add_argument("--num-topk", type=int, default=8, help="Router TopK")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="Search mode",
    )
    parser.add_argument("--output", default="results.json", help="Output JSON file")

    args = parser.parse_args()

    # Create tuner
    tuner = SingleNodeTuner(
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        num_experts=args.num_experts,
        num_topk=args.num_topk,
    )

    # Setup test data
    tuner.setup_data()

    # Define search space
    if args.mode == "quick":
        nvl_range = [4, 8, 12, 16, 20, 24]
    else:  # full
        nvl_range = list(range(2, 33, 2))

    # Run tuning
    start_time = time.time()
    best_dispatch, dispatch_results = tuner.tune_dispatch(nvl_range)
    best_combine, combine_results = tuner.tune_combine(best_dispatch, nvl_range)
    total_time = time.time() - start_time

    # Save results (rank 0 only)
    if tuner.is_rank0():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Calculate metrics
        dispatch_times = [r.time_us for r in dispatch_results]
        combine_times = [r.time_us for r in combine_results]
        worst_dispatch = max(dispatch_times)
        worst_combine = max(combine_times)
        best_dispatch_time = best_dispatch.as_tuple()[-1]  # Get time from results
        best_combine_time = best_combine.as_tuple()[-1]

        # Find actual best times
        best_d_result = min(dispatch_results, key=lambda r: r.time_us)
        best_c_result = min(combine_results, key=lambda r: r.time_us)
        worst_d_result = max(dispatch_results, key=lambda r: r.time_us)
        worst_c_result = max(combine_results, key=lambda r: r.time_us)

        # Calculate improvements
        dispatch_improvement = (
            (worst_d_result.time_us - best_d_result.time_us) / worst_d_result.time_us
        ) * 100
        combine_improvement = (
            (worst_c_result.time_us - best_c_result.time_us) / worst_c_result.time_us
        ) * 100

        result = {
            "timestamp": timestamp,
            "ep_size": args.ep_size,
            "num_ranks": tuner.num_ranks,
            "config": {
                "dispatch": best_dispatch.as_tuple(),
                "combine": best_combine.as_tuple(),
            },
            "performance": {
                "dispatch": {
                    "best_time_us": best_d_result.time_us,
                    "best_bandwidth_gbps": best_d_result.bandwidth_gbps,
                    "worst_time_us": worst_d_result.time_us,
                    "improvement_pct": dispatch_improvement,
                },
                "combine": {
                    "best_time_us": best_c_result.time_us,
                    "best_bandwidth_gbps": best_c_result.bandwidth_gbps,
                    "worst_time_us": worst_c_result.time_us,
                    "improvement_pct": combine_improvement,
                },
            },
            "parameters": {
                "num_tokens": args.num_tokens,
                "hidden": args.hidden,
                "num_experts": args.num_experts,
                "num_topk": args.num_topk,
            },
            "total_time_sec": total_time,
        }

        # Save to output file
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        # Also save detailed log
        log_dir = Path("scripts/deepep/torchtitan_deepep_tune/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"tune_ep{args.ep_size}_{timestamp}.json"
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)

        # Print summary
        print(f"\n{'='*80}")
        print(f"TUNING COMPLETE")
        print(f"{'='*80}")
        print(f"Optimal Configuration:")
        print(f"  Dispatch: {best_dispatch.as_tuple()}")
        print(f"  Combine:  {best_combine.as_tuple()}")
        print(f"\nPerformance Improvement:")
        print(f"  Dispatch: {dispatch_improvement:.1f}% faster (worst→optimal)")
        print(f"  Combine:  {combine_improvement:.1f}% faster (worst→optimal)")
        print(f"\nBest Bandwidth:")
        print(f"  Dispatch: {best_d_result.bandwidth_gbps:.1f} GB/s")
        print(f"  Combine:  {best_c_result.bandwidth_gbps:.1f} GB/s")
        print(f"\nResults saved:")
        print(f"  {output_path}")
        print(f"  {log_path}")
        print(f"\nTo use in torchtitan:")
        print(
            f"  PrimusTurboFlexTokenDispatcher.turbo_deepep_dispatch_tuned_config = {best_dispatch.as_tuple()}"
        )
        print(
            f"  PrimusTurboFlexTokenDispatcher.turbo_deepep_combine_tuned_config = {best_combine.as_tuple()}"
        )
        print(f"{'='*80}\n")

    # Cleanup
    tuner.cleanup()


if __name__ == "__main__":
    main()
