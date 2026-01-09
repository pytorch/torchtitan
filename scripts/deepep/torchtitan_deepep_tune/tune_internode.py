#!/usr/bin/env python3
"""
DeepEP Internode Tuning for TorchTitan (matches actual training setup)

This script tunes DeepEP for the ACTUAL setup used in torchtitan:
- B200 GPUs (not H800)
- Qwen3-30B-A3B: dim=2048, 128 experts, top_k=8
- Internode format: (nvl_chunk, nvl_buffer, rdma_chunk, rdma_buffer)

Based on: /home/phuc/workspace/moe/DeepEP/tests/test_internode.py
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
    print("ERROR: deep_ep not found")
    sys.exit(1)

sys.path.insert(0, "/home/phuc/workspace/moe/DeepEP/tests")
from utils import bench_kineto, init_dist, inplace_unique


@dataclass
class Config:
    """Internode config: (nvl_chunk, nvl_buffer, rdma_chunk, rdma_buffer)"""

    nvl_chunk: int
    nvl_buffer: int
    rdma_chunk: int
    rdma_buffer: int

    def to_deepep(self, num_sms: int) -> deep_ep.Config:
        """Convert to DeepEP Config with 5 params"""
        return deep_ep.Config(
            num_sms,
            self.nvl_chunk,
            self.nvl_buffer,
            self.rdma_chunk,
            self.rdma_buffer,
        )

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.nvl_chunk, self.nvl_buffer, self.rdma_chunk, self.rdma_buffer)


class DeepEPTuner:
    """Tunes DeepEP for actual torchtitan setup"""

    def __init__(
        self,
        num_tokens: int = 4096,
        hidden: int = 2048,  # Qwen3-30B dim
        num_experts: int = 128,  # Qwen3-30B-A3B
        num_topk: int = 8,
        num_topk_groups: int = 4,
    ):
        self.num_tokens = num_tokens
        self.hidden = hidden
        self.num_experts = num_experts
        self.num_topk = num_topk
        self.num_topk_groups = num_topk_groups

        # Init distributed
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", 8))
        self.rank, self.num_ranks, self.group = init_dist(
            self.local_rank, num_local_ranks
        )
        self.num_nodes = int(os.environ.get("WORLD_SIZE", 1))
        self.num_sms = 24

        # Buffer sizes (from benchmark_internode.py)
        self.nvl_buffer_size = 720 if self.num_ranks in (24, 48, 96, 144, 160) else 512
        self.rdma_buffer_size = 128

        # Create buffer
        num_qps_per_rank = max(self.num_sms, 0)
        self.buffer = deep_ep.Buffer(
            self.group,
            int(2e9),
            int(1e9),
            low_latency_mode=False,
            num_qps_per_rank=num_qps_per_rank,
            explicitly_destroy=True,
        )

        if self.local_rank == 0:
            print(f"\n{'='*80}")
            print(f"DeepEP Tuner (Internode Format)")
            print(f"{'='*80}")
            print(f"Hardware: B200 GPUs")
            print(f"Ranks: {self.num_ranks} (nodes: {self.num_nodes})")
            print(f"Model: Qwen3-30B-A3B")
            print(f"  Tokens: {num_tokens}, Hidden: {hidden}")
            print(
                f"  Experts: {num_experts}, TopK: {num_topk}, TopK Groups: {num_topk_groups}"
            )
            print(f"Buffers: NVL={self.nvl_buffer_size}, RDMA={self.rdma_buffer_size}")
            print(f"{'='*80}\n")

    def is_rank0(self) -> bool:
        return self.local_rank == 0

    def setup_data(self):
        """Setup test data (from test_internode.py)"""
        # Test tensors
        self.x = (
            torch.ones(
                (self.num_tokens, self.hidden), dtype=torch.bfloat16, device="cuda"
            )
            * self.rank
        )

        # Random scores with group-based routing (like Qwen3)
        scores = (
            torch.randn(
                (self.num_tokens, self.num_experts), dtype=torch.float32, device="cuda"
            ).abs()
            + 1
        )
        group_scores = scores.view(self.num_tokens, self.num_nodes, -1).amax(dim=-1)
        group_idx = torch.topk(
            group_scores, k=self.num_topk_groups, dim=-1, sorted=False
        ).indices

        # Create grouped scores (group-limited routing)
        masked_scores = scores.clone()
        for i in range(self.num_nodes):
            mask = (group_idx == i).any(dim=-1, keepdim=True)
            node_mask = torch.zeros(
                self.num_tokens, self.num_experts, dtype=torch.bool, device="cuda"
            )
            start_expert = i * (self.num_experts // self.num_nodes)
            end_expert = (i + 1) * (self.num_experts // self.num_nodes)
            node_mask[:, start_expert:end_expert] = True
            masked_scores = torch.where(
                mask & node_mask,
                masked_scores,
                torch.tensor(-float("inf"), device="cuda"),
            )

        self.topk_idx = torch.topk(
            masked_scores, self.num_topk, dim=-1, largest=True, sorted=False
        )[1]
        self.topk_idx = self.topk_idx.to(deep_ep.topk_idx_t)

        # Get layout
        (
            self.num_tokens_per_rank,
            self.num_tokens_per_rdma_rank,
            self.num_tokens_per_expert,
            self.is_token_in_rank,
            _,
        ) = self.buffer.get_dispatch_layout(self.topk_idx, self.num_experts)

        # Get dispatch handle for combine tuning
        config = Config(8, self.nvl_buffer_size, 16, self.rdma_buffer_size).to_deepep(
            self.num_sms
        )
        self.recv_x, _, _, _, self.handle, _ = self.buffer.dispatch(
            x=self.x,
            num_tokens_per_rank=self.num_tokens_per_rank,
            num_tokens_per_rdma_rank=self.num_tokens_per_rdma_rank,
            is_token_in_rank=self.is_token_in_rank,
            num_tokens_per_expert=self.num_tokens_per_expert,
            config=config,
        )

        # Calculate bandwidth metrics
        rank_idx = self.topk_idx // (self.num_experts // self.num_ranks)
        rank_idx = rank_idx.to(torch.int64)
        rank_idx.masked_fill_(self.topk_idx == -1, -1)
        rdma_idx = self.topk_idx // (self.num_experts // self.num_nodes)
        rdma_idx.masked_fill_(self.topk_idx == -1, -1)
        num_rdma_token_sent = rdma_idx.ne(-1).sum().item()

        self.dispatch_rdma_send_bytes = num_rdma_token_sent * self.hidden * 2
        self.dispatch_nvl_recv_bytes = self.recv_x.numel() * 2

    def tune_dispatch(self, nvl_range: List[int], rdma_range: List[int]):
        """Tune dispatch phase"""
        if self.is_rank0():
            print(f"\n{'='*80}")
            print(f"PHASE 1: Tuning Dispatch")
            print(
                f"Testing {len(nvl_range)} x {len(rdma_range)} = {len(nvl_range)*len(rdma_range)} configs"
            )
            print(f"{'='*80}\n")

        best_time = float("inf")
        best_config = None
        results = []

        total = len(nvl_range) * len(rdma_range)
        count = 0

        for nvl_chunk in nvl_range:
            for rdma_chunk in rdma_range:
                count += 1
                config = Config(
                    nvl_chunk, self.nvl_buffer_size, rdma_chunk, self.rdma_buffer_size
                )

                try:
                    tune_args = {
                        "x": self.x,
                        "handle": self.handle,
                        "config": config.to_deepep(self.num_sms),
                    }
                    t, notify_t = bench_kineto(
                        lambda: self.buffer.dispatch(**tune_args),
                        ("dispatch", "notify"),
                        suppress_kineto_output=True,
                    )

                    rdma_bw = self.dispatch_rdma_send_bytes / 1e9 / t
                    nvl_bw = self.dispatch_nvl_recv_bytes / 1e9 / t

                    results.append(
                        {
                            "config": config.as_tuple(),
                            "time_us": t * 1e6,
                            "rdma_gbps": rdma_bw,
                            "nvl_gbps": nvl_bw,
                        }
                    )

                    if t < best_time:
                        best_time = t
                        best_config = config
                        marker = "★"
                    else:
                        marker = " "

                    if self.is_rank0():
                        print(
                            f"[{count:3d}/{total}] {marker} NVL={nvl_chunk:2d} RDMA={rdma_chunk:2d} | "
                            f"{t*1e6:6.0f}µs | RDMA:{rdma_bw:5.1f} NVL:{nvl_bw:5.1f} GB/s"
                        )

                except Exception as e:
                    if self.is_rank0():
                        print(
                            f"[{count:3d}/{total}]   NVL={nvl_chunk:2d} RDMA={rdma_chunk:2d} | ERROR: {e}"
                        )

        if self.is_rank0():
            print(
                f"\n[BEST DISPATCH] {best_config.as_tuple()} | {best_time*1e6:.0f}µs\n"
            )

        return best_config, results

    def tune_combine(
        self, dispatch_config: Config, nvl_range: List[int], rdma_range: List[int]
    ):
        """Tune combine phase"""
        if self.is_rank0():
            print(f"\n{'='*80}")
            print(f"PHASE 2: Tuning Combine")
            print(
                f"Testing {len(nvl_range)} x {len(rdma_range)} = {len(nvl_range)*len(rdma_range)} configs"
            )
            print(f"{'='*80}\n")

        best_time = float("inf")
        best_config = None
        results = []

        total = len(nvl_range) * len(rdma_range)
        count = 0

        for nvl_chunk in nvl_range:
            for rdma_chunk in rdma_range:
                count += 1
                config = Config(
                    nvl_chunk, self.nvl_buffer_size, rdma_chunk, self.rdma_buffer_size
                )

                try:
                    tune_args = {
                        "x": self.recv_x,
                        "handle": self.handle,
                        "config": config.to_deepep(self.num_sms),
                    }
                    t, notify_t = bench_kineto(
                        lambda: self.buffer.combine(**tune_args),
                        ("combine", "notify"),
                        suppress_kineto_output=True,
                    )

                    rdma_bw = self.dispatch_rdma_send_bytes / 1e9 / t
                    nvl_bw = self.dispatch_nvl_recv_bytes / 1e9 / t

                    results.append(
                        {
                            "config": config.as_tuple(),
                            "time_us": t * 1e6,
                            "rdma_gbps": rdma_bw,
                            "nvl_gbps": nvl_bw,
                        }
                    )

                    if t < best_time:
                        best_time = t
                        best_config = config
                        marker = "★"
                    else:
                        marker = " "

                    if self.is_rank0():
                        print(
                            f"[{count:3d}/{total}] {marker} NVL={nvl_chunk:2d} RDMA={rdma_chunk:2d} | "
                            f"{t*1e6:6.0f}µs | RDMA:{rdma_bw:5.1f} NVL:{nvl_bw:5.1f} GB/s"
                        )

                except Exception as e:
                    if self.is_rank0():
                        print(
                            f"[{count:3d}/{total}]   NVL={nvl_chunk:2d} RDMA={rdma_chunk:2d} | ERROR: {e}"
                        )

        if self.is_rank0():
            print(
                f"\n[BEST COMBINE] {best_config.as_tuple()} | {best_time*1e6:.0f}µs\n"
            )

        return best_config, results

    def cleanup(self):
        """Cleanup"""
        self.buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Tune DeepEP for actual torchtitan setup"
    )
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--mode", choices=["quick", "medium", "full"], default="medium")
    parser.add_argument("--output-dir", default="results")

    args = parser.parse_args()

    # Qwen3-30B-A3B parameters
    tuner = DeepEPTuner(
        num_tokens=4096,
        hidden=2048,  # Qwen3-30B dim
        num_experts=128,  # Qwen3-30B-A3B
        num_topk=8,
        num_topk_groups=4,
    )

    tuner.setup_data()

    # Define search ranges based on mode
    if args.mode == "quick":
        nvl_dispatch_range = [4, 8, 12, 16]
        rdma_dispatch_range = [4, 8, 12, 16]
        nvl_combine_range = [1, 2, 4, 6]
        rdma_combine_range = [8, 12, 16]
    elif args.mode == "medium":
        nvl_dispatch_range = list(range(4, 25, 4))
        rdma_dispatch_range = list(range(4, 21, 4))
        nvl_combine_range = list(range(1, 8, 1))
        rdma_combine_range = list(range(8, 21, 4))
    else:  # full
        nvl_dispatch_range = list(range(4, 45, 4))
        rdma_dispatch_range = list(range(4, 33, 4))
        nvl_combine_range = list(range(1, 8, 1))
        rdma_combine_range = list(range(8, 33, 4))

    # Run tuning
    start = time.time()
    best_dispatch, dispatch_results = tuner.tune_dispatch(
        nvl_dispatch_range, rdma_dispatch_range
    )
    best_combine, combine_results = tuner.tune_combine(
        best_dispatch, nvl_combine_range, rdma_combine_range
    )
    elapsed = time.time() - start

    # Save results (rank 0 only)
    if tuner.is_rank0():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Find worst configs
        worst_dispatch = max(dispatch_results, key=lambda x: x["time_us"])
        best_dispatch_result = min(dispatch_results, key=lambda x: x["time_us"])
        worst_combine = max(combine_results, key=lambda x: x["time_us"])
        best_combine_result = min(combine_results, key=lambda x: x["time_us"])

        # Calculate improvements
        dispatch_improvement = (
            (worst_dispatch["time_us"] - best_dispatch_result["time_us"])
            / worst_dispatch["time_us"]
        ) * 100
        combine_improvement = (
            (worst_combine["time_us"] - best_combine_result["time_us"])
            / worst_combine["time_us"]
        ) * 100

        result = {
            "timestamp": timestamp,
            "hardware": "B200",
            "model": "Qwen3-30B-A3B",
            "ep_size": args.ep_size,
            "num_ranks": tuner.num_ranks,
            "optimal_config": {
                "dispatch": best_dispatch.as_tuple(),
                "combine": best_combine.as_tuple(),
            },
            "performance": {
                "dispatch": {
                    "best": best_dispatch_result,
                    "worst": worst_dispatch,
                    "improvement_pct": dispatch_improvement,
                },
                "combine": {
                    "best": best_combine_result,
                    "worst": worst_combine,
                    "improvement_pct": combine_improvement,
                },
            },
            "all_results": {
                "dispatch": dispatch_results,
                "combine": combine_results,
            },
            "parameters": {
                "num_tokens": tuner.num_tokens,
                "hidden": tuner.hidden,
                "num_experts": tuner.num_experts,
                "num_topk": tuner.num_topk,
            },
            "tuning_time_sec": elapsed,
        }

        # Save full results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / f"ep{args.ep_size}_{timestamp}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        # Create summary
        summary_dir = Path("summary")
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "timestamp": timestamp,
            "hardware": "B200",
            "model": "Qwen3-30B-A3B (dim=2048, 128 experts, top-k=8)",
            "ep_size": args.ep_size,
            "optimal_config": {
                "dispatch": best_dispatch.as_tuple(),
                "combine": best_combine.as_tuple(),
            },
            "improvement": {
                "dispatch": f"{dispatch_improvement:.1f}%",
                "combine": f"{combine_improvement:.1f}%",
            },
            "best_performance": {
                "dispatch_us": best_dispatch_result["time_us"],
                "combine_us": best_combine_result["time_us"],
            },
        }
        summary_file = summary_dir / f"ep{args.ep_size}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print(f"\n{'='*80}")
        print(f"TUNING COMPLETE")
        print(f"{'='*80}")
        print(f"Hardware: B200")
        print(f"Model: Qwen3-30B-A3B (dim={tuner.hidden}, {tuner.num_experts} experts)")
        print(f"\nOptimal Configuration:")
        print(f"  Dispatch: {best_dispatch.as_tuple()}")
        print(f"  Combine:  {best_combine.as_tuple()}")
        print(f"\nPerformance Improvement (worst → best):")
        print(
            f"  Dispatch: {dispatch_improvement:.1f}% faster ({worst_dispatch['time_us']:.0f}µs → {best_dispatch_result['time_us']:.0f}µs)"
        )
        print(
            f"  Combine:  {combine_improvement:.1f}% faster ({worst_combine['time_us']:.0f}µs → {best_combine_result['time_us']:.0f}µs)"
        )
        print(f"\nBest Bandwidth:")
        print(
            f"  Dispatch: RDMA={best_dispatch_result['rdma_gbps']:.1f} GB/s, NVL={best_dispatch_result['nvl_gbps']:.1f} GB/s"
        )
        print(
            f"  Combine:  RDMA={best_combine_result['rdma_gbps']:.1f} GB/s, NVL={best_combine_result['nvl_gbps']:.1f} GB/s"
        )
        print(f"\nResults saved:")
        print(f"  Full:    {result_file}")
        print(f"  Summary: {summary_file}")
        print(f"\nTo use in torchtitan/distributed/deepep/utils.py:")
        print(f"  turbo_deepep_dispatch_tuned_config = {best_dispatch.as_tuple()}")
        print(f"  turbo_deepep_combine_tuned_config = {best_combine.as_tuple()}")
        print(f"{'='*80}\n")

    tuner.cleanup()


if __name__ == "__main__":
    main()
