# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepEP Benchmark Script for Multi-Node Tuning.

Sweeps over seq_len, batch_size, and model configs to find optimal DeepEP settings.
Uses the same tuning approach as tune_intranode_v2.py with proper handle reuse.

Usage:
    # Default sweep with Kimi K2 model config
    torchrun --nproc_per_node=8 -m torchtitan.distributed.deepep.benchmark

    # Use a specific model preset
    torchrun --nproc_per_node=8 -m torchtitan.distributed.deepep.benchmark --model qwen3_30b_a3b

    # Custom model params
    torchrun --nproc_per_node=8 -m torchtitan.distributed.deepep.benchmark \
        --hidden 7168 --num_experts 256 --topk 8

    # Custom sweep range
    torchrun --nproc_per_node=8 -m torchtitan.distributed.deepep.benchmark \
        --seq_len 4096,8192,16384 --batch_size 1,2,4
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import torch
import torch.distributed as dist

try:
    from deep_ep import Buffer, Config
except ImportError as e:
    raise ImportError(
        "DeepEP is required. Install from: https://github.com/deepseek-ai/deepep"
    ) from e


# Simple logger that only prints on rank 0
class _Logger:
    def info(self, msg):
        print(f"[INFO] {msg}")

    def error(self, msg):
        print(f"[ERROR] {msg}")

    def warning(self, msg):
        print(f"[WARN] {msg}")


logger = _Logger()

# =============================================================================
# MODEL PRESETS
# =============================================================================
# Define model configs here - easy to add new models

MODEL_PRESETS = {
    "kimi_k2": {
        "hidden_dim": 7168,
        "num_experts": 384,
        "num_topk": 8,
        "description": "Kimi K2 (1T MoE)",
    },
    "deepseek_v3": {
        "hidden_dim": 7168,
        "num_experts": 256,
        "num_topk": 8,
        "description": "DeepSeek V3 (671B MoE)",
    },
    "qwen3_30b_a3b": {
        "hidden_dim": 2048,
        "num_experts": 128,
        "num_topk": 8,
        "description": "Qwen3 30B-A3B MoE",
    },
    "qwen3_235b_a22b": {
        "hidden_dim": 4096,
        "num_experts": 128,
        "num_topk": 8,
        "description": "Qwen3 235B-A22B MoE",
    },
    "mixtral_8x7b": {
        "hidden_dim": 4096,
        "num_experts": 8,
        "num_topk": 2,
        "description": "Mixtral 8x7B",
    },
    "mixtral_8x22b": {
        "hidden_dim": 6144,
        "num_experts": 8,
        "num_topk": 2,
        "description": "Mixtral 8x22B",
    },
    # Debug model for testing
    "debug": {
        "hidden_dim": 256,
        "num_experts": 8,
        "num_topk": 2,
        "description": "Debug model (small)",
    },
}

# Default sweep ranges
DEFAULT_SEQ_LENS = [2048, 4096, 8192, 16384, 24576, 32768]
DEFAULT_BATCH_SIZES = [1, 2, 4]


@dataclass
class ModelConfig:
    """Model configuration for benchmarking."""

    hidden_dim: int
    num_experts: int
    num_topk: int
    name: str = "custom"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    seq_len: int
    batch_size: int
    num_tokens: int
    # Dispatch results
    dispatch_config: Tuple[int, ...]
    dispatch_time_us: float
    dispatch_bandwidth_gbps: float
    # Combine results
    combine_config: Tuple[int, ...]
    combine_time_us: float
    combine_bandwidth_gbps: float
    # Config details
    best_num_sms: int
    is_internode: bool


def _parse_int_list(s: str) -> List[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(",")]


def _get_gpu_sm_range() -> List[int]:
    """Auto-detect GPU type and return appropriate SM search range."""
    try:
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "b200" in gpu_name or "b100" in gpu_name:
            return [24, 32, 48, 64]
        elif "h200" in gpu_name or "h100" in gpu_name:
            return [16, 20, 24, 28, 32]
        elif "a100" in gpu_name:
            return [16, 20, 24, 28, 32]
        else:
            return [24]
    except Exception:
        return [24]


def _detect_internode(buffer: Buffer) -> Tuple[bool, int, int]:
    """Detect if communication requires internode (RDMA) or is intranode only."""
    import os

    num_ranks = buffer.group_size
    num_rdma_ranks = buffer.runtime.get_num_rdma_ranks()
    is_internode = num_rdma_ranks > 1

    if is_internode:
        num_nodes = num_rdma_ranks
        local_world_size = num_ranks // num_nodes
    else:
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", num_ranks))
        num_nodes = 1

    return is_internode, local_world_size, num_nodes


def _bench_fn(fn, warmup: int = 3, repeat: int = 5) -> float:
    """Benchmark a function and return average time in seconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / repeat


def tune_single_config(
    buffer: Buffer,
    num_tokens: int,
    hidden_dim: int,
    num_experts: int,
    num_topk: int,
    sms_range: List[int],
    is_internode: bool,
    warmup: int = 3,
    repeat: int = 5,
    rank: int = 0,
) -> Tuple[Tuple[int, ...], float, float, Tuple[int, ...], float, float, int]:
    """
    Tune dispatch and combine for a single num_tokens configuration.

    Tunes over:
    - num_sms: Number of SMs to use
    - nvl_chunk: NVLink chunk size
    - rdma_chunk: RDMA chunk size (internode only)

    For internode mode, uses joint num_sms tuning (dispatch+combine must use same num_sms).

    Returns:
        (dispatch_config, dispatch_time_us, dispatch_bw,
         combine_config, combine_time_us, combine_bw, best_num_sms)
    """
    num_ranks = buffer.group_size

    # Create test data
    x = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]

    # Get dispatch layout
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)

    # Buffer size ranges to tune
    nvl_buffer_sizes = [256, 512, 1024]
    rdma_buffer_sizes = [64, 128, 256]

    # Config ranges based on mode
    if is_internode:
        nvl_dispatch_range = list(
            range(4, 48, 8)
        )  # Coarser for speed: 4,12,20,28,36,44
        rdma_dispatch_range = list(range(4, 36, 8))  # 4,12,20,28
        nvl_combine_range = list(range(1, 10, 2))  # 1,3,5,7,9
        rdma_combine_range = list(range(8, 36, 8))  # 8,16,24,32
    else:
        nvl_dispatch_range = list(range(4, 34, 4))  # 4,8,12,16,20,24,28,32
        rdma_dispatch_range = [16]  # dummy
        nvl_combine_range = list(range(1, 17, 2))  # 1,3,5,7,9,11,13,15
        rdma_combine_range = [16]  # dummy
        rdma_buffer_sizes = [128]  # dummy for intranode

    def make_config(
        sms: int,
        nvl_chunk: int,
        nvl_buf: int,
        rdma_chunk: int = 16,
        rdma_buf: int = 128,
    ) -> Config:
        if is_internode:
            return Config(sms, nvl_chunk, nvl_buf, rdma_chunk, rdma_buf)
        else:
            return Config(sms, nvl_chunk, nvl_buf)

    # Initial dispatch to get handle and measure bytes
    initial_config = make_config(
        sms_range[0],
        nvl_dispatch_range[0],
        nvl_buffer_sizes[0],
        rdma_dispatch_range[0],
        rdma_buffer_sizes[0],
    )
    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=scores.gather(1, topk_idx).float(),
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=initial_config,
    )

    dispatch_recv_bytes = recv_x.numel() * 2
    combine_send_bytes = dispatch_recv_bytes

    # ============================================================================
    # TUNE num_sms, nvl_chunk, nvl_buffer, rdma_chunk, rdma_buffer
    # For internode: joint tuning (same num_sms for dispatch and combine)
    # For intranode: can tune independently
    # ============================================================================

    best_total_time = float("inf")
    best_num_sms = sms_range[0]
    best_dispatch_nvl = nvl_dispatch_range[0]
    best_dispatch_nvl_buf = nvl_buffer_sizes[0]
    best_dispatch_rdma = rdma_dispatch_range[0]
    best_dispatch_rdma_buf = rdma_buffer_sizes[0]
    best_dispatch_time = float("inf")
    best_combine_nvl = nvl_combine_range[0]
    best_combine_nvl_buf = nvl_buffer_sizes[0]
    best_combine_rdma = rdma_combine_range[0]
    best_combine_rdma_buf = rdma_buffer_sizes[0]
    best_combine_time = float("inf")

    for num_sms in sms_range:
        # ============ TUNE DISPATCH for this num_sms ============
        sms_best_dispatch_time = float("inf")
        sms_best_dispatch_nvl = nvl_dispatch_range[0]
        sms_best_dispatch_nvl_buf = nvl_buffer_sizes[0]
        sms_best_dispatch_rdma = rdma_dispatch_range[0]
        sms_best_dispatch_rdma_buf = rdma_buffer_sizes[0]

        # Get fresh handle for this num_sms
        try:
            init_cfg = make_config(
                num_sms,
                nvl_dispatch_range[0],
                nvl_buffer_sizes[0],
                rdma_dispatch_range[0],
                rdma_buffer_sizes[0],
            )
            recv_x, _, _, _, handle, _ = buffer.dispatch(
                x,
                topk_idx=topk_idx,
                topk_weights=scores.gather(1, topk_idx).float(),
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                config=init_cfg,
            )
        except RuntimeError:
            continue

        for nvl_chunk in nvl_dispatch_range:
            for nvl_buf in nvl_buffer_sizes:
                for rdma_chunk in rdma_dispatch_range:
                    for rdma_buf in rdma_buffer_sizes:
                        config = make_config(
                            num_sms, nvl_chunk, nvl_buf, rdma_chunk, rdma_buf
                        )

                        def dispatch_fn():
                            buffer.dispatch(x, handle=handle, config=config)

                        try:
                            t = _bench_fn(dispatch_fn, warmup, repeat)
                        except RuntimeError:
                            continue

                        if t < sms_best_dispatch_time:
                            sms_best_dispatch_time = t
                            sms_best_dispatch_nvl = nvl_chunk
                            sms_best_dispatch_nvl_buf = nvl_buf
                            sms_best_dispatch_rdma = rdma_chunk
                            sms_best_dispatch_rdma_buf = rdma_buf

        if sms_best_dispatch_time == float("inf"):
            continue

        # Re-dispatch with best dispatch config to get handle for combine
        try:
            best_disp_cfg = make_config(
                num_sms,
                sms_best_dispatch_nvl,
                sms_best_dispatch_nvl_buf,
                sms_best_dispatch_rdma,
                sms_best_dispatch_rdma_buf,
            )
            recv_x, _, _, _, handle, _ = buffer.dispatch(
                x,
                topk_idx=topk_idx,
                topk_weights=scores.gather(1, topk_idx).float(),
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                config=best_disp_cfg,
            )
        except RuntimeError:
            continue

        # ============ TUNE COMBINE for this num_sms ============
        sms_best_combine_time = float("inf")
        sms_best_combine_nvl = nvl_combine_range[0]
        sms_best_combine_nvl_buf = nvl_buffer_sizes[0]
        sms_best_combine_rdma = rdma_combine_range[0]
        sms_best_combine_rdma_buf = rdma_buffer_sizes[0]

        for nvl_chunk in nvl_combine_range:
            for nvl_buf in nvl_buffer_sizes:
                for rdma_chunk in rdma_combine_range:
                    for rdma_buf in rdma_buffer_sizes:
                        config = make_config(
                            num_sms, nvl_chunk, nvl_buf, rdma_chunk, rdma_buf
                        )

                        def combine_fn():
                            buffer.combine(recv_x, handle=handle, config=config)

                        try:
                            t = _bench_fn(combine_fn, warmup, repeat)
                        except RuntimeError:
                            continue

                        if t < sms_best_combine_time:
                            sms_best_combine_time = t
                            sms_best_combine_nvl = nvl_chunk
                            sms_best_combine_nvl_buf = nvl_buf
                            sms_best_combine_rdma = rdma_chunk
                            sms_best_combine_rdma_buf = rdma_buf

        if sms_best_combine_time == float("inf"):
            continue

        # Check if this num_sms is best overall
        total_time = sms_best_dispatch_time + sms_best_combine_time
        if total_time < best_total_time:
            best_total_time = total_time
            best_num_sms = num_sms
            best_dispatch_time = sms_best_dispatch_time
            best_dispatch_nvl = sms_best_dispatch_nvl
            best_dispatch_nvl_buf = sms_best_dispatch_nvl_buf
            best_dispatch_rdma = sms_best_dispatch_rdma
            best_dispatch_rdma_buf = sms_best_dispatch_rdma_buf
            best_combine_time = sms_best_combine_time
            best_combine_nvl = sms_best_combine_nvl
            best_combine_nvl_buf = sms_best_combine_nvl_buf
            best_combine_rdma = sms_best_combine_rdma
            best_combine_rdma_buf = sms_best_combine_rdma_buf

    # Build result configs
    if is_internode:
        dispatch_cfg = (
            best_dispatch_nvl,
            best_dispatch_nvl_buf,
            best_dispatch_rdma,
            best_dispatch_rdma_buf,
        )
        combine_cfg = (
            best_combine_nvl,
            best_combine_nvl_buf,
            best_combine_rdma,
            best_combine_rdma_buf,
        )
    else:
        dispatch_cfg = (best_dispatch_nvl, best_dispatch_nvl_buf)
        combine_cfg = (best_combine_nvl, best_combine_nvl_buf)

    dispatch_bw = (
        dispatch_recv_bytes / 1e9 / best_dispatch_time
        if best_dispatch_time > 0 and best_dispatch_time != float("inf")
        else 0
    )
    combine_bw = (
        combine_send_bytes / 1e9 / best_combine_time
        if best_combine_time > 0 and best_combine_time != float("inf")
        else 0
    )

    return (
        dispatch_cfg,
        best_dispatch_time * 1e6 if best_dispatch_time != float("inf") else 0,  # us
        dispatch_bw,
        combine_cfg,
        best_combine_time * 1e6 if best_combine_time != float("inf") else 0,  # us
        combine_bw,
        best_num_sms,
    )


def run_sweep(
    model_config: ModelConfig,
    seq_lens: List[int],
    batch_sizes: List[int],
    group: dist.ProcessGroup,
    warmup: int = 3,
    repeat: int = 5,
) -> List[BenchmarkResult]:
    """Run sweep over seq_len and batch_size combinations."""
    rank = dist.get_rank(group)
    ep_size = group.size()

    # Validate num_experts
    if model_config.num_experts % ep_size != 0:
        if rank == 0:
            logger.error(
                f"num_experts={model_config.num_experts} not divisible by ep_size={ep_size}"
            )
        return []

    # Create buffer with sufficient size for large models
    # For single-node (<=8 GPUs), use NVL only (RDMA=0)
    # For multi-node, use both NVL and RDMA
    import os

    max_tokens = max(seq_lens) * max(batch_sizes)
    hidden_bytes = model_config.hidden_dim * 2  # bfloat16

    # Detect if multi-node based on environment
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", ep_size))
    is_multinode = ep_size > local_world_size

    # Calculate minimum buffer size based on max tokens
    # But cap at 8GB to avoid DeepEP's int overflow issue
    min_nvl_bytes = max_tokens * hidden_bytes * ep_size * 2  # 2x safety margin
    max_nvl_bytes = int(8e9)  # 8GB max due to DeepEP int32 limitation

    # Use at least 2GB for NVL, but cap at 8GB
    num_nvl_bytes = min(max(int(2e9), min_nvl_bytes), max_nvl_bytes)

    # RDMA only needed for multi-node
    if is_multinode:
        min_rdma_bytes = max_tokens * hidden_bytes * 2
        num_rdma_bytes = max(int(1e9), min_rdma_bytes)
    else:
        num_rdma_bytes = 0

    if rank == 0:
        mode_str = "multi-node" if is_multinode else "single-node"
        logger.info(
            f"Buffer ({mode_str}): NVL={num_nvl_bytes / 1e9:.1f}GB, RDMA={num_rdma_bytes / 1e9:.1f}GB"
        )

    buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    is_internode, local_world_size, num_nodes = _detect_internode(buffer)

    if rank == 0:
        mode = "internode" if is_internode else "intranode"
        logger.info(f"Mode: {mode} ({num_nodes} nodes, {ep_size} GPUs)")
        logger.info(f"Model: {model_config.name}")
        logger.info(
            f"  hidden={model_config.hidden_dim}, experts={model_config.num_experts}, topk={model_config.num_topk}"
        )
        logger.info(f"Sweep: seq_lens={seq_lens}, batch_sizes={batch_sizes}")

    # Get SM range to tune over for this GPU
    sms_range = _get_gpu_sm_range()

    if rank == 0:
        logger.info(f"Tuning over SM range: {sms_range}")

    results = []
    total_configs = len(seq_lens) * len(batch_sizes)
    config_idx = 0

    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            config_idx += 1
            num_tokens = seq_len * batch_size

            if rank == 0:
                logger.info(
                    f"[{config_idx}/{total_configs}] seq_len={seq_len}, batch_size={batch_size}, tokens={num_tokens}"
                )

            try:
                (
                    dispatch_cfg,
                    dispatch_time_us,
                    dispatch_bw,
                    combine_cfg,
                    combine_time_us,
                    combine_bw,
                    best_num_sms,
                ) = tune_single_config(
                    buffer=buffer,
                    num_tokens=num_tokens,
                    hidden_dim=model_config.hidden_dim,
                    num_experts=model_config.num_experts,
                    num_topk=model_config.num_topk,
                    sms_range=sms_range,
                    is_internode=is_internode,
                    warmup=warmup,
                    repeat=repeat,
                    rank=rank,
                )

                result = BenchmarkResult(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    num_tokens=num_tokens,
                    dispatch_config=dispatch_cfg,
                    dispatch_time_us=dispatch_time_us,
                    dispatch_bandwidth_gbps=dispatch_bw,
                    combine_config=combine_cfg,
                    combine_time_us=combine_time_us,
                    combine_bandwidth_gbps=combine_bw,
                    best_num_sms=best_num_sms,
                    is_internode=is_internode,
                )
                results.append(result)

                if rank == 0:
                    logger.info(
                        f"  -> dispatch={dispatch_bw:.1f}GB/s, combine={combine_bw:.1f}GB/s, num_sms={best_num_sms}"
                    )

            except Exception as e:
                if rank == 0:
                    logger.error(f"  -> FAILED: {e}")
                continue

            torch.cuda.empty_cache()
            dist.barrier(group)

    return results


def print_results_table(
    results: List[BenchmarkResult], model_config: ModelConfig, rank: int
):
    """Print results in a formatted table with rankings."""
    if rank != 0:
        return

    print("\n" + "=" * 140)
    print(f"DEEPEP BENCHMARK RESULTS - {model_config.name}")
    print("=" * 140)

    if not results:
        print("No successful benchmark results.")
        print("=" * 140)
        return

    r0 = results[0]
    mode = "internode" if r0.is_internode else "intranode"
    print(f"Mode: {mode}")
    print(
        f"Model: hidden={model_config.hidden_dim}, experts={model_config.num_experts}, topk={model_config.num_topk}"
    )
    print("-" * 155)

    # Calculate rankings based on total bandwidth (dispatch + combine)
    sorted_by_bw = sorted(
        results,
        key=lambda x: x.dispatch_bandwidth_gbps + x.combine_bandwidth_gbps,
        reverse=True,
    )
    rank_map = {id(r): i + 1 for i, r in enumerate(sorted_by_bw)}

    # Find max bandwidth for relative comparison
    max_total_bw = (
        sorted_by_bw[0].dispatch_bandwidth_gbps + sorted_by_bw[0].combine_bandwidth_gbps
    )

    # Header
    header = (
        f"{'seq_len':>8} | {'lbs':>4} | {'tokens':>7} | "
        f"{'disp_bw':>10} | {'comb_bw':>10} | {'total_bw':>10} | "
        f"{'disp_us':>9} | {'comb_us':>9} | {'total_us':>10} | "
        f"{'sms':>4} | {'vs_best':>7} | {'rank':>4}"
    )
    print(header)
    print("-" * 155)

    for r in results:
        total_bw = r.dispatch_bandwidth_gbps + r.combine_bandwidth_gbps
        total_us = r.dispatch_time_us + r.combine_time_us
        vs_best = (total_bw / max_total_bw) * 100
        ranking = rank_map[id(r)]

        # Add medal for top 3
        rank_str = f"#{ranking}"
        if ranking == 1:
            rank_str = "#1 *"
        elif ranking == 2:
            rank_str = "#2"
        elif ranking == 3:
            rank_str = "#3"

        row = (
            f"{r.seq_len:>8} | {r.batch_size:>4} | {r.num_tokens:>7} | "
            f"{r.dispatch_bandwidth_gbps:>8.1f}GB/s | {r.combine_bandwidth_gbps:>8.1f}GB/s | {total_bw:>8.1f}GB/s | "
            f"{r.dispatch_time_us:>7.1f}us | {r.combine_time_us:>7.1f}us | {total_us:>8.1f}us | "
            f"{r.best_num_sms:>4} | {vs_best:>6.1f}% | {rank_str:>4}"
        )
        print(row)

    print("=" * 140)

    # Print top 5 configurations
    print("\nTOP 5 CONFIGURATIONS (by total bandwidth):")
    print("-" * 100)
    for i, r in enumerate(sorted_by_bw[:5]):
        total_bw = r.dispatch_bandwidth_gbps + r.combine_bandwidth_gbps
        total_us = r.dispatch_time_us + r.combine_time_us
        vs_best = (total_bw / max_total_bw) * 100
        print(
            f"  #{i+1}: seq_len={r.seq_len:>6}, lbs={r.batch_size:>2}, tokens={r.num_tokens:>6} | "
            f"total_bw={total_bw:>6.1f}GB/s | total_time={total_us:>8.1f}us | vs_best={vs_best:>5.1f}%"
        )

    # Detailed best config
    best = sorted_by_bw[0]
    print("\n" + "-" * 100)
    print("BEST CONFIGURATION DETAILS:")
    print("-" * 100)
    print(
        f"  seq_len={best.seq_len}, lbs={best.batch_size}, tokens={best.num_tokens}, num_sms={best.best_num_sms}"
    )
    if best.is_internode:
        print(
            f"  Dispatch: {best.dispatch_bandwidth_gbps:.1f} GB/s ({best.dispatch_time_us:.1f} us)"
        )
        print(
            f"    config: nvl_chunk={best.dispatch_config[0]}, "
            f"nvl_buf={best.dispatch_config[1]}, "
            f"rdma_chunk={best.dispatch_config[2]}, "
            f"rdma_buf={best.dispatch_config[3]}"
        )
        print(
            f"  Combine:  {best.combine_bandwidth_gbps:.1f} GB/s ({best.combine_time_us:.1f} us)"
        )
        print(
            f"    config: nvl_chunk={best.combine_config[0]}, "
            f"nvl_buf={best.combine_config[1]}, "
            f"rdma_chunk={best.combine_config[2]}, "
            f"rdma_buf={best.combine_config[3]}"
        )
    else:
        print(
            f"  Dispatch: {best.dispatch_bandwidth_gbps:.1f} GB/s ({best.dispatch_time_us:.1f} us)"
        )
        print(
            f"    config: nvl_chunk={best.dispatch_config[0]}, nvl_buf={best.dispatch_config[1]}"
        )
        print(
            f"  Combine:  {best.combine_bandwidth_gbps:.1f} GB/s ({best.combine_time_us:.1f} us)"
        )
        print(
            f"    config: nvl_chunk={best.combine_config[0]}, nvl_buf={best.combine_config[1]}"
        )
    print(
        f"  Total:    {best.dispatch_bandwidth_gbps + best.combine_bandwidth_gbps:.1f} GB/s "
        f"({best.dispatch_time_us + best.combine_time_us:.1f} us)"
    )

    # Recommendations section
    print("\n" + "-" * 100)
    print("RECOMMENDATIONS:")
    print("-" * 100)

    # Find the "efficiency knee" - where we get 95% of max bandwidth with fewer tokens
    threshold_95 = max_total_bw * 0.95
    efficient_configs = [
        r
        for r in sorted_by_bw
        if (r.dispatch_bandwidth_gbps + r.combine_bandwidth_gbps) >= threshold_95
    ]
    if efficient_configs:
        min_tokens_95 = min(r.num_tokens for r in efficient_configs)
        best_efficient = min(
            (r for r in efficient_configs if r.num_tokens == min_tokens_95),
            key=lambda x: x.num_tokens,
        )
        print(f"  - Minimum tokens for 95% efficiency: {min_tokens_95} tokens")
        print(
            f"    (seq_len={best_efficient.seq_len}, lbs={best_efficient.batch_size})"
        )

    threshold_90 = max_total_bw * 0.90
    efficient_configs_90 = [
        r
        for r in sorted_by_bw
        if (r.dispatch_bandwidth_gbps + r.combine_bandwidth_gbps) >= threshold_90
    ]
    if efficient_configs_90:
        min_tokens_90 = min(r.num_tokens for r in efficient_configs_90)
        best_efficient_90 = min(
            (r for r in efficient_configs_90 if r.num_tokens == min_tokens_90),
            key=lambda x: x.num_tokens,
        )
        print(f"  - Minimum tokens for 90% efficiency: {min_tokens_90} tokens")
        print(
            f"    (seq_len={best_efficient_90.seq_len}, lbs={best_efficient_90.batch_size})"
        )

    print("=" * 140)


def save_results_json(
    results: List[BenchmarkResult],
    model_config: ModelConfig,
    filepath: str,
    rank: int,
):
    """Save results to JSON file."""
    if rank != 0 or not results:
        return

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": {
            "name": model_config.name,
            "hidden_dim": model_config.hidden_dim,
            "num_experts": model_config.num_experts,
            "num_topk": model_config.num_topk,
        },
        "setup": {
            "is_internode": results[0].is_internode,
        },
        "results": [],
    }

    for r in results:
        output["results"].append(
            {
                "seq_len": r.seq_len,
                "batch_size": r.batch_size,
                "num_tokens": r.num_tokens,
                "best_num_sms": r.best_num_sms,
                "dispatch_config": list(r.dispatch_config),
                "dispatch_time_us": round(r.dispatch_time_us, 2),
                "dispatch_bandwidth_gbps": round(r.dispatch_bandwidth_gbps, 2),
                "combine_config": list(r.combine_config),
                "combine_time_us": round(r.combine_time_us, 2),
                "combine_bandwidth_gbps": round(r.combine_bandwidth_gbps, 2),
            }
        )

    # Add best config
    best = max(
        results, key=lambda x: x.dispatch_bandwidth_gbps + x.combine_bandwidth_gbps
    )
    output["best"] = {
        "seq_len": best.seq_len,
        "batch_size": best.batch_size,
        "best_num_sms": best.best_num_sms,
        "dispatch_config": list(best.dispatch_config),
        "combine_config": list(best.combine_config),
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="DeepEP Benchmark Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Model presets available:
{chr(10).join(f'  {k}: {v["description"]}' for k, v in MODEL_PRESETS.items())}

Examples:
    # Default sweep with Kimi K2
    torchrun --nproc_per_node=8 -m torchtitan.distributed.deepep.benchmark

    # Use Qwen3 preset
    torchrun --nproc_per_node=8 -m torchtitan.distributed.deepep.benchmark --model qwen3_30b_a3b

    # Quick test with debug model
    torchrun --nproc_per_node=8 -m torchtitan.distributed.deepep.benchmark --model debug --seq_len 2048,4096

    # Custom sweep
    torchrun --nproc_per_node=8 -m torchtitan.distributed.deepep.benchmark \\
        --hidden 7168 --num_experts 256 --topk 8 \\
        --seq_len 4096,8192,16384 --batch_size 1,2,4
        """,
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="kimi_k2",
        choices=list(MODEL_PRESETS.keys()),
        help="Model preset (default: kimi_k2)",
    )

    # Custom model params (override preset)
    parser.add_argument(
        "--hidden", type=int, default=None, help="Hidden dimension (overrides preset)"
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=None,
        help="Number of experts (overrides preset)",
    )
    parser.add_argument(
        "--topk", type=int, default=None, help="Top-k experts (overrides preset)"
    )

    # Sweep parameters
    parser.add_argument(
        "--seq_len",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SEQ_LENS),
        help=f"Sequence lengths (default: {DEFAULT_SEQ_LENS})",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default=",".join(str(x) for x in DEFAULT_BATCH_SIZES),
        help=f"Batch sizes (default: {DEFAULT_BATCH_SIZES})",
    )

    # Benchmark settings
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=5, help="Repeat iterations")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Create EP group
    ep_group = dist.new_group(list(range(world_size)))

    # Build model config
    preset = MODEL_PRESETS[args.model]
    model_config = ModelConfig(
        hidden_dim=args.hidden if args.hidden else preset["hidden_dim"],
        num_experts=args.num_experts if args.num_experts else preset["num_experts"],
        num_topk=args.topk if args.topk else preset["num_topk"],
        name=args.model
        if not any([args.hidden, args.num_experts, args.topk])
        else "custom",
    )

    # Parse sweep params
    seq_lens = _parse_int_list(args.seq_len)
    batch_sizes = _parse_int_list(args.batch_size)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("DeepEP Benchmark")
        logger.info("=" * 60)

    # Run sweep
    results = run_sweep(
        model_config=model_config,
        seq_lens=seq_lens,
        batch_sizes=batch_sizes,
        group=ep_group,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    # Print and save
    print_results_table(results, model_config, rank)

    output_file = (
        args.output or f"deepep_benchmark_{model_config.name}_{world_size}gpu.json"
    )
    save_results_json(results, model_config, output_file, rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
