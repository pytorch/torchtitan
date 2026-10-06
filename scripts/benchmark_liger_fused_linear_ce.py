# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark TorchTitan compiled CE against Liger fused linear CE.

This measures the complete single-GPU LM-head loss path, including the linear
projection, summed cross-entropy, token normalization, and backward pass. The
TorchTitan case uses ``ChunkedLossWrapper`` with local-compiled
``cross_entropy_loss``. The Liger case uses
``liger_fused_linear_cross_entropy``, backed by
``LigerFusedLinearCrossEntropyFunction``.

Example:

    python scripts/benchmark_liger_fused_linear_ce.py \
        --num-tokens 16384 --hidden-size 4096 --vocab-size 128256 \
        --num-chunks 8 --liger-chunk-mem-const 1 4
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import time

from dataclasses import dataclass

import torch

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.linear import Linear
from torchtitan_recipes.overrides.liger_fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyHead,
    LigerFusedLinearCrossEntropyLoss,
)


_GIB = 1024**3


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    median_ms: float
    mean_ms: float
    min_ms: float
    p10_ms: float
    p90_ms: float
    tokens_per_second: float
    peak_increment_gib: float
    peak_total_gib: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=16384)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=128256)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument(
        "--liger-chunk-mem-const",
        type=int,
        nargs="+",
        default=[1, 4],
        help="Liger transient-logits budget multipliers to benchmark.",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "num_tokens": args.num_tokens,
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
        "num_chunks": args.num_chunks,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Benchmark arguments must be positive: {invalid}")
    if args.steps < 10:
        raise ValueError("Performance measurements require at least 10 steps")
    if args.num_tokens % args.num_chunks != 0:
        raise ValueError("num_tokens must be divisible by num_chunks")
    if any(value <= 0 for value in args.liger_chunk_mem_const):
        raise ValueError("liger_chunk_mem_const values must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")


def _make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.randn(
        args.num_tokens,
        args.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    labels = torch.randint(
        0,
        args.vocab_size,
        (args.num_tokens,),
        device="cuda",
    )
    labels[::97] = -100
    return hidden, labels


def _benchmark_case(
    name: str,
    head: Linear,
    loss_fn: ChunkedLossWrapper,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    warmup_steps: int,
    steps: int,
) -> BenchmarkResult:
    global_valid_tokens = (labels != -100).sum()

    def step() -> None:
        head.zero_grad(set_to_none=True)
        hidden.grad = None
        loss, _ = loss_fn(hidden, labels, global_valid_tokens)
        loss.backward()

    for _ in range(warmup_steps):
        step()
    torch.cuda.synchronize()

    head.zero_grad(set_to_none=True)
    hidden.grad = None
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    durations_ms = []
    for _ in range(steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        durations_ms.append((time.perf_counter() - start) * 1000)

    peak_total = torch.cuda.max_memory_allocated()
    median_ms = statistics.median(durations_ms)
    deciles = statistics.quantiles(durations_ms, n=10, method="inclusive")
    return BenchmarkResult(
        name=name,
        median_ms=median_ms,
        mean_ms=statistics.mean(durations_ms),
        min_ms=min(durations_ms),
        p10_ms=deciles[0],
        p90_ms=deciles[-1],
        tokens_per_second=hidden.shape[0] / (median_ms / 1000),
        peak_increment_gib=(peak_total - baseline_allocated) / _GIB,
        peak_total_gib=peak_total / _GIB,
    )


def _run_compiled_ce(args: argparse.Namespace) -> BenchmarkResult:
    torch.manual_seed(args.seed)
    head = (
        Linear.Config(
            in_features=args.hidden_size,
            out_features=args.vocab_size,
        )
        .build()
        .to(device="cuda", dtype=torch.bfloat16)
    )
    loss_fn = ChunkedLossWrapper.Config(
        num_chunks=args.num_chunks,
        loss_fn=CrossEntropyLoss.Config(global_vocab_size=args.vocab_size),
    ).build()
    loss_fn.set_lm_head(head)
    hidden, labels = _make_inputs(args)
    result = _benchmark_case(
        "compiled_ce",
        head,
        loss_fn,
        hidden,
        labels,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
    )
    del loss_fn, head, hidden, labels
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _run_liger(
    args: argparse.Namespace,
    chunk_mem_const: int,
) -> BenchmarkResult:
    torch.manual_seed(args.seed)
    head = (
        LigerFusedLinearCrossEntropyHead.Config(
            in_features=args.hidden_size,
            out_features=args.vocab_size,
            chunk_mem_const=chunk_mem_const,
        )
        .build()
        .to(device="cuda", dtype=torch.bfloat16)
    )
    loss_fn = LigerFusedLinearCrossEntropyLoss.Config().build()
    loss_fn.set_lm_head(head)
    hidden, labels = _make_inputs(args)
    result = _benchmark_case(
        f"liger_c{chunk_mem_const}",
        head,
        loss_fn,
        hidden,
        labels,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
    )
    del loss_fn, head, hidden, labels
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _liger_chunk_size(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    chunk_mem_const: int,
) -> int:
    increase_factor = math.ceil(vocab_size / (chunk_mem_const * hidden_size))
    requested_size = math.ceil(num_tokens / increase_factor)
    return min(1 << (requested_size - 1).bit_length(), num_tokens)


def _print_results(args: argparse.Namespace, results: list[BenchmarkResult]) -> None:
    print(
        f"shape: T={args.num_tokens}, H={args.hidden_size}, V={args.vocab_size}, "
        f"dtype=bf16, warmup={args.warmup_steps}, measured_steps={args.steps}"
    )
    print(f"compiled_ce chunk_size={args.num_tokens // args.num_chunks}")
    for chunk_mem_const in args.liger_chunk_mem_const:
        print(
            f"liger_c{chunk_mem_const} estimated_chunk_size="
            f"{_liger_chunk_size(args.num_tokens, args.hidden_size, args.vocab_size, chunk_mem_const)}"
        )

    header = (
        f"{'case':<16} {'median_ms':>11} {'p10_ms':>10} {'p90_ms':>10} "
        f"{'loss_tok/s':>14} {'speedup':>9} {'peak_inc_GiB':>14} {'peak_GiB':>12}"
    )
    print(header)
    print("-" * len(header))
    baseline_ms = results[0].median_ms
    for result in results:
        print(
            f"{result.name:<16} {result.median_ms:>11.3f} "
            f"{result.p10_ms:>10.3f} {result.p90_ms:>10.3f} "
            f"{result.tokens_per_second:>14.1f} "
            f"{baseline_ms / result.median_ms:>8.2f}x "
            f"{result.peak_increment_gib:>14.3f} {result.peak_total_gib:>12.3f}"
        )


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    torch.cuda.set_device(0)
    apply_local_compile(["loss"])
    try:
        results = [_run_compiled_ce(args)]
        results.extend(
            _run_liger(args, chunk_mem_const)
            for chunk_mem_const in args.liger_chunk_mem_const
        )
    finally:
        apply_local_compile([])
    _print_results(args, results)


if __name__ == "__main__":
    main()
