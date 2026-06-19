# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import itertools

import helion
import torch
import torch.nn.functional as F
import triton.testing as tt

from torchtitan.experiments.helion_kernels.rope import (
    _qk_rmsnorm_rope_cos_sin_fwd_positions,
    apply_qk_rmsnorm_rotary_emb_cos_sin_helion,
    apply_rotary_emb_cos_sin_helion,
)
from torchtitan.models.common.rope import apply_rotary_emb_cos_sin


@dataclass(frozen=True)
class _SweepResult:
    config: helion.Config
    ms: float | None
    error: str | None = None


def _make_inputs(
    *,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    xq = torch.randn(
        batch_size,
        seq_len,
        n_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    xk = torch.randn(
        batch_size,
        seq_len,
        n_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    rope_cache = torch.randn(
        seq_len + 16,
        head_dim * 2,
        device="cuda",
        dtype=torch.float32,
    )
    positions = torch.arange(seq_len, device="cuda", dtype=torch.int32).repeat(
        batch_size, 1
    )
    return xq, xk, q_weight, k_weight, rope_cache, positions


def _separate_core(
    xq: torch.Tensor,
    xk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq = F.rms_norm(xq, (xq.shape[-1],), q_weight, eps)
    xk = F.rms_norm(xk, (xk.shape[-1],), k_weight, eps)
    return apply_rotary_emb_cos_sin(xq, xk, rope_cache, positions)


def _separate_helion_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq = F.rms_norm(xq, (xq.shape[-1],), q_weight, eps)
    xk = F.rms_norm(xk, (xk.shape[-1],), k_weight, eps)
    return apply_rotary_emb_cos_sin_helion(xq, xk, rope_cache, positions)


def _bench_shape(
    *,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    eps: float,
) -> tuple[float, float, float, float]:
    xq, xk, q_weight, k_weight, rope_cache, positions = _make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
    )

    def fused() -> tuple[torch.Tensor, torch.Tensor]:
        return apply_qk_rmsnorm_rotary_emb_cos_sin_helion(
            xq, xk, q_weight, k_weight, rope_cache, positions, eps
        )

    def separate_core() -> tuple[torch.Tensor, torch.Tensor]:
        return _separate_core(xq, xk, q_weight, k_weight, rope_cache, positions, eps)

    def separate_helion_rope() -> tuple[torch.Tensor, torch.Tensor]:
        return _separate_helion_rope(
            xq, xk, q_weight, k_weight, rope_cache, positions, eps
        )

    compiled_separate = torch.compile(_separate_core, fullgraph=True)

    def separate_compile() -> tuple[torch.Tensor, torch.Tensor]:
        return compiled_separate(xq, xk, q_weight, k_weight, rope_cache, positions, eps)

    expected = separate_core()
    actual = fused()
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-2, atol=7e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-2, atol=7e-2)
    separate_compile()
    torch.cuda.synchronize()

    fused_ms = tt.do_bench(
        fused,
        warmup=warmup,
        rep=iters,
        return_mode="median",
    )
    separate_core_ms = tt.do_bench(
        separate_core,
        warmup=warmup,
        rep=iters,
        return_mode="median",
    )
    separate_helion_rope_ms = tt.do_bench(
        separate_helion_rope,
        warmup=warmup,
        rep=iters,
        return_mode="median",
    )
    separate_compile_ms = tt.do_bench(
        separate_compile,
        warmup=warmup,
        rep=iters,
        return_mode="median",
    )
    return fused_ms, separate_core_ms, separate_helion_rope_ms, separate_compile_ms


def _make_sweep_configs(
    q_row_tiles: Iterable[int],
    k_row_tiles: Iterable[int],
    num_warps: Iterable[int],
    *,
    split_qk_tiles: bool,
) -> list[helion.Config]:
    if split_qk_tiles:
        tile_pairs = itertools.product(q_row_tiles, k_row_tiles)
    else:
        tile_pairs = ((tile, tile) for tile in q_row_tiles)
    return [
        helion.Config(block_sizes=[q_tile, k_tile], num_warps=warps)
        for q_tile, k_tile in tile_pairs
        for warps in num_warps
    ]


def _format_config(config: helion.Config) -> str:
    q_tile, k_tile = config.block_sizes
    return f"q={q_tile}, k={k_tile}, warps={config.config.get('num_warps', 'default')}"


def _first_error_line(exc: Exception) -> str:
    return str(exc).splitlines()[0] or type(exc).__name__


def _sweep_shape(
    *,
    configs: list[helion.Config],
    batch_size: int,
    seq_len: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    eps: float,
) -> list[_SweepResult]:
    xq, xk, q_weight, k_weight, rope_cache, positions = _make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
    )
    expected = _separate_core(xq, xk, q_weight, k_weight, rope_cache, positions, eps)
    kernel_args = _qk_rmsnorm_rope_cos_sin_fwd_positions.normalize_args(
        xq, xk, q_weight, k_weight, rope_cache, positions, eps
    )
    bound_kernel = _qk_rmsnorm_rope_cos_sin_fwd_positions.bind(kernel_args)

    results: list[_SweepResult] = []
    for config in configs:
        try:
            bound_kernel.set_config(config)
            actual = bound_kernel(*kernel_args)
            torch.testing.assert_close(actual[0], expected[0], rtol=1e-2, atol=7e-2)
            torch.testing.assert_close(actual[1], expected[1], rtol=1e-2, atol=7e-2)
            torch.cuda.synchronize()
            ms = tt.do_bench(
                lambda: bound_kernel(*kernel_args),
                warmup=warmup,
                rep=iters,
                return_mode="median",
            )
            results.append(_SweepResult(config=config, ms=ms))
        except Exception as exc:
            results.append(
                _SweepResult(config=config, ms=None, error=_first_error_line(exc))
            )
    return results


def _print_sweep_results(
    *,
    seq_len: int,
    results: list[_SweepResult],
    top_k: int,
) -> None:
    valid = sorted(
        (result for result in results if result.ms is not None),
        key=lambda result: result.ms if result.ms is not None else float("inf"),
    )
    failed = [result for result in results if result.ms is None]
    print()
    print(f"S={seq_len}: swept {len(results)} configs, {len(failed)} failed")
    print(f"{'rank':>4s} {'time (us)':>10s}  config")
    print(f"{'----':>4s} {'----------':>10s}  {'-' * 30}")
    for rank, result in enumerate(valid[:top_k], start=1):
        assert result.ms is not None
        print(f"{rank:>4d} {result.ms * 1000:>10.2f}  {_format_config(result.config)}")
    if failed:
        first_failed = failed[0]
        print(
            f"failed examples: {_format_config(first_failed.config)} "
            f"({first_failed.error})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fused Q/K RMSNorm + RoPE")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[1, 192, 1088])
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep Helion configs for the fused kernel after the baseline benchmark.",
    )
    parser.add_argument(
        "--sweep-q-row-tiles",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
        help="Q row tile sizes to sweep.",
    )
    parser.add_argument(
        "--sweep-k-row-tiles",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
        help="K row tile sizes to sweep when --sweep-split-qk-tiles is set.",
    )
    parser.add_argument(
        "--sweep-warps",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="num_warps values to sweep.",
    )
    parser.add_argument(
        "--sweep-split-qk-tiles",
        action="store_true",
        help="Sweep the cross product of Q and K row tiles.",
    )
    parser.add_argument(
        "--sweep-top-k",
        type=int,
        default=5,
        help="Number of fastest configs to print per shape.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"B={args.batch_size}, Hq={args.n_heads}, Hkv={args.n_kv_heads}, "
        f"D={args.head_dim}, dtype=bfloat16"
    )
    print(
        f"{'S':>6s} {'fused (us)':>12s} {'separate (us)':>14s} "
        f"{'sep+helion rope':>16s} {'compiled sep':>13s} {'speedup':>9s}"
    )
    print("-" * 80)

    with torch.inference_mode():
        for seq_len in args.seq_lens:
            (
                fused_ms,
                separate_core_ms,
                separate_helion_rope_ms,
                separate_compile_ms,
            ) = _bench_shape(
                batch_size=args.batch_size,
                seq_len=seq_len,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                head_dim=args.head_dim,
                warmup=args.warmup,
                iters=args.iters,
                eps=args.eps,
            )
            speedup = separate_core_ms / fused_ms
            print(
                f"{seq_len:>6d} {fused_ms * 1000:>12.2f} "
                f"{separate_core_ms * 1000:>14.2f} "
                f"{separate_helion_rope_ms * 1000:>16.2f} "
                f"{separate_compile_ms * 1000:>13.2f} "
                f"{speedup:>8.2f}x"
            )

        if args.sweep:
            configs = _make_sweep_configs(
                args.sweep_q_row_tiles,
                args.sweep_k_row_tiles,
                args.sweep_warps,
                split_qk_tiles=args.sweep_split_qk_tiles,
            )
            print()
            print(
                "Sweep config space: "
                f"{len(configs)} configs "
                f"(q tiles={args.sweep_q_row_tiles}, "
                f"k tiles={args.sweep_k_row_tiles if args.sweep_split_qk_tiles else args.sweep_q_row_tiles}, "
                f"warps={args.sweep_warps})"
            )
            for seq_len in args.seq_lens:
                results = _sweep_shape(
                    configs=configs,
                    batch_size=args.batch_size,
                    seq_len=seq_len,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    head_dim=args.head_dim,
                    warmup=args.warmup,
                    iters=args.iters,
                    eps=args.eps,
                )
                _print_sweep_results(
                    seq_len=seq_len,
                    results=results,
                    top_k=args.sweep_top_k,
                )


if __name__ == "__main__":
    main()
