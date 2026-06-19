# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import triton.testing as tt

from torchtitan.experiments.helion_kernels.rope import (
    _rope_cos_sin_bwd_positions_with_config,
    apply_rotary_emb_cos_sin_helion,
)


FwdFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]
BwdFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    tuple[torch.Tensor, torch.Tensor],
]


def apply_rotary_emb_cos_sin_reference(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = xq.shape[-1]
    rope_cache = rope_cache[None, :, None, :].expand(xq.shape[0], -1, -1, -1)
    rope_cache = torch.gather(
        rope_cache,
        dim=1,
        index=positions.view(xq.shape[0], xq.shape[1], 1, 1).expand(
            xq.shape[0], xq.shape[1], 1, head_dim * 2
        ),
    )
    cos = rope_cache[..., :head_dim]
    sin = rope_cache[..., head_dim:]

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    xq_out = (xq.float() * cos) + (rotate_half(xq.float()) * sin)
    xk_out = (xk.float() * cos) + (rotate_half(xk.float()) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _parse_shape(shape_spec: str) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    shape_pair, sep, count_str = shape_spec.partition(":")
    count = int(count_str) if sep else 1
    xq_shape_str, pair_sep, xk_shape_str = shape_pair.partition("/")
    if not pair_sep:
        raise ValueError(
            "RoPE shapes must use XQ_SHAPE/XK_SHAPE with optional :COUNT suffix"
        )
    xq_shape = tuple(int(dim) for dim in xq_shape_str.split("x"))
    xk_shape = tuple(int(dim) for dim in xk_shape_str.split("x"))
    if len(xq_shape) != 4 or len(xk_shape) != 4:
        raise ValueError(f"RoPE shapes must be 4D, got {shape_spec}")
    if xq_shape[0] != xk_shape[0] or xq_shape[1] != xk_shape[1]:
        raise ValueError(f"Q/K batch and sequence must match, got {shape_spec}")
    if xq_shape[-1] != xk_shape[-1]:
        raise ValueError(f"Q/K head_dim must match, got {shape_spec}")
    return xq_shape, xk_shape, count


def _make_inputs(
    xq_shape: tuple[int, ...],
    xk_shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xq = torch.randn(*xq_shape, device="cuda", dtype=dtype).requires_grad_(
        requires_grad
    )
    xk = torch.randn(*xk_shape, device="cuda", dtype=dtype).requires_grad_(
        requires_grad
    )
    seq_len = xq_shape[1]
    head_dim = xq_shape[-1]
    rope_cache = torch.randn(
        seq_len + 16,
        head_dim * 2,
        device="cuda",
        dtype=torch.float32,
    )
    positions = torch.arange(seq_len, device="cuda", dtype=torch.int32).repeat(
        xq_shape[0], 1
    )
    return xq, xk, rope_cache, positions


def _aten_bwd_fn(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
    grad_xq_out: torch.Tensor,
    grad_xk_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_out, xk_out = apply_rotary_emb_cos_sin_reference(
        xq, xk, rope_cache, positions
    )
    grad_xq, grad_xk = torch.autograd.grad(
        (xq_out, xk_out),
        (xq, xk),
        (grad_xq_out, grad_xk_out),
    )
    return grad_xq, grad_xk


def _make_fwd_fn(name: str) -> FwdFn:
    if name == "aten":
        return apply_rotary_emb_cos_sin_reference
    if name == "compile":
        return torch.compile(apply_rotary_emb_cos_sin_reference, fullgraph=True)
    if name == "helion":
        return apply_rotary_emb_cos_sin_helion
    raise ValueError(f"unknown backend {name}")


def _make_bwd_fn(name: str) -> BwdFn:
    if name == "aten":
        return _aten_bwd_fn
    if name == "compile":
        torch._dynamo.config.trace_autograd_ops = True
        return torch.compile(_aten_bwd_fn, fullgraph=True)
    if name == "helion":

        def fn(
            xq: torch.Tensor,
            xk: torch.Tensor,
            rope_cache: torch.Tensor,
            positions: torch.Tensor,
            grad_xq_out: torch.Tensor,
            grad_xk_out: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return _rope_cos_sin_bwd_positions_with_config(
                grad_xq_out,
                grad_xk_out,
                rope_cache,
                positions,
            )

        return fn
    raise ValueError(f"unknown backend {name}")


def _benchmark_fwd(
    fn: FwdFn,
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> float:
    return tt.do_bench(
        lambda: fn(xq, xk, rope_cache, positions),
        warmup=warmup,
        rep=iters,
        return_mode="median",
    )


def _benchmark_bwd(
    fn: BwdFn,
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
    grad_xq_out: torch.Tensor,
    grad_xk_out: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> float:
    return tt.do_bench(
        lambda: fn(xq, xk, rope_cache, positions, grad_xq_out, grad_xk_out),
        warmup=warmup,
        rep=iters,
        return_mode="median",
    )


def _assert_close_pair(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
    *,
    rtol: float,
    atol: float,
) -> None:
    torch.testing.assert_close(actual[0], expected[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(actual[1], expected[1], rtol=rtol, atol=atol)


def _print_totals(
    rows: list[dict[str, float | int | str | None]],
    totals: dict[str, float],
    backend_names: list[str],
) -> None:
    print()
    print(f"{'weighted total':>33s} {'':>5s}  ", end="")
    total_values = []
    for name in backend_names:
        if all(row.get(name) is not None for row in rows):
            total_values.append(f"{totals[name]:>10.3f}ms")
        else:
            total_values.append(f"{'N/A':>11s}")
    print("  ".join(total_values))

    if "aten" in backend_names and all(row.get("aten") is not None for row in rows):
        aten_total = totals["aten"]
        print(f"{'speedup vs aten':>33s} {'':>5s}  ", end="")
        speedups = []
        for name in backend_names:
            if all(row.get(name) is not None for row in rows):
                speedups.append(f"{aten_total / totals[name]:>10.2f}x")
            else:
                speedups.append(f"{'N/A':>11s}")
        print("  ".join(speedups))


def benchmark_forward(
    shape_specs: list[str],
    backend_names: list[str],
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> None:
    print()
    print("=== RoPE forward ===")
    header = "  ".join(f"{name:>11s}" for name in backend_names)
    print(f"{'xq / xk shape':>33s} {'cnt':>5s}  {header}")
    print(f"{'':>33s} {'':>5s}  {'  '.join(['-----------'] * len(backend_names))}")

    fns = {name: _make_fwd_fn(name) for name in backend_names}
    totals = {name: 0.0 for name in backend_names}
    rows: list[dict[str, float | int | str | None]] = []
    with torch.inference_mode():
        for shape_spec in shape_specs:
            xq_shape, xk_shape, count = _parse_shape(shape_spec)
            xq, xk, rope_cache, positions = _make_inputs(
                xq_shape, xk_shape, dtype, requires_grad=False
            )
            reference = apply_rotary_emb_cos_sin_reference(
                xq, xk, rope_cache, positions
            )
            row: dict[str, float | int | str | None] = {
                "shape": f"{'x'.join(map(str, xq_shape))}/"
                f"{'x'.join(map(str, xk_shape))}",
                "count": count,
            }
            values = []
            for name, fn in fns.items():
                try:
                    actual = fn(xq, xk, rope_cache, positions)
                    _assert_close_pair(actual, reference, rtol=1e-2, atol=1e-2)
                    ms = _benchmark_fwd(
                        fn,
                        xq,
                        xk,
                        rope_cache,
                        positions,
                        warmup=warmup,
                        iters=iters,
                    )
                    row[name] = ms
                    totals[name] += ms * count
                    values.append(f"{ms * 1000:>10.2f}us")
                except Exception as exc:
                    row[name] = None
                    values.append(f"{'ERR':>11s}")
                    print(f"    {name} error for {shape_spec}: {exc}")
            rows.append(row)
            count_str = f"x{count}" if count > 1 else ""
            print(f"{row['shape']:>33s} {count_str:>5s}  {'  '.join(values)}")

    _print_totals(rows, totals, backend_names)


def benchmark_backward(
    shape_specs: list[str],
    backend_names: list[str],
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> None:
    print()
    print("=== RoPE backward ===")
    header = "  ".join(f"{name:>11s}" for name in backend_names)
    print(f"{'xq / xk shape':>33s} {'cnt':>5s}  {header}")
    print(f"{'':>33s} {'':>5s}  {'  '.join(['-----------'] * len(backend_names))}")

    fns = {name: _make_bwd_fn(name) for name in backend_names}
    totals = {name: 0.0 for name in backend_names}
    rows: list[dict[str, float | int | str | None]] = []
    for shape_spec in shape_specs:
        xq_shape, xk_shape, count = _parse_shape(shape_spec)
        xq, xk, rope_cache, positions = _make_inputs(
            xq_shape, xk_shape, dtype, requires_grad=True
        )
        grad_xq_out = torch.randn_like(xq)
        grad_xk_out = torch.randn_like(xk)
        reference = _aten_bwd_fn(
            xq, xk, rope_cache, positions, grad_xq_out, grad_xk_out
        )
        row = {
            "shape": f"{'x'.join(map(str, xq_shape))}/"
            f"{'x'.join(map(str, xk_shape))}",
            "count": count,
        }
        values = []
        for name, fn in fns.items():
            try:
                actual = fn(xq, xk, rope_cache, positions, grad_xq_out, grad_xk_out)
                _assert_close_pair(actual, reference, rtol=3e-2, atol=3e-2)
                ms = _benchmark_bwd(
                    fn,
                    xq,
                    xk,
                    rope_cache,
                    positions,
                    grad_xq_out,
                    grad_xk_out,
                    warmup=warmup,
                    iters=iters,
                )
                row[name] = ms
                totals[name] += ms * count
                values.append(f"{ms * 1000:>10.2f}us")
            except Exception as exc:
                row[name] = None
                values.append(f"{'ERR':>11s}")
                print(f"    {name} error for {shape_spec}: {exc}")
        rows.append(row)
        count_str = f"x{count}" if count > 1 else ""
        print(f"{row['shape']:>33s} {count_str:>5s}  {'  '.join(values)}")

    _print_totals(rows, totals, backend_names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RoPE implementations")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["aten", "compile", "helion"],
        choices=["aten", "compile", "helion"],
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=[
            "1x1x8x128/1x1x1x128:64",
            "1x192x8x128/1x192x1x128:64",
            "1x1088x8x128/1x1088x1x128:64",
        ],
        help="Shapes as XQ_SHAPE/XK_SHAPE with optional :COUNT suffix.",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["fwd", "bwd"],
        choices=["fwd", "bwd"],
        help="Benchmark forward, backward, or both.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch._dynamo.config.trace_autograd_ops = True
    dtype = getattr(torch, args.dtype)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"dtype: {args.dtype}")

    if "fwd" in args.directions:
        benchmark_forward(args.shapes, args.backends, dtype, args.warmup, args.iters)
    if "bwd" in args.directions:
        print()
        print(
            "Backward timings use autograd for aten/compile and the direct "
            "Helion backward kernel for Helion."
        )
        benchmark_backward(args.shapes, args.backends, dtype, args.warmup, args.iters)


if __name__ == "__main__":
    main()
