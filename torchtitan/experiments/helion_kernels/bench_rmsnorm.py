# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F
import triton.testing as tt

from torchtitan.experiments.helion_kernels.rmsnorm import (
    rms_norm_helion,
    rms_norm_helion_raw,
)


BackendFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
BackendFactory = Callable[[int, float], BackendFn | None]


def _parse_shape(shape_spec: str) -> tuple[tuple[int, ...], int]:
    shape_str, sep, count_str = shape_spec.partition(":")
    count = int(count_str) if sep else 1
    return tuple(int(dim) for dim in shape_str.split("x")), count


def _make_aten_fn(normalized_shape: int, eps: float) -> BackendFn:
    def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (normalized_shape,), weight, eps)

    return fn


def _make_compile_fn(normalized_shape: int, eps: float) -> BackendFn:
    return torch.compile(_make_aten_fn(normalized_shape, eps), fullgraph=True)


def _make_helion_fn(normalized_shape: int, eps: float) -> BackendFn:
    def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return rms_norm_helion(x, weight, eps)

    return fn


def _make_helion_raw_fn(normalized_shape: int, eps: float) -> BackendFn:
    def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return rms_norm_helion_raw(x, weight, eps)

    return fn


def _make_vllm_fn(normalized_shape: int, eps: float) -> BackendFn | None:
    try:
        from vllm._custom_ops import rms_norm
    except ImportError:
        return None

    def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        rms_norm(out, x, weight, eps)
        return out

    return fn


def _make_quack_fn(normalized_shape: int, eps: float) -> BackendFn | None:
    try:
        from quack import rmsnorm as quack_rmsnorm
    except ImportError:
        return None

    def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return quack_rmsnorm(x, weight=weight, eps=eps)

    return fn


def _make_sixlib_fn(normalized_shape: int, eps: float) -> BackendFn | None:
    try:
        from ops.interfaces.norm.rms_norm import rmsnorm_op
    except ImportError:
        return None

    def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return rmsnorm_op(x, weight, eps=eps, backend="triton")

    return fn


def _make_llama4x_fn(normalized_shape: int, eps: float) -> BackendFn | None:
    try:
        from llama4x.ops.triton.rms_norm import rmsnorm_fwd
    except ImportError:
        return None

    def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        y, _rstd = rmsnorm_fwd(x, weight, eps)
        return y

    return fn


_BACKENDS: dict[str, BackendFactory] = {
    "aten": _make_aten_fn,
    "compile": _make_compile_fn,
    "helion": _make_helion_fn,
    "helion_raw": _make_helion_raw_fn,
    "vllm": _make_vllm_fn,
    "quack": _make_quack_fn,
    "sixlib": _make_sixlib_fn,
    "llama4x": _make_llama4x_fn,
}


def _benchmark_fn(
    fn: BackendFn,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> float:
    return tt.do_bench(
        lambda: fn(x, weight),
        warmup=warmup,
        rep=iters,
        return_mode="median",
    )


def _available_backends(
    backend_names: list[str],
    dtype: torch.dtype,
    eps: float,
) -> dict[str, BackendFactory]:
    available: dict[str, BackendFactory] = {}
    test_x = torch.randn(2, 128, dtype=dtype, device="cuda")
    test_weight = torch.randn(128, dtype=dtype, device="cuda")
    for name in backend_names:
        factory = _BACKENDS[name]
        fn = factory(128, eps)
        if fn is None:
            print(f"  {name}: not available")
            continue
        try:
            out = fn(test_x, test_weight)
            if out.shape != test_x.shape:
                raise ValueError(f"unexpected output shape {out.shape}")
            available[name] = factory
            print(f"  {name}: available")
        except Exception as exc:
            print(f"  {name}: not available ({exc})")
    return available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RMSNorm implementations")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["aten", "helion", "helion_raw", "compile"],
        choices=sorted(_BACKENDS),
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=[
            "1x1x5120:129",
            "1x8x128:64",
            "1x1x128:64",
            "1x192x5120:129",
            "192x8x128:64",
            "1x192x128:64",
            "1x1088x5120:129",
            "1088x8x128:64",
            "1x1088x128:64",
        ],
        help="Input shapes as MxN or BxSxN, with optional :COUNT suffix.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = getattr(torch, args.dtype)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"dtype: {args.dtype}, eps: {args.eps}")
    print()

    available = _available_backends(args.backends, dtype, args.eps)
    if not available:
        raise RuntimeError("No requested RMSNorm backends are available")
    print()

    header = "  ".join(f"{name:>11s}" for name in available)
    print(f"{'shape':>16s} {'cnt':>5s}  {header}")
    print(f"{'':>16s} {'':>5s}  {'  '.join(['-----------'] * len(available))}")

    totals = {name: 0.0 for name in available}
    rows: list[dict[str, float | int | str | None]] = []
    with torch.inference_mode():
        for shape_spec in args.shapes:
            shape, count = _parse_shape(shape_spec)
            normalized_shape = shape[-1]
            x = torch.randn(*shape, dtype=dtype, device="cuda")
            weight = torch.randn(normalized_shape, dtype=dtype, device="cuda")

            row: dict[str, float | int | str | None] = {
                "shape": "x".join(str(dim) for dim in shape),
                "count": count,
            }
            values = []
            reference = F.rms_norm(x, (normalized_shape,), weight, args.eps)
            for name, factory in available.items():
                fn = factory(normalized_shape, args.eps)
                assert fn is not None
                try:
                    actual = fn(x, weight)
                    torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)
                    ms = _benchmark_fn(
                        fn,
                        x,
                        weight,
                        warmup=args.warmup,
                        iters=args.iters,
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
            print(f"{row['shape']:>16s} {count_str:>5s}  {'  '.join(values)}")

    print()
    print(f"{'weighted total':>16s} {'':>5s}  ", end="")
    total_values = []
    for name in available:
        if all(row.get(name) is not None for row in rows):
            total_values.append(f"{totals[name]:>10.3f}ms")
        else:
            total_values.append(f"{'N/A':>11s}")
    print("  ".join(total_values))

    if "aten" in available and all(row.get("aten") is not None for row in rows):
        aten_total = totals["aten"]
        print(f"{'speedup vs aten':>16s} {'':>5s}  ", end="")
        speedups = []
        for name in available:
            if all(row.get(name) is not None for row in rows):
                speedups.append(f"{aten_total / totals[name]:>10.2f}x")
            else:
                speedups.append(f"{'N/A':>11s}")
        print("  ".join(speedups))


if __name__ == "__main__":
    main()
