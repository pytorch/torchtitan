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
    _rms_norm_helion_bwd_2d_with_config,
    rms_norm_helion,
    rms_norm_helion_raw,
)


BackendFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
BackendFactory = Callable[[int, float], BackendFn | None]
BwdFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]


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


def _aten_bwd_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_out: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = F.rms_norm(x, (x.shape[-1],), weight, eps)
    grad_x, grad_weight = torch.autograd.grad(out, (x, weight), grad_out)
    return grad_x, grad_weight


def _make_bwd_fn(name: str, normalized_shape: int, eps: float) -> BwdFn | None:
    if name == "compile":
        torch._dynamo.config.trace_autograd_ops = True

        def fn(
            x: torch.Tensor,
            weight: torch.Tensor,
            grad_out: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return _aten_bwd_fn(x, weight, grad_out, eps)

        return torch.compile(fn, fullgraph=True)

    if name == "helion_raw":

        def fn(
            x: torch.Tensor,
            weight: torch.Tensor,
            grad_out: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            grad_x_2d, grad_weight = _rms_norm_helion_bwd_2d_with_config(
                grad_out.reshape(-1, normalized_shape),
                x.reshape(-1, normalized_shape),
                weight,
                eps,
            )
            return grad_x_2d.reshape_as(x), grad_weight

        return fn

    factory = _BACKENDS[name]
    fwd_fn = factory(normalized_shape, eps)
    if fwd_fn is None:
        return None

    def fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = fwd_fn(x, weight)
        grad_x, grad_weight = torch.autograd.grad(out, (x, weight), grad_out)
        return grad_x, grad_weight

    return fn


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


def _benchmark_bwd_fn(
    fn: BwdFn,
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> float:
    return tt.do_bench(
        lambda: fn(x, weight, grad_out),
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
    test_x = torch.randn(1, 128, dtype=dtype, device="cuda")
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


def _available_bwd_backends(
    backend_names: list[str],
    dtype: torch.dtype,
    eps: float,
) -> dict[str, Callable[[int, float], BwdFn | None]]:
    available: dict[str, Callable[[int, float], BwdFn | None]] = {}
    test_x = torch.randn(1, 128, dtype=dtype, device="cuda", requires_grad=True)
    test_weight = torch.randn(128, dtype=dtype, device="cuda", requires_grad=True)
    test_grad_out = torch.randn_like(test_x)
    for name in backend_names:
        try:
            fn = _make_bwd_fn(name, 128, eps)
            if fn is None:
                print(f"  {name}: not available")
                continue
            out = fn(test_x, test_weight, test_grad_out)
            if out[0].shape != test_x.shape or out[1].shape != test_weight.shape:
                raise ValueError(
                    f"unexpected grad shapes {out[0].shape}, {out[1].shape}"
                )
            available[name] = lambda normalized_shape, eps, name=name: _make_bwd_fn(
                name, normalized_shape, eps
            )
            print(f"  {name}: available")
        except Exception as exc:
            print(f"  {name}: not available ({exc})")
    return available


def _print_table_header(title: str, backend_names: list[str]) -> None:
    print()
    print(f"=== {title} ===")
    header = "  ".join(f"{name:>11s}" for name in backend_names)
    print(f"{'shape':>16s} {'cnt':>5s}  {header}")
    print(f"{'':>16s} {'':>5s}  {'  '.join(['-----------'] * len(backend_names))}")


def _print_totals(
    rows: list[dict[str, float | int | str | None]],
    totals: dict[str, float],
    backend_names: list[str],
) -> None:
    print()
    print(f"{'weighted total':>16s} {'':>5s}  ", end="")
    total_values = []
    for name in backend_names:
        if all(row.get(name) is not None for row in rows):
            total_values.append(f"{totals[name]:>10.3f}ms")
        else:
            total_values.append(f"{'N/A':>11s}")
    print("  ".join(total_values))

    if "aten" in backend_names and all(row.get("aten") is not None for row in rows):
        aten_total = totals["aten"]
        print(f"{'speedup vs aten':>16s} {'':>5s}  ", end="")
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
    eps: float,
    warmup: int,
    iters: int,
) -> None:
    print("Forward backend availability:")
    available = _available_backends(backend_names, dtype, eps)
    if not available:
        raise RuntimeError("No requested RMSNorm forward backends are available")

    names = list(available)
    _print_table_header("RMSNorm forward", names)
    totals = {name: 0.0 for name in names}
    rows: list[dict[str, float | int | str | None]] = []
    with torch.inference_mode():
        for shape_spec in shape_specs:
            shape, count = _parse_shape(shape_spec)
            normalized_shape = shape[-1]
            x = torch.randn(*shape, dtype=dtype, device="cuda")
            weight = torch.randn(normalized_shape, dtype=dtype, device="cuda")
            reference = F.rms_norm(x, (normalized_shape,), weight, eps)

            row: dict[str, float | int | str | None] = {
                "shape": "x".join(str(dim) for dim in shape),
                "count": count,
            }
            values = []
            for name, factory in available.items():
                fn = factory(normalized_shape, eps)
                assert fn is not None
                try:
                    actual = fn(x, weight)
                    torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)
                    ms = _benchmark_fn(
                        fn,
                        x,
                        weight,
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
            print(f"{row['shape']:>16s} {count_str:>5s}  {'  '.join(values)}")

    _print_totals(rows, totals, names)


def benchmark_backward(
    shape_specs: list[str],
    backend_names: list[str],
    dtype: torch.dtype,
    eps: float,
    warmup: int,
    iters: int,
) -> None:
    print()
    print("Backward backend availability:")
    available = _available_bwd_backends(backend_names, dtype, eps)
    if not available:
        raise RuntimeError("No requested RMSNorm backward backends are available")

    names = list(available)
    _print_table_header("RMSNorm backward", names)
    totals = {name: 0.0 for name in names}
    rows: list[dict[str, float | int | str | None]] = []
    for shape_spec in shape_specs:
        shape, count = _parse_shape(shape_spec)
        normalized_shape = shape[-1]
        x = torch.randn(*shape, dtype=dtype, device="cuda", requires_grad=True)
        weight = torch.randn(
            normalized_shape, dtype=dtype, device="cuda", requires_grad=True
        )
        grad_out = torch.randn_like(x)
        reference = _aten_bwd_fn(x, weight, grad_out, eps)

        row = {
            "shape": "x".join(str(dim) for dim in shape),
            "count": count,
        }
        values = []
        for name, factory in available.items():
            fn = factory(normalized_shape, eps)
            assert fn is not None
            try:
                actual = fn(x, weight, grad_out)
                torch.testing.assert_close(
                    actual[0], reference[0], rtol=2e-2, atol=2e-2
                )
                torch.testing.assert_close(
                    actual[1], reference[1], rtol=2e-2, atol=2e-2
                )
                ms = _benchmark_bwd_fn(
                    fn,
                    x,
                    weight,
                    grad_out,
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
        print(f"{row['shape']:>16s} {count_str:>5s}  {'  '.join(values)}")

    _print_totals(rows, totals, names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RMSNorm implementations")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["aten", "compile", "helion_raw"],
        choices=sorted(_BACKENDS),
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["fwd", "bwd"],
        choices=["fwd", "bwd"],
        help="Benchmark forward, backward, or both.",
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
    torch._dynamo.config.trace_autograd_ops = True
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"dtype: {args.dtype}, eps: {args.eps}")
    print()

    if "fwd" in args.directions:
        benchmark_forward(
            args.shapes,
            args.backends,
            dtype,
            args.eps,
            args.warmup,
            args.iters,
        )
    if "bwd" in args.directions:
        print()
        print(
            "Backward timings use autograd for aten/compile and direct Helion "
            "backward kernels for Helion backends."
        )
        benchmark_backward(
            args.shapes,
            args.backends,
            dtype,
            args.eps,
            args.warmup,
            args.iters,
        )


if __name__ == "__main__":
    main()
