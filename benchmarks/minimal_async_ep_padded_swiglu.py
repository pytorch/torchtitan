# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone MinimalAsyncEP padded SwiGLU experiment.

MinimalAsyncEP sizes the dispatched expert input for worst-case receive capacity.
GroupedExperts uses ``torch._grouped_mm(..., offs=...)`` for the matmuls, but
the intermediate SiLU and multiply operate on the full padded ``(R, F)`` tensor.

This script prototypes a fused SiLU+mul kernel that skips rows after the active
row count. Tail rows in the prototype output are intentionally unspecified; the
intended consumer must only read rows before the same active row count. It is
standalone so it can be benchmarked without touching the live MoE or
MinimalAsyncEP code paths.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@triton.jit
def _active_silu_mul_forward_kernel(
    gate,
    up,
    out,
    active_rows_ptr,
    num_cols: tl.constexpr,
    gate_stride_m: tl.constexpr,
    gate_stride_n: tl.constexpr,
    up_stride_m: tl.constexpr,
    up_stride_n: tl.constexpr,
    out_stride_m: tl.constexpr,
    out_stride_n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
) -> None:
    row_start = tl.program_id(0) * block_m
    active_rows = tl.load(active_rows_ptr)
    if row_start >= active_rows:
        return

    rows = row_start + tl.arange(0, block_m)
    cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
    mask = (rows[:, None] < active_rows) & (cols[None, :] < num_cols)

    gate_values = tl.load(
        gate + rows[:, None] * gate_stride_m + cols[None, :] * gate_stride_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up_values = tl.load(
        up + rows[:, None] * up_stride_m + cols[None, :] * up_stride_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    silu = gate_values * tl.sigmoid(gate_values)
    tl.store(
        out + rows[:, None] * out_stride_m + cols[None, :] * out_stride_n,
        silu * up_values,
        mask=mask,
    )


@triton.jit
def _active_silu_mul_backward_kernel(
    grad_out,
    gate,
    up,
    grad_gate,
    grad_up,
    active_rows_ptr,
    num_cols: tl.constexpr,
    grad_out_stride_m: tl.constexpr,
    grad_out_stride_n: tl.constexpr,
    gate_stride_m: tl.constexpr,
    gate_stride_n: tl.constexpr,
    up_stride_m: tl.constexpr,
    up_stride_n: tl.constexpr,
    grad_gate_stride_m: tl.constexpr,
    grad_gate_stride_n: tl.constexpr,
    grad_up_stride_m: tl.constexpr,
    grad_up_stride_n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
) -> None:
    row_start = tl.program_id(0) * block_m
    active_rows = tl.load(active_rows_ptr)
    if row_start >= active_rows:
        return

    rows = row_start + tl.arange(0, block_m)
    cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
    mask = (rows[:, None] < active_rows) & (cols[None, :] < num_cols)

    grad_values = tl.load(
        grad_out
        + rows[:, None] * grad_out_stride_m
        + cols[None, :] * grad_out_stride_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate_values = tl.load(
        gate + rows[:, None] * gate_stride_m + cols[None, :] * gate_stride_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up_values = tl.load(
        up + rows[:, None] * up_stride_m + cols[None, :] * up_stride_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    sigmoid = tl.sigmoid(gate_values)
    silu = gate_values * sigmoid
    silu_grad = sigmoid * (1.0 + gate_values * (1.0 - sigmoid))

    tl.store(
        grad_gate
        + rows[:, None] * grad_gate_stride_m
        + cols[None, :] * grad_gate_stride_n,
        grad_values * up_values * silu_grad,
        mask=mask,
    )
    tl.store(
        grad_up + rows[:, None] * grad_up_stride_m + cols[None, :] * grad_up_stride_n,
        grad_values * silu,
        mask=mask,
    )


def _grid_rows(
    capacity_rows: int,
    active_rows: torch.Tensor,
    block_m: int,
    mode: str,
) -> int:
    if mode == "active":
        return triton.cdiv(int(active_rows.item()), block_m)
    if mode == "full":
        return triton.cdiv(capacity_rows, block_m)
    raise ValueError(f"unknown grid mode: {mode}")


def active_silu_mul_forward(
    gate: torch.Tensor,
    up: torch.Tensor,
    active_rows: torch.Tensor,
    out: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    grid_mode: str,
) -> torch.Tensor:
    if gate.shape != up.shape or gate.shape != out.shape:
        raise ValueError("gate, up, and out must have the same shape")
    if gate.ndim != 2:
        raise ValueError("expected 2D tensors")

    capacity_rows, num_cols = gate.shape
    grid = (
        _grid_rows(capacity_rows, active_rows, block_m, grid_mode),
        triton.cdiv(num_cols, block_n),
    )
    _active_silu_mul_forward_kernel[grid](
        gate,
        up,
        out,
        active_rows,
        num_cols=num_cols,
        gate_stride_m=gate.stride(0),
        gate_stride_n=gate.stride(1),
        up_stride_m=up.stride(0),
        up_stride_n=up.stride(1),
        out_stride_m=out.stride(0),
        out_stride_n=out.stride(1),
        block_m=block_m,
        block_n=block_n,
        num_warps=8,
    )
    return out


class ActiveSiluMul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        active_rows: torch.Tensor,
        block_m: int,
        block_n: int,
        grid_mode: str,
    ) -> torch.Tensor:
        out = torch.empty_like(gate)
        active_silu_mul_forward(
            gate,
            up,
            active_rows,
            out,
            block_m=block_m,
            block_n=block_n,
            grid_mode=grid_mode,
        )
        ctx.save_for_backward(gate, up, active_rows)
        ctx.block_m = block_m
        ctx.block_n = block_n
        ctx.grid_mode = grid_mode
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate, up, active_rows = ctx.saved_tensors
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        capacity_rows, num_cols = gate.shape
        grid = (
            _grid_rows(capacity_rows, active_rows, ctx.block_m, ctx.grid_mode),
            triton.cdiv(num_cols, ctx.block_n),
        )
        _active_silu_mul_backward_kernel[grid](
            grad_out,
            gate,
            up,
            grad_gate,
            grad_up,
            active_rows,
            num_cols=num_cols,
            grad_out_stride_m=grad_out.stride(0),
            grad_out_stride_n=grad_out.stride(1),
            gate_stride_m=gate.stride(0),
            gate_stride_n=gate.stride(1),
            up_stride_m=up.stride(0),
            up_stride_n=up.stride(1),
            grad_gate_stride_m=grad_gate.stride(0),
            grad_gate_stride_n=grad_gate.stride(1),
            grad_up_stride_m=grad_up.stride(0),
            grad_up_stride_n=grad_up.stride(1),
            block_m=ctx.block_m,
            block_n=ctx.block_n,
            num_warps=8,
        )
        return grad_gate, grad_up, None, None, None, None


@dataclass(frozen=True)
class Measurement:
    name: str
    ms: float


def benchmark_cuda(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def max_active_diff(actual: torch.Tensor, expected: torch.Tensor, active_rows: int) -> float:
    return (
        actual[:active_rows].float() - expected[:active_rows].float()
    ).abs().max().item()


def run_forward(args: argparse.Namespace) -> list[Measurement]:
    dtype = _DTYPES[args.dtype]
    device = torch.device("cuda")
    gate = torch.randn(
        args.capacity_rows,
        args.hidden_dim,
        device=device,
        dtype=dtype,
    )
    up = torch.randn_like(gate)
    active_rows = torch.tensor([args.active_rows], device=device, dtype=torch.int64)
    torch_out = torch.empty_like(gate)
    triton_out = torch.empty_like(gate)

    torch.mul(F.silu(gate), up, out=torch_out)
    active_silu_mul_forward(
        gate,
        up,
        active_rows,
        triton_out,
        block_m=args.block_m,
        block_n=args.block_n,
        grid_mode="full",
    )
    torch.cuda.synchronize()
    diff = max_active_diff(triton_out, torch_out, args.active_rows)
    print(f"forward max active diff: {diff:.6g}")

    results = [
        Measurement(
            "torch full capacity silu + mul",
            benchmark_cuda(
                lambda: torch.mul(F.silu(gate), up, out=torch_out),
                warmup=args.warmup,
                iters=args.iters,
            ),
        ),
        Measurement(
            "triton fused full capacity",
            benchmark_cuda(
                lambda: active_silu_mul_forward(
                    gate,
                    up,
                    active_rows,
                    triton_out,
                    block_m=args.block_m,
                    block_n=args.block_n,
                    grid_mode="full",
                ),
                warmup=args.warmup,
                iters=args.iters,
            ),
        ),
        Measurement(
            "triton fused active grid ideal",
            benchmark_cuda(
                lambda: active_silu_mul_forward(
                    gate,
                    up,
                    active_rows,
                    triton_out,
                    block_m=args.block_m,
                    block_n=args.block_n,
                    grid_mode="active",
                ),
                warmup=args.warmup,
                iters=args.iters,
            ),
        ),
    ]
    if args.include_compiled:
        compiled_full = torch.compile(lambda lhs, rhs: F.silu(lhs) * rhs)
        compiled_out = compiled_full(gate, up)
        torch.cuda.synchronize()
        diff = max_active_diff(compiled_out, torch_out, args.active_rows)
        print(f"compiled max active diff: {diff:.6g}")
        results.append(
            Measurement(
                "torch.compile full capacity",
                benchmark_cuda(
                    lambda: compiled_full(gate, up),
                    warmup=args.warmup,
                    iters=args.iters,
                ),
            )
        )
    return results


def run_backward(args: argparse.Namespace) -> list[Measurement]:
    dtype = _DTYPES[args.dtype]
    device = torch.device("cuda")
    active_rows = torch.tensor([args.active_rows], device=device, dtype=torch.int64)
    gate = torch.randn(
        args.capacity_rows,
        args.hidden_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    up = torch.randn_like(gate, requires_grad=True)
    grad = torch.randn_like(gate)
    grad[args.active_rows :].zero_()

    def torch_backward() -> None:
        gate.grad = None
        up.grad = None
        out = F.silu(gate) * up
        out.backward(grad)

    def active_backward(grid_mode: str) -> None:
        gate.grad = None
        up.grad = None
        out = ActiveSiluMul.apply(
            gate,
            up,
            active_rows,
            args.block_m,
            args.block_n,
            grid_mode,
        )
        out.backward(grad)

    torch_backward()
    torch_gate_grad = gate.grad.detach().clone()
    torch_up_grad = up.grad.detach().clone()
    active_backward("full")
    torch.cuda.synchronize()
    gate_diff = max_active_diff(gate.grad, torch_gate_grad, args.active_rows)
    up_diff = max_active_diff(up.grad, torch_up_grad, args.active_rows)
    print(f"backward max active grad diff: gate={gate_diff:.6g}, up={up_diff:.6g}")

    return [
        Measurement(
            "torch full capacity backward",
            benchmark_cuda(torch_backward, warmup=args.warmup, iters=args.iters),
        ),
        Measurement(
            "triton fused backward full grid",
            benchmark_cuda(
                lambda: active_backward("full"),
                warmup=args.warmup,
                iters=args.iters,
            ),
        ),
        Measurement(
            "triton fused backward active ideal",
            benchmark_cuda(
                lambda: active_backward("active"),
                warmup=args.warmup,
                iters=args.iters,
            ),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capacity-rows",
        type=int,
        default=786_432,
        help="Worst-case MinimalAsyncEP receive rows.",
    )
    parser.add_argument(
        "--active-rows",
        type=int,
        default=98_304,
        help="Real rows, normally num_tokens_per_expert.sum().",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=1408,
        help="MoE intermediate hidden dim F.",
    )
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    parser.add_argument("--block-m", type=int, default=4)
    parser.add_argument("--block-n", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--include-compiled",
        action="store_true",
        help="Also benchmark torch.compile(F.silu(gate) * up) in forward mode.",
    )
    args = parser.parse_args()

    if args.capacity_rows <= 0:
        raise ValueError("capacity rows must be positive")
    if args.hidden_dim <= 0:
        raise ValueError("hidden dim must be positive")
    if not (0 < args.active_rows <= args.capacity_rows):
        raise ValueError("active rows must be in (0, capacity rows]")
    if args.block_m <= 0 or args.block_n <= 0:
        raise ValueError("block sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    return args


def main() -> None:
    args = parse_args()
    padding_factor = args.capacity_rows / args.active_rows
    print(
        "shape: "
        f"capacity_rows={args.capacity_rows}, active_rows={args.active_rows}, "
        f"hidden_dim={args.hidden_dim}, dtype={args.dtype}, "
        f"padding_factor={padding_factor:.2f}x"
    )
    print(
        f"kernel: block_m={args.block_m}, block_n={args.block_n}, "
        f"device={torch.cuda.get_device_name()}"
    )

    results = run_backward(args) if args.backward else run_forward(args)
    baseline = results[0].ms
    for result in results:
        speedup = baseline / result.ms
        print(f"{result.name:35s} {result.ms:8.3f} ms  {speedup:6.2f}x")


if __name__ == "__main__":
    main()
