# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark Kimi K3 elementwise and reduction fusion candidates.

This benchmark compares eager PyTorch, the current TorchInductor-compiled
implementations, and available handwritten kernels. It reports training
forward and backward latency independently and validates outputs and gradients
against eager PyTorch before timing.

Example:

    CUDA_VISIBLE_DEVICES=2 python benchmarks/kimi_k3_fusion_microbench.py \
        --preset k3 --warmup-ms 100 --rep-ms 500
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton
import triton.language as tl
from fla.modules.fused_norm_gate import rms_norm_gated
from fla.ops.attnres import fused_attnres

from torchtitan.models.kimi_k3.kda import _compiled_rms_norm_gated, _rms_norm_gated
from torchtitan.models.kimi_k3.model import (
    _attention_residual,
    _compiled_attention_residual,
)
from torchtitan.models.kimi_k3.moe import _compiled_situ_glu, _situ_glu
from triton.language.extra import libdevice


MiB = 1024**2


@triton.jit
def _situ_forward_kernel(
    gate,
    up,
    output,
    numel: tl.constexpr,
    beta: tl.constexpr,
    linear_beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(
        tl.int64
    )
    mask = offsets < numel
    gate_value = tl.load(gate + offsets, mask=mask).to(tl.float32)
    up_value = tl.load(up + offsets, mask=mask).to(tl.float32)

    gate_tanh = libdevice.tanh(gate_value / beta)
    up_tanh = libdevice.tanh(up_value / linear_beta)
    output_value = beta * gate_tanh * tl.sigmoid(gate_value) * linear_beta * up_tanh
    tl.store(output + offsets, output_value, mask=mask)


@triton.jit
def _situ_backward_kernel(
    gate,
    up,
    grad_output,
    grad_gate,
    grad_up,
    numel: tl.constexpr,
    beta: tl.constexpr,
    linear_beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(
        tl.int64
    )
    mask = offsets < numel
    gate_value = tl.load(gate + offsets, mask=mask).to(tl.float32)
    up_value = tl.load(up + offsets, mask=mask).to(tl.float32)
    grad_value = tl.load(grad_output + offsets, mask=mask).to(tl.float32)

    gate_tanh = libdevice.tanh(gate_value / beta)
    up_tanh = libdevice.tanh(up_value / linear_beta)
    gate_sigmoid = tl.sigmoid(gate_value)
    capped_gate = beta * gate_tanh
    capped_up = linear_beta * up_tanh

    gate_derivative = (1.0 - gate_tanh * gate_tanh) * gate_sigmoid
    gate_derivative += capped_gate * gate_sigmoid * (1.0 - gate_sigmoid)
    up_derivative = 1.0 - up_tanh * up_tanh

    tl.store(
        grad_gate + offsets,
        grad_value * capped_up * gate_derivative,
        mask=mask,
    )
    tl.store(
        grad_up + offsets,
        grad_value * capped_gate * gate_sigmoid * up_derivative,
        mask=mask,
    )


class _TritonSiTUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        beta: float,
        linear_beta: float,
    ) -> torch.Tensor:
        if not gate.is_contiguous() or not up.is_contiguous():
            raise ValueError("The Triton SiTU benchmark requires contiguous inputs.")
        if gate.shape != up.shape:
            raise ValueError("SiTU gate and up tensors must have identical shapes.")

        output = torch.empty_like(gate)
        numel = gate.numel()
        block_size = 256
        _situ_forward_kernel[(triton.cdiv(numel, block_size),)](
            gate,
            up,
            output,
            numel=numel,
            beta=beta,
            linear_beta=linear_beta,
            BLOCK_SIZE=block_size,
        )
        ctx.save_for_backward(gate, up)
        ctx.beta = beta
        ctx.linear_beta = linear_beta
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        gate, up = ctx.saved_tensors
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        numel = gate.numel()
        block_size = 256
        _situ_backward_kernel[(triton.cdiv(numel, block_size),)](
            gate,
            up,
            grad_output.contiguous(),
            grad_gate,
            grad_up,
            numel=numel,
            beta=ctx.beta,
            linear_beta=ctx.linear_beta,
            BLOCK_SIZE=block_size,
        )
        return grad_gate, grad_up, None, None


def _triton_situ_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: float,
) -> torch.Tensor:
    return _TritonSiTUFunction.apply(gate, up, beta, linear_beta)


@dataclass(frozen=True)
class Invocation:
    forward: Callable[[], torch.Tensor]
    grad_inputs: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class Implementation:
    name: str
    build: Callable[[], Invocation]


@dataclass(frozen=True)
class BenchmarkCase:
    pattern: str
    shape: str
    implementations: tuple[Implementation, ...]
    atol: float
    rtol: float
    grad_relative_l2_tolerance: float = 1e-3


@dataclass(frozen=True)
class Result:
    pattern: str
    shape: str
    implementation: str
    forward_ms: float
    backward_ms: float
    peak_memory_mib: float
    output_max_abs_error: float
    grad_max_abs_error: float
    grad_relative_l2_error: float
    validation: str


def _randn(
    shape: Sequence[int],
    *,
    seed: int,
    dtype: torch.dtype,
    requires_grad: bool = True,
) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randn(
        tuple(shape),
        device="cuda",
        dtype=dtype,
        generator=generator,
        requires_grad=requires_grad,
    )


def _situ_case(
    name: str,
    rows: int,
    dim: int,
    dtype: torch.dtype,
) -> BenchmarkCase:
    def build(function: Callable[..., torch.Tensor]) -> Invocation:
        gate_RD = _randn((rows, dim), seed=11, dtype=dtype)
        up_RD = _randn((rows, dim), seed=12, dtype=dtype)
        return Invocation(
            forward=lambda: function(gate_RD, up_RD, 4.0, 25.0),
            grad_inputs=(gate_RD, up_RD),
        )

    return BenchmarkCase(
        pattern="SiTU",
        shape=f"{name}: [{rows}, {dim}]",
        implementations=(
            Implementation("eager", lambda: build(_situ_glu)),
            Implementation("torch.compile", lambda: build(_compiled_situ_glu)),
            Implementation("handwritten Triton", lambda: build(_triton_situ_glu)),
        ),
        atol=2e-2,
        rtol=2e-2,
    )


def _attention_residual_case(
    name: str,
    num_tokens: int,
    num_saved_residuals: int,
    dim: int,
    dtype: torch.dtype,
) -> BenchmarkCase:
    def build(function: Callable[..., torch.Tensor]) -> Invocation:
        prefix_sum_TD = _randn((num_tokens, dim), seed=21, dtype=dtype)
        block_residual_TND = _randn(
            (num_tokens, num_saved_residuals, dim),
            seed=22,
            dtype=dtype,
        )
        projection_weight_D = _randn((dim,), seed=23, dtype=dtype)
        norm_weight_D = _randn((dim,), seed=24, dtype=dtype)
        return Invocation(
            forward=lambda: function(
                prefix_sum_TD,
                block_residual_TND,
                projection_weight_D,
                norm_weight_D,
                1e-5,
            ),
            grad_inputs=(
                prefix_sum_TD,
                block_residual_TND,
                projection_weight_D,
                norm_weight_D,
            ),
        )

    def fla_implementation(
        prefix_sum_TD: torch.Tensor,
        block_residual_TND: torch.Tensor,
        projection_weight_D: torch.Tensor,
        norm_weight_D: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        residuals = [*block_residual_TND.unbind(dim=1), prefix_sum_TD]
        return fused_attnres(
            query=projection_weight_D,
            residuals=residuals,
            rms_weight=norm_weight_D,
            rms_eps=norm_eps,
            checkpoint_level=1,
        )

    return BenchmarkCase(
        pattern="Attention residual",
        shape=(f"{name}: T={num_tokens}, saved={num_saved_residuals}, D={dim}"),
        implementations=(
            Implementation("eager", lambda: build(_attention_residual)),
            Implementation(
                "torch.compile",
                lambda: build(_compiled_attention_residual),
            ),
            Implementation("FLA Triton", lambda: build(fla_implementation)),
        ),
        atol=3e-2,
        rtol=3e-2,
    )


def _gated_rmsnorm_case(
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> BenchmarkCase:
    def build(function: Callable[..., torch.Tensor]) -> Invocation:
        x = _randn(shape, seed=31, dtype=dtype)
        gate = _randn(shape, seed=32, dtype=dtype)
        weight = _randn((shape[-1],), seed=33, dtype=dtype)
        return Invocation(
            forward=lambda: function(x, gate, weight, 1e-5),
            grad_inputs=(x, gate, weight),
        )

    def fla_implementation(
        x: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return rms_norm_gated(
            x,
            gate,
            weight,
            None,
            activation="sigmoid",
            eps=eps,
        )

    return BenchmarkCase(
        pattern="Gated RMSNorm",
        shape=f"{name}: {list(shape)}",
        implementations=(
            Implementation("eager", lambda: build(_rms_norm_gated)),
            Implementation(
                "torch.compile",
                lambda: build(_compiled_rms_norm_gated),
            ),
            Implementation("FLA Triton", lambda: build(fla_implementation)),
        ),
        atol=2e-2,
        rtol=2e-2,
    )


def _make_cases(preset: str, dtype: torch.dtype) -> list[BenchmarkCase]:
    if preset == "smoke":
        return [
            _situ_case("routed-small", 1024, 3072, dtype),
            _attention_residual_case("small", 1024, 4, 7168, dtype),
            _gated_rmsnorm_case("small", (1, 1024, 96, 128), dtype),
        ]

    return [
        _situ_case("routed-expert", 8192, 3072, dtype),
        _situ_case("shared-expert", 32768, 6144, dtype),
        _attention_residual_case("two-source", 32768, 1, 7168, dtype),
        _attention_residual_case("four-source", 32768, 3, 7168, dtype),
        _attention_residual_case("six-source", 32768, 5, 7168, dtype),
        _attention_residual_case("eight-source", 32768, 7, 7168, dtype),
        _gated_rmsnorm_case("KDA output", (8, 4096, 96, 128), dtype),
    ]


def _gradient(
    output: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    return torch.autograd.grad(
        output,
        inputs,
        grad_outputs=grad_output,
        retain_graph=retain_graph,
    )


def _max_abs_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    return float((actual.float() - expected.float()).abs().max().detach())


def _relative_l2_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    difference = actual.float() - expected.float()
    denominator = expected.float().norm().clamp_min(1e-30)
    return float((difference.norm() / denominator).detach())


def _validate_case(
    case: BenchmarkCase,
) -> dict[str, tuple[float, float, float, str]]:
    reference = case.implementations[0].build()
    reference_output = reference.forward()
    grad_output = _randn(
        reference_output.shape,
        seed=41,
        dtype=reference_output.dtype,
        requires_grad=False,
    )
    reference_grads = _gradient(
        reference_output,
        reference.grad_inputs,
        grad_output,
        retain_graph=False,
    )
    reference_output = reference_output.detach()
    reference_grads = tuple(grad.detach() for grad in reference_grads)
    del reference

    errors = {case.implementations[0].name: (0.0, 0.0, 0.0, "reference")}
    for implementation in case.implementations[1:]:
        invocation = implementation.build()
        output = invocation.forward()
        grads = _gradient(
            output,
            invocation.grad_inputs,
            grad_output,
            retain_graph=False,
        )
        output_error = _max_abs_error(output, reference_output)
        grad_error = 0.0
        grad_relative_l2_error = 0.0
        validation = "pass"
        try:
            torch.testing.assert_close(
                output,
                reference_output,
                atol=case.atol,
                rtol=case.rtol,
            )
        except AssertionError:
            validation = "FAIL"
        for actual_grad, reference_grad in zip(grads, reference_grads, strict=True):
            grad_error = max(
                grad_error,
                _max_abs_error(actual_grad, reference_grad),
            )
            relative_l2_error = _relative_l2_error(actual_grad, reference_grad)
            grad_relative_l2_error = max(
                grad_relative_l2_error,
                relative_l2_error,
            )
            if relative_l2_error > case.grad_relative_l2_tolerance:
                validation = "FAIL"
        errors[implementation.name] = (
            output_error,
            grad_error,
            grad_relative_l2_error,
            validation,
        )
        del invocation, output, grads
        gc.collect()
        torch.cuda.empty_cache()
    return errors


def _benchmark_implementation(
    case: BenchmarkCase,
    implementation: Implementation,
    validation: tuple[float, float, float, str],
    *,
    warmup_ms: int,
    rep_ms: int,
) -> Result:
    invocation = implementation.build()

    # Materialize any compilation or autotuning before measuring.
    output = invocation.forward()
    grad_output = _randn(
        output.shape,
        seed=42,
        dtype=output.dtype,
        requires_grad=False,
    )
    _gradient(output, invocation.grad_inputs, grad_output, retain_graph=False)
    torch.cuda.synchronize()
    del output

    forward_ms = float(
        triton.testing.do_bench(
            invocation.forward,
            warmup=warmup_ms,
            rep=rep_ms,
            return_mode="median",
        )
    )

    def forward_backward() -> tuple[torch.Tensor, ...]:
        output = invocation.forward()
        return _gradient(
            output,
            invocation.grad_inputs,
            grad_output,
            retain_graph=False,
        )

    forward_backward()
    torch.cuda.synchronize()
    forward_backward_ms = float(
        triton.testing.do_bench(
            forward_backward,
            warmup=warmup_ms,
            rep=rep_ms,
            return_mode="median",
        )
    )
    backward_ms = max(forward_backward_ms - forward_ms, 0.0)

    del invocation
    gc.collect()
    torch.cuda.empty_cache()

    memory_invocation = implementation.build()
    torch.cuda.synchronize()
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    memory_output = memory_invocation.forward()
    memory_grad = _randn(
        memory_output.shape,
        seed=43,
        dtype=memory_output.dtype,
        requires_grad=False,
    )
    _gradient(
        memory_output,
        memory_invocation.grad_inputs,
        memory_grad,
        retain_graph=False,
    )
    torch.cuda.synchronize()
    peak_memory_mib = (torch.cuda.max_memory_allocated() - baseline_memory) / MiB
    del memory_invocation, memory_output, memory_grad
    gc.collect()
    torch.cuda.empty_cache()

    return Result(
        pattern=case.pattern,
        shape=case.shape,
        implementation=implementation.name,
        forward_ms=forward_ms,
        backward_ms=backward_ms,
        peak_memory_mib=peak_memory_mib,
        output_max_abs_error=validation[0],
        grad_max_abs_error=validation[1],
        grad_relative_l2_error=validation[2],
        validation=validation[3],
    )


def _print_results(results: list[Result]) -> None:
    print("Backward latency is derived as median(F+B) - median(F).")
    print(
        "| Pattern | Shape | Implementation | Forward (ms) | "
        "Backward (ms) | F+B (ms) | Speedup vs eager | Peak delta (MiB) | "
        "Output max abs | Grad max abs | Grad rel L2 | Validation |"
    )
    print(
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | --- |"
    )
    eager_totals: dict[tuple[str, str], float] = {}
    for result in results:
        total_ms = result.forward_ms + result.backward_ms
        key = (result.pattern, result.shape)
        if result.implementation == "eager":
            eager_totals[key] = total_ms
        speedup = eager_totals[key] / total_ms
        print(
            f"| {result.pattern} | {result.shape} | {result.implementation} | "
            f"{result.forward_ms:.3f} | {result.backward_ms:.3f} | "
            f"{total_ms:.3f} | {speedup:.2f}x | "
            f"{result.peak_memory_mib:.1f} | "
            f"{result.output_max_abs_error:.3e} | "
            f"{result.grad_max_abs_error:.3e} | "
            f"{result.grad_relative_l2_error:.3e} | {result.validation} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=("smoke", "k3"),
        default="smoke",
        help="Use fast validation shapes or full Kimi K3 benchmark shapes.",
    )
    parser.add_argument(
        "--pattern",
        choices=("all", "situ", "attention-residual", "gated-rmsnorm"),
        default="all",
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if args.warmup_ms <= 0 or args.rep_ms <= 0:
        raise ValueError("warmup-ms and rep-ms must be positive.")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_float32_matmul_precision("highest")
    dtype = torch.bfloat16
    cases = _make_cases(args.preset, dtype)
    pattern_names = {
        "situ": "SiTU",
        "attention-residual": "Attention residual",
        "gated-rmsnorm": "Gated RMSNorm",
    }
    if args.pattern != "all":
        cases = [case for case in cases if case.pattern == pattern_names[args.pattern]]

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"GPU: {device.name}")
    print(
        f"Versions: torch={torch.__version__}; triton={triton.__version__}; "
        f"flash-linear-attention={version('flash-linear-attention')}"
    )
    print(f"Preset: {args.preset}; dtype: {dtype}; FLA handwritten kernels enabled")

    results = []
    for case in cases:
        print(f"Validating {case.pattern} ({case.shape})...", flush=True)
        errors = _validate_case(case)
        for implementation in case.implementations:
            print(
                f"Benchmarking {case.pattern} / {implementation.name}...",
                flush=True,
            )
            results.append(
                _benchmark_implementation(
                    case,
                    implementation,
                    errors[implementation.name],
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                )
            )

    print()
    _print_results(results)


if __name__ == "__main__":
    main()
