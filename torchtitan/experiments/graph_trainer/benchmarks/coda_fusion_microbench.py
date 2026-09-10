# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone eager-versus-FlexGEMM microbenchmarks for DSV3 CODA patterns.

Every case uses a shape grounded in the DSV3-671B joint FX graph captured with
local batch 24 and sequence length 4096. The eager functions intentionally use
plain PyTorch operations so the source epilogue remains visible. The FlexGEMM
functions spell out the corresponding epilogue callback and are compiled as a
full graph with Inductor.

Run one case before attempting the full suite because several cases allocate
multiple gigabytes of inputs::

    python -m torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_microbench \
        --case f3_attention_output

Pass ``--config`` once to force one QuACK configuration for every FlexGEMM in
a case, or once per FlexGEMM to tune a multi-GEMM case independently::

    --config '{"tile_m": 256, "tile_n": 256, "cluster_m": 2, "cluster_n": 1}'
"""

import argparse
import ast
import dataclasses
import importlib.metadata
import inspect
import json
import re
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch._higher_order_ops import flex_gemm
from torch._inductor.utils import run_and_get_code


M = 24 * 4096
MODEL_WIDTH = 7168
SHARED_EXPERT_WIDTH = 2048
RMSNORM_GROUP = 512
RMSNORM_BACKWARD_GROUP = 128
EPS = 1e-5

TensorTree = torch.Tensor | tuple["TensorTree", ...] | list["TensorTree"]
BenchmarkFn = Callable[..., TensorTree]
InputFactory = Callable[[torch.device], tuple[torch.Tensor, ...]]
FlexFactory = Callable[[tuple[dict[str, Any], ...]], BenchmarkFn]


@dataclasses.dataclass(frozen=True)
class BenchmarkCase:
    name: str
    pattern: str
    description: str
    shape: str
    num_flex_gemms: int
    make_inputs: InputFactory
    eager: BenchmarkFn
    make_flex: FlexFactory
    atol: float = 0.05
    rtol: float = 0.02
    fast_math_flex_gemms: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class Timing:
    median_ms: float
    minimum_ms: float
    maximum_ms: float
    round_means_ms: tuple[float, ...]
    samples_ms: tuple[float, ...]


def _randn(
    shape: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    scale: float = 0.02,
) -> torch.Tensor:
    return torch.empty(tuple(shape), device=device, dtype=dtype).normal_(std=scale)


def _bf16(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    return _randn(shape, device=device, dtype=torch.bfloat16)


def _fp32(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    return _randn(shape, device=device, dtype=torch.float32)


def eager_b1_lm_head_input_grad_cast(
    grad: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    input_grad = torch.mm(grad, weight)
    return input_grad.float()


def make_flex_b1_lm_head_input_grad_cast(
    options: tuple[dict[str, Any], ...],
) -> BenchmarkFn:
    def flex_b1_lm_head_input_grad_cast(
        grad: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        def epilogue(accumulator: torch.Tensor) -> torch.Tensor:
            return accumulator.float().bfloat16().float()

        return flex_gemm(
            torch.mm,
            (grad, weight),
            epilogue,
            kernel_options=options[0],
        )

    return flex_b1_lm_head_input_grad_cast


def eager_b6_weight_grad_cast(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    weight_grad = torch.mm(lhs, rhs)
    return weight_grad.float()


def make_flex_b6_weight_grad_cast(
    options: tuple[dict[str, Any], ...],
) -> BenchmarkFn:
    def flex_b6_weight_grad_cast(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        def epilogue(accumulator: torch.Tensor) -> torch.Tensor:
            return accumulator.float().bfloat16().float()

        return flex_gemm(
            torch.mm,
            (lhs, rhs),
            epilogue,
            kernel_options=options[0],
        )

    return flex_b6_weight_grad_cast


def eager_f6_router_sigmoid_bias(
    tokens: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_scores = torch.sigmoid(torch.mm(tokens, weight))
    return raw_scores, raw_scores + expert_bias


def make_flex_f6_router_sigmoid_bias(
    options: tuple[dict[str, Any], ...],
) -> BenchmarkFn:
    def flex_f6_router_sigmoid_bias(
        tokens: torch.Tensor,
        weight: torch.Tensor,
        expert_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bias_2d = expert_bias.view(1, -1)

        def epilogue(
            accumulator: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            raw_scores = torch.sigmoid(accumulator)
            return raw_scores, raw_scores + bias_2d

        return flex_gemm(
            torch.mm,
            (tokens, weight),
            epilogue,
            kernel_options=options[0],
        )

    return flex_f6_router_sigmoid_bias


def eager_f4_shared_expert_swiglu(
    tokens: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w1_output = torch.mm(tokens, w1)
    activated = torch.nn.functional.silu(w1_output)
    gate = torch.mm(tokens, w3)
    return activated, gate, activated * gate


def make_flex_f4_shared_expert_swiglu(
    options: tuple[dict[str, Any], ...],
) -> BenchmarkFn:
    def flex_f4_shared_expert_swiglu(
        tokens: torch.Tensor,
        w1: torch.Tensor,
        w3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def silu_epilogue(accumulator: torch.Tensor) -> torch.Tensor:
            rounded = accumulator.float().bfloat16()
            return torch.nn.functional.silu(rounded)

        activated = flex_gemm(
            torch.mm,
            (tokens, w1),
            silu_epilogue,
            kernel_options=options[0],
        )

        def gate_epilogue(
            accumulator: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            gate = accumulator.float().bfloat16()
            return gate, activated * gate

        gate, product = flex_gemm(
            torch.mm,
            (tokens, w3),
            gate_epilogue,
            kernel_options=options[1],
        )
        return activated, gate, product

    return flex_f4_shared_expert_swiglu


def eager_b2_shared_expert_swiglu_backward(
    output_grad: torch.Tensor,
    w2: torch.Tensor,
    saved_silu: torch.Tensor,
    saved_gate: torch.Tensor,
    saved_w1: torch.Tensor,
    w3: torch.Tensor,
    w1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    branch_grad = torch.mm(output_grad, w2)
    gate_grad = branch_grad * saved_silu
    silu_grad = torch.ops.aten.silu_backward.default(
        branch_grad * saved_gate,
        saved_w1,
    )
    w3_input_grad = torch.mm(gate_grad, w3)
    w1_input_grad = torch.mm(silu_grad, w1)
    return gate_grad, silu_grad, w3_input_grad + w1_input_grad


def make_flex_b2_shared_expert_swiglu_backward(
    options: tuple[dict[str, Any], ...],
) -> BenchmarkFn:
    def flex_b2_shared_expert_swiglu_backward(
        output_grad: torch.Tensor,
        w2: torch.Tensor,
        saved_silu: torch.Tensor,
        saved_gate: torch.Tensor,
        saved_w1: torch.Tensor,
        w3: torch.Tensor,
        w1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def branch_epilogue(
            accumulator: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            branch_grad = accumulator.float().bfloat16()
            gate_grad = branch_grad * saved_silu
            silu_grad = torch.ops.aten.silu_backward.default(
                branch_grad * saved_gate,
                saved_w1,
            )
            return gate_grad, silu_grad

        gate_grad, silu_grad = flex_gemm(
            torch.mm,
            (output_grad, w2),
            branch_epilogue,
            kernel_options=options[0],
        )
        w3_input_grad = torch.mm(gate_grad, w3)

        def input_grad_epilogue(accumulator: torch.Tensor) -> torch.Tensor:
            w1_input_grad = accumulator.float().bfloat16()
            return w3_input_grad + w1_input_grad

        input_grad = flex_gemm(
            torch.mm,
            (silu_grad, w1),
            input_grad_epilogue,
            kernel_options=options[1],
        )
        return gate_grad, silu_grad, input_grad

    return flex_b2_shared_expert_swiglu_backward


def eager_b4_router_input_grad_add(
    score_grad: torch.Tensor,
    router_weight: torch.Tensor,
    expert_input_grad: torch.Tensor,
) -> torch.Tensor:
    router_input_grad = torch.mm(score_grad, router_weight).bfloat16()
    return expert_input_grad + router_input_grad


def make_flex_b4_router_input_grad_add(
    options: tuple[dict[str, Any], ...],
) -> BenchmarkFn:
    def flex_b4_router_input_grad_add(
        score_grad: torch.Tensor,
        router_weight: torch.Tensor,
        expert_input_grad: torch.Tensor,
    ) -> torch.Tensor:
        def epilogue(accumulator: torch.Tensor) -> torch.Tensor:
            router_input_grad = accumulator.bfloat16()
            return expert_input_grad + router_input_grad

        return flex_gemm(
            torch.mm,
            (score_grad, router_weight),
            epilogue,
            kernel_options=options[0],
        )

    return flex_b4_router_input_grad_add


def eager_b5_mla_rmsnorm_backward(
    output_grad: torch.Tensor,
    projection_weight: torch.Tensor,
    norm_input: torch.Tensor,
    rstd: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad = torch.mm(output_grad, projection_weight)
    return torch.ops.aten._fused_rms_norm_backward.default(
        grad,
        norm_input,
        [grad.shape[-1]],
        rstd,
        gamma,
        [True, True],
    )


def make_flex_b5_mla_rmsnorm_backward(
    options: tuple[dict[str, Any], ...],
) -> BenchmarkFn:
    def flex_b5_mla_rmsnorm_backward(
        output_grad: torch.Tensor,
        projection_weight: torch.Tensor,
        norm_input: torch.Tensor,
        rstd: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_2d = gamma.view(1, -1)

        def epilogue(
            accumulator: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            rounded = accumulator.float().bfloat16()
            rounded_fp32 = rounded.float()
            x_hat = norm_input.float() * rstd
            grad_x_hat = rounded_fp32 * gamma_2d.float()
            row_products = x_hat * grad_x_hat
            partial_row_dot = row_products.view(
                row_products.shape[0],
                -1,
                RMSNORM_BACKWARD_GROUP,
            ).sum(-1)
            return rounded, partial_row_dot

        rounded, partial_row_dot = flex_gemm(
            torch.mm,
            (output_grad, projection_weight),
            epilogue,
            kernel_options=options[0],
        )
        rounded_fp32 = rounded.float()
        x_hat = norm_input.float() * rstd
        grad_x_hat = rounded_fp32 * gamma_2d.float()
        row_dot = partial_row_dot.sum(-1, keepdim=True)
        correction = (x_hat / rounded.shape[-1]) * row_dot
        grad_input = ((grad_x_hat - correction) * rstd).bfloat16()
        grad_weight = (rounded_fp32 * x_hat).sum(0).bfloat16()
        return grad_input, grad_weight

    return flex_b5_mla_rmsnorm_backward


def eager_b7_attention_input_grad_merge(
    kv_grad: torch.Tensor,
    kv_weight: torch.Tensor,
    q_grad: torch.Tensor,
    q_weight: torch.Tensor,
) -> torch.Tensor:
    kv_input_grad = torch.mm(kv_grad, kv_weight)
    q_input_grad = torch.mm(q_grad, q_weight)
    return kv_input_grad + q_input_grad


def make_flex_b7_attention_input_grad_merge(
    options: tuple[dict[str, Any], ...],
) -> BenchmarkFn:
    def flex_b7_attention_input_grad_merge(
        kv_grad: torch.Tensor,
        kv_weight: torch.Tensor,
        q_grad: torch.Tensor,
        q_weight: torch.Tensor,
    ) -> torch.Tensor:
        kv_input_grad = torch.mm(kv_grad, kv_weight)

        def epilogue(accumulator: torch.Tensor) -> torch.Tensor:
            q_input_grad = accumulator.float().bfloat16()
            return kv_input_grad + q_input_grad

        return flex_gemm(
            torch.mm,
            (q_grad, q_weight),
            epilogue,
            kernel_options=options[0],
        )

    return flex_b7_attention_input_grad_merge


def eager_f2_q_rmsnorm(
    tokens: torch.Tensor,
    wq_a: torch.Tensor,
    gamma: torch.Tensor,
    wq_b: torch.Tensor,
) -> torch.Tensor:
    q_low_rank = torch.mm(tokens, wq_a)
    normalized = torch.nn.functional.rms_norm(
        q_low_rank,
        (q_low_rank.shape[-1],),
        gamma,
        EPS,
    )
    return torch.mm(normalized, wq_b)


def make_flex_f2_q_rmsnorm(
    options: tuple[dict[str, Any], ...],
) -> BenchmarkFn:
    def flex_f2_q_rmsnorm(
        tokens: torch.Tensor,
        wq_a: torch.Tensor,
        gamma: torch.Tensor,
        wq_b: torch.Tensor,
    ) -> torch.Tensor:
        gamma_2d = gamma.view(1, -1)

        def first_epilogue(
            accumulator: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            rounded = accumulator.float().bfloat16()
            rounded_fp32 = rounded.float()
            weighted = (rounded_fp32 * gamma_2d).bfloat16()
            partial_mean_square = (
                rounded_fp32.view(rounded.shape[0], -1, RMSNORM_GROUP).square().mean(-1)
            )
            return weighted, partial_mean_square

        weighted, partial_mean_square = flex_gemm(
            torch.mm,
            (tokens, wq_a),
            first_epilogue,
            kernel_options=options[0],
        )
        rstd = (partial_mean_square.mean(-1, keepdim=True) + EPS).rsqrt()

        def second_epilogue(accumulator: torch.Tensor) -> torch.Tensor:
            return (accumulator.float() * rstd).bfloat16()

        return flex_gemm(
            torch.mm,
            (weighted, wq_b),
            second_epilogue,
            kernel_options=options[1],
        )

    return flex_f2_q_rmsnorm


def eager_f2_kv_rmsnorm(
    tokens: torch.Tensor,
    wkv_a: torch.Tensor,
    gamma: torch.Tensor,
    wkv_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_low_rank = torch.mm(tokens, wkv_a)
    active = kv_low_rank[:, :512]
    rope_tail = kv_low_rank[:, 512:]
    normalized = torch.nn.functional.rms_norm(active, (512,), gamma, EPS)
    return torch.mm(normalized, wkv_b), rope_tail


def make_flex_f2_kv_rmsnorm(
    options: tuple[dict[str, Any], ...],
) -> BenchmarkFn:
    def flex_f2_kv_rmsnorm(
        tokens: torch.Tensor,
        wkv_a: torch.Tensor,
        gamma: torch.Tensor,
        wkv_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_full = torch.nn.functional.pad(
            gamma.view(1, 512),
            (0, 64),
            value=1.0,
        )

        def first_epilogue(
            accumulator: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            raw = accumulator.float().bfloat16()
            raw_fp32 = raw.float()
            weighted_full = (raw_fp32 * gamma_full).bfloat16()
            partial_mean_square = raw_fp32.view(raw.shape[0], -1, 64).square().mean(-1)
            return weighted_full, raw, partial_mean_square

        weighted_full, raw, partial_mean_square = flex_gemm(
            torch.mm,
            (tokens, wkv_a),
            first_epilogue,
            kernel_options=options[0],
        )
        weighted = weighted_full[:, :512]
        active_partials = partial_mean_square[:, :8]
        rstd = (active_partials.mean(-1, keepdim=True) + EPS).rsqrt()

        def second_epilogue(accumulator: torch.Tensor) -> torch.Tensor:
            return (accumulator.float() * rstd).bfloat16()

        output = flex_gemm(
            torch.mm,
            (weighted, wkv_b),
            second_epilogue,
            kernel_options=options[1],
        )
        return output, raw[:, 512:]

    return flex_f2_kv_rmsnorm


def eager_f3_attention_output(
    attention: torch.Tensor,
    wo: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    attention_output = torch.mm(attention, wo)
    total = residual + attention_output
    normalized = torch.nn.functional.rms_norm(
        total,
        (total.shape[-1],),
        gamma,
        EPS,
    )
    return normalized, total


def make_flex_f3_attention_output(
    options: tuple[dict[str, Any], ...],
) -> BenchmarkFn:
    def flex_f3_attention_output(
        attention: torch.Tensor,
        wo: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def epilogue(
            accumulator: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            attention_output = accumulator.float().bfloat16()
            total = residual + attention_output
            partial_mean_square = (
                total.float().view(total.shape[0], -1, RMSNORM_GROUP).square().mean(-1)
            )
            return total, partial_mean_square

        total, partial_mean_square = flex_gemm(
            torch.mm,
            (attention, wo),
            epilogue,
            kernel_options=options[0],
        )
        rstd = (partial_mean_square.mean(-1, keepdim=True) + EPS).rsqrt()
        normalized = (total.float() * rstd * gamma.float()).bfloat16()
        return normalized, total

    return flex_f3_attention_output


def eager_f3_moe_output(
    shared_activation: torch.Tensor,
    shared_w2: torch.Tensor,
    routed_output: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shared_output = torch.mm(shared_activation, shared_w2)
    moe_output = routed_output + shared_output
    total = residual + moe_output
    normalized = torch.nn.functional.rms_norm(
        total,
        (total.shape[-1],),
        gamma,
        EPS,
    )
    return normalized, total


def make_flex_f3_moe_output(
    options: tuple[dict[str, Any], ...],
) -> BenchmarkFn:
    def flex_f3_moe_output(
        shared_activation: torch.Tensor,
        shared_w2: torch.Tensor,
        routed_output: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def epilogue(
            accumulator: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            shared_output = accumulator.float().bfloat16()
            moe_output = routed_output + shared_output
            total = residual + moe_output
            partial_mean_square = (
                total.float().view(total.shape[0], -1, RMSNORM_GROUP).square().mean(-1)
            )
            return total, partial_mean_square

        total, partial_mean_square = flex_gemm(
            torch.mm,
            (shared_activation, shared_w2),
            epilogue,
            kernel_options=options[0],
        )
        rstd = (partial_mean_square.mean(-1, keepdim=True) + EPS).rsqrt()
        normalized = (total.float() * rstd * gamma.float()).bfloat16()
        return normalized, total

    return flex_f3_moe_output


def _make_b1_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return _bf16((12288, 129280), device), _bf16((129280, MODEL_WIDTH), device)


def _make_b6_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return _bf16((SHARED_EXPERT_WIDTH, M), device), _bf16((M, MODEL_WIDTH), device)


def _make_f6_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _fp32((M, MODEL_WIDTH), device),
        _fp32((MODEL_WIDTH, 256), device),
        _fp32((256,), device),
    )


def _make_f4_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((M, MODEL_WIDTH), device),
        _bf16((MODEL_WIDTH, SHARED_EXPERT_WIDTH), device),
        _bf16((MODEL_WIDTH, SHARED_EXPERT_WIDTH), device),
    )


def _make_b2_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((M, MODEL_WIDTH), device),
        _bf16((MODEL_WIDTH, SHARED_EXPERT_WIDTH), device),
        _bf16((M, SHARED_EXPERT_WIDTH), device),
        _bf16((M, SHARED_EXPERT_WIDTH), device),
        _bf16((M, SHARED_EXPERT_WIDTH), device),
        _bf16((SHARED_EXPERT_WIDTH, MODEL_WIDTH), device),
        _bf16((SHARED_EXPERT_WIDTH, MODEL_WIDTH), device),
    )


def _make_b4_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _fp32((M, 256), device),
        _fp32((256, MODEL_WIDTH), device),
        _bf16((M, MODEL_WIDTH), device),
    )


def _make_b5_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((M, 24576), device),
        _bf16((24576, 1536), device),
        _bf16((M, 1536), device),
        torch.empty((M, 1), device=device, dtype=torch.float32).uniform_(0.5, 1.5),
        _bf16((1536,), device),
    )


def _make_b7_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((M, 576), device),
        _bf16((576, MODEL_WIDTH), device),
        _bf16((M, 1536), device),
        _bf16((1536, MODEL_WIDTH), device),
    )


def _make_f2_q_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((M, MODEL_WIDTH), device),
        _bf16((MODEL_WIDTH, 1536), device),
        torch.ones((1536,), device=device, dtype=torch.bfloat16),
        _bf16((1536, 24576), device),
    )


def _make_f2_kv_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((M, MODEL_WIDTH), device),
        _bf16((MODEL_WIDTH, 576), device),
        torch.ones((512,), device=device, dtype=torch.bfloat16),
        _bf16((512, 32768), device),
    )


def _make_f3_attention_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((M, 16384), device),
        _bf16((16384, MODEL_WIDTH), device),
        _bf16((M, MODEL_WIDTH), device),
        torch.ones((MODEL_WIDTH,), device=device, dtype=torch.bfloat16),
    )


def _make_f3_moe_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((M, SHARED_EXPERT_WIDTH), device),
        _bf16((SHARED_EXPERT_WIDTH, MODEL_WIDTH), device),
        _bf16((M, MODEL_WIDTH), device),
        _bf16((M, MODEL_WIDTH), device),
        torch.ones((MODEL_WIDTH,), device=device, dtype=torch.bfloat16),
    )


CASES: dict[str, BenchmarkCase] = {
    case.name: case
    for case in (
        BenchmarkCase(
            "b1_lm_head_input_grad_cast",
            "B1",
            "LM-head input-gradient BF16 store followed by FP32 cast",
            "(12288, 129280) @ (129280, 7168)",
            1,
            _make_b1_inputs,
            eager_b1_lm_head_input_grad_cast,
            make_flex_b1_lm_head_input_grad_cast,
        ),
        BenchmarkCase(
            "b2_shared_expert_swiglu_backward",
            "B2",
            "Shared-expert SwiGLU branch derivatives and input-gradient merge",
            "M=98304, D=7168, P=2048; three BF16 GEMMs",
            2,
            _make_b2_inputs,
            eager_b2_shared_expert_swiglu_backward,
            make_flex_b2_shared_expert_swiglu_backward,
            3.1e-5,
            1e-2,
        ),
        BenchmarkCase(
            "b4_router_input_grad_add",
            "B4",
            "FP32 router input-gradient GEMM, BF16 store, and expert-gradient add",
            "(98304, 256) @ (256, 7168)",
            1,
            _make_b4_inputs,
            eager_b4_router_input_grad_add,
            make_flex_b4_router_input_grad_add,
            4.9e-4,
            1e-2,
        ),
        BenchmarkCase(
            "b5_mla_q_rmsnorm_backward",
            "B5",
            "MLA Q input-gradient projection plus RMSNorm backward",
            "(98304, 24576) @ (24576, 1536)",
            1,
            _make_b5_inputs,
            eager_b5_mla_rmsnorm_backward,
            make_flex_b5_mla_rmsnorm_backward,
            0.05,
            0.02,
        ),
        BenchmarkCase(
            "b6_weight_grad_cast",
            "B6",
            "Frequent shared-expert BF16 weight-gradient GEMM and FP32 cast",
            "(2048, 98304) @ (98304, 7168)",
            1,
            _make_b6_inputs,
            eager_b6_weight_grad_cast,
            make_flex_b6_weight_grad_cast,
        ),
        BenchmarkCase(
            "b7_attention_input_grad_merge",
            "B7",
            "KV and Q input-gradient GEMMs followed by BF16 add",
            "(98304, 576) @ (576, 7168) + (98304, 1536) @ (1536, 7168)",
            1,
            _make_b7_inputs,
            eager_b7_attention_input_grad_merge,
            make_flex_b7_attention_input_grad_merge,
        ),
        BenchmarkCase(
            "f2_q_rmsnorm",
            "F2-Q",
            "MLA Q low-rank projection, RMSNorm, and expanded projection",
            "7168 -> 1536 -> 24576 at M=98304",
            2,
            _make_f2_q_inputs,
            eager_f2_q_rmsnorm,
            make_flex_f2_q_rmsnorm,
            3.2e-2,
            2e-2,
        ),
        BenchmarkCase(
            "f2_kv_rmsnorm",
            "F2-KV",
            "Segmented MLA KV projection, RMSNorm, and expanded projection",
            "7168 -> 576, RMSNorm(512), 512 -> 32768 at M=98304",
            2,
            _make_f2_kv_inputs,
            eager_f2_kv_rmsnorm,
            make_flex_f2_kv_rmsnorm,
            1.6e-2,
            2e-2,
        ),
        BenchmarkCase(
            "f3_attention_output",
            "F3-A",
            "Attention WO projection, residual add, and FFN RMSNorm",
            "(98304, 16384) @ (16384, 7168)",
            1,
            _make_f3_attention_inputs,
            eager_f3_attention_output,
            make_flex_f3_attention_output,
            0.07,
            2e-2,
        ),
        BenchmarkCase(
            "f3_moe_output",
            "F3-B",
            "Shared W2 projection, routed add, residual add, and next RMSNorm",
            "(98304, 2048) @ (2048, 7168)",
            1,
            _make_f3_moe_inputs,
            eager_f3_moe_output,
            make_flex_f3_moe_output,
            0.07,
            2e-2,
        ),
        BenchmarkCase(
            "f4_shared_expert_swiglu",
            "F4",
            "Shared-expert W1/W3 GEMMs with SiLU and multiply",
            "two (98304, 7168) @ (7168, 2048) GEMMs",
            2,
            _make_f4_inputs,
            eager_f4_shared_expert_swiglu,
            make_flex_f4_shared_expert_swiglu,
            fast_math_flex_gemms=(0,),
        ),
        BenchmarkCase(
            "f6_router_sigmoid_bias",
            "F6",
            "FP32 router GEMM with sigmoid and expert bias",
            "(98304, 7168) @ (7168, 256)",
            1,
            _make_f6_inputs,
            eager_f6_router_sigmoid_bias,
            make_flex_f6_router_sigmoid_bias,
            1.5e-5,
            1e-4,
            fast_math_flex_gemms=(0,),
        ),
    )
}


def _kernel_options(
    case: BenchmarkCase,
    *,
    configs: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    parsed_configs = tuple(json.loads(config) for config in configs)
    if any(not isinstance(config, dict) for config in parsed_configs):
        raise ValueError("every --config value must decode to a JSON object")
    if len(parsed_configs) not in (0, 1, case.num_flex_gemms):
        raise ValueError(
            f"{case.name} has {case.num_flex_gemms} FlexGEMMs; pass zero, one, "
            f"or {case.num_flex_gemms} --config values"
        )
    if len(parsed_configs) == 1:
        parsed_configs = parsed_configs * case.num_flex_gemms
    if not parsed_configs:
        parsed_configs = ({},) * case.num_flex_gemms

    options = []
    for index, config in enumerate(parsed_configs):
        value: dict[str, Any] = {"backend": "QUACK", "tuned": True}
        if index in case.fast_math_flex_gemms:
            value["fast_math"] = True
        if config:
            value["config"] = config
        options.append(value)
    return tuple(options)


def _flatten(value: TensorTree) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    tensors = []
    for child in value:
        tensors.extend(_flatten(child))
    return tensors


def _correctness(
    actual: TensorTree,
    expected: TensorTree,
    *,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    actual_tensors = _flatten(actual)
    expected_tensors = _flatten(expected)
    if len(actual_tensors) != len(expected_tensors):
        raise AssertionError(
            f"output count differs: {len(actual_tensors)} != {len(expected_tensors)}"
        )

    reports = []
    for index, (actual_tensor, expected_tensor) in enumerate(
        zip(actual_tensors, expected_tensors, strict=True)
    ):
        if actual_tensor.shape != expected_tensor.shape:
            raise AssertionError(
                f"output {index} shape differs: "
                f"{actual_tensor.shape} != {expected_tensor.shape}"
            )
        difference = (actual_tensor.float() - expected_tensor.float()).abs()
        max_abs = difference.max().item()
        mean_abs = difference.mean().item()
        allowed = atol + rtol * expected_tensor.float().abs()
        passed = (difference <= allowed).all().item()
        relative_l2 = (
            (difference.square().sum() / expected_tensor.float().square().sum())
            .sqrt()
            .item()
        )
        reports.append(
            {
                "output": index,
                "shape": list(actual_tensor.shape),
                "dtype": str(actual_tensor.dtype),
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "relative_l2": relative_l2,
                "atol": atol,
                "rtol": rtol,
                "passed": passed,
            }
        )
        if not passed:
            raise AssertionError(
                f"output {index} violates atol={atol}, rtol={rtol}; "
                f"max_abs={max_abs}, relative_l2={relative_l2}"
            )
    return reports


def _benchmark(
    fn: BenchmarkFn,
    inputs: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    iterations: int,
    flush_cache_mb: int,
) -> Timing:
    cache = None
    if flush_cache_mb:
        cache = torch.empty(
            flush_cache_mb * 1024 * 1024 // 4,
            device=inputs[0].device,
            dtype=torch.float32,
        )

    for _ in range(warmup):
        if cache is not None:
            cache.zero_()
        fn(*inputs)
    torch.cuda.synchronize(inputs[0].device)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        if cache is not None:
            cache.zero_()
        start.record()
        fn(*inputs)
        end.record()
    torch.cuda.synchronize(inputs[0].device)
    samples = tuple(start.elapsed_time(end) for start, end in zip(starts, ends))
    return Timing(
        median_ms=statistics.median(samples),
        minimum_ms=min(samples),
        maximum_ms=max(samples),
        round_means_ms=(statistics.mean(samples),),
        samples_ms=samples,
    )


def _make_cuda_graph_replay(
    fn: BenchmarkFn,
    inputs: tuple[torch.Tensor, ...],
    *,
    warmup: int,
) -> Callable[[], None]:
    device = inputs[0].device
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream):
        for _ in range(max(1, warmup)):
            fn(*inputs)
    torch.cuda.current_stream(device).wait_stream(warmup_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = fn(*inputs)

    def replay() -> None:
        graph.replay()
        # Keep graph-owned output storage alive for the lifetime of the replay.
        _ = static_output

    return replay


def _benchmark_cuda_graphs(
    candidates: tuple[tuple[str, Callable[[], None]], ...],
    *,
    device: torch.device,
    rounds: int,
    iterations: int,
) -> dict[str, Timing]:
    round_samples: dict[str, list[tuple[float, ...]]] = {
        name: [] for name, _ in candidates
    }
    for round_index in range(rounds):
        ordered = candidates if round_index % 2 == 0 else candidates[::-1]
        for name, replay in ordered:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                replay()
            end.record()
            torch.cuda.synchronize(device)
            elapsed_ms = start.elapsed_time(end)
            round_samples[name].append((elapsed_ms / iterations,))

    timings = {}
    for name, samples_by_round in round_samples.items():
        samples = tuple(value for group in samples_by_round for value in group)
        round_means = tuple(statistics.mean(group) for group in samples_by_round)
        timings[name] = Timing(
            median_ms=statistics.median(round_means),
            minimum_ms=min(samples),
            maximum_ms=max(samples),
            round_means_ms=round_means,
            samples_ms=samples,
        )
    return timings


def _run_case(
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    inputs = case.make_inputs(device)
    options = _kernel_options(
        case,
        configs=args.config,
    )
    flex = case.make_flex(options)

    eager_compile_start = time.perf_counter()
    compiled_eager = torch.compile(case.eager, backend="inductor", fullgraph=True)
    compiled_expected = compiled_eager(*inputs)
    torch.cuda.synchronize(device)
    eager_compile_seconds = time.perf_counter() - eager_compile_start

    flex_compile_start = time.perf_counter()
    compiled_flex = torch.compile(flex, backend="inductor", fullgraph=True)
    actual, generated_sources = run_and_get_code(compiled_flex, *inputs)
    torch.cuda.synchronize(device)
    flex_compile_seconds = time.perf_counter() - flex_compile_start

    generated_source = "\n".join(generated_sources)
    num_flex_gemm_calls = generated_source.count("flex_gemm_epilogue(")
    if num_flex_gemm_calls != case.num_flex_gemms:
        raise RuntimeError(
            f"expected {case.num_flex_gemms} generated FlexGEMM calls, "
            f"found {num_flex_gemm_calls}"
        )
    selected_configs = tuple(
        dict(ast.literal_eval(config_key))
        for config_key in re.findall(
            r"config_key=(\(.*?\)), config_is_lowering_validated",
            generated_source,
        )
    )
    if len(selected_configs) != case.num_flex_gemms:
        raise RuntimeError(
            f"expected {case.num_flex_gemms} generated QuACK configs, "
            f"found {len(selected_configs)}"
        )

    expected = case.eager(*inputs)
    torch.cuda.synchronize(device)
    flex_correctness = _correctness(
        actual,
        expected,
        atol=case.atol,
        rtol=case.rtol,
    )
    compiled_eager_correctness = _correctness(
        compiled_expected,
        expected,
        atol=case.atol,
        rtol=case.rtol,
    )

    source_eager_timing = _benchmark(
        case.eager,
        inputs,
        warmup=args.warmup,
        iterations=args.iterations,
        flush_cache_mb=args.flush_cache_mb,
    )
    compiled_eager_replay = _make_cuda_graph_replay(
        compiled_eager,
        inputs,
        warmup=args.warmup,
    )
    flex_replay = _make_cuda_graph_replay(
        compiled_flex,
        inputs,
        warmup=args.warmup,
    )
    replay_timings = _benchmark_cuda_graphs(
        (
            ("compiled_eager", compiled_eager_replay),
            ("flex_gemm", flex_replay),
        ),
        device=device,
        rounds=args.rounds,
        iterations=args.iterations,
    )
    compiled_eager_timing = replay_timings["compiled_eager"]
    flex_timing = replay_timings["flex_gemm"]
    result = {
        "case": case.name,
        "pattern": case.pattern,
        "shape": case.shape,
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cutlass_dsl_version": importlib.metadata.version("nvidia-cutlass-dsl"),
        "triton_version": importlib.metadata.version("triton"),
        "kernel_options": options,
        "selected_quack_configs": selected_configs,
        "benchmark_contract": {
            "primary": "fixed-pointer CUDA-graph replay",
            "rounds": args.rounds,
            "iterations_per_round": args.iterations,
            "alternating_candidate_order": True,
        },
        "compile_seconds": {
            "compiled_eager": eager_compile_seconds,
            "flex_gemm": flex_compile_seconds,
        },
        "generated_flex_gemm_calls": num_flex_gemm_calls,
        "correctness": {
            "compiled_eager": compiled_eager_correctness,
            "flex_gemm": flex_correctness,
        },
        "source_eager": dataclasses.asdict(source_eager_timing),
        "compiled_eager": dataclasses.asdict(compiled_eager_timing),
        "flex_gemm": dataclasses.asdict(flex_timing),
        "speedup_vs_compiled_eager": (
            compiled_eager_timing.median_ms / flex_timing.median_ms
        ),
        "speedup_vs_source_eager": (
            source_eager_timing.median_ms / flex_timing.median_ms
        ),
    }
    print(json.dumps(result, indent=2))
    return result


def _show_cases(cases: Mapping[str, BenchmarkCase] = CASES) -> None:
    print("case | pattern | FlexGEMMs | real shape")
    print("--- | --- | ---: | ---")
    for case in cases.values():
        print(f"{case.name} | {case.pattern} | {case.num_flex_gemms} | {case.shape}")


def _show_source(case: BenchmarkCase) -> None:
    print(f"# {case.name}: eager")
    print(inspect.getsource(case.eager))
    print(f"# {case.name}: FlexGEMM factory")
    print(inspect.getsource(case.make_flex))


def _parse_args(
    cases: Mapping[str, BenchmarkCase] = CASES,
    *,
    description: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--case", choices=(*cases, "all"))
    parser.add_argument("--list", action="store_true", help="list all cases")
    parser.add_argument(
        "--show-source",
        action="store_true",
        help="print the selected eager and FlexGEMM implementations",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="JSON QuACK config; repeat once per FlexGEMM for independent tuning",
    )
    parser.add_argument(
        "--flush-cache-mb",
        type=int,
        default=0,
        help="zero this many MiB before every timed call",
    )
    parser.add_argument("--output", help="write all JSON results to this path")
    args = parser.parse_args()
    if not args.list and args.case is None:
        parser.error("--case is required unless --list is used")
    if args.warmup < 0 or args.rounds <= 0 or args.iterations <= 0:
        parser.error(
            "--warmup must be nonnegative; --rounds and --iterations must be positive"
        )
    return args


def run_benchmark_suite(
    cases: Mapping[str, BenchmarkCase],
    *,
    description: str | None = None,
) -> None:
    args = _parse_args(cases, description=description)
    if args.list:
        _show_cases(cases)
        return

    names = tuple(cases) if args.case == "all" else (args.case,)
    assert all(name is not None for name in names)
    if args.show_source:
        for name in names:
            assert name is not None
            _show_source(cases[name])
        return

    if not torch.cuda.is_available():
        raise RuntimeError("the CODA microbenchmarks require CUDA")
    if torch.cuda.get_device_capability(args.device) < (10, 0):
        raise RuntimeError("QUACK FlexGEMM requires an SM100-or-later GPU")

    results = []
    for name in names:
        assert name is not None
        results.append(_run_case(cases[name], args))
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2)
            output_file.write("\n")


def main() -> None:
    run_benchmark_suite(CASES)


if __name__ == "__main__":
    main()
