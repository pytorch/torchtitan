#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Microbenchmark every registered CODA pattern at representative model sizes.

The eager FX graph is the baseline. The candidate is the same graph after one
named CODA pass has replaced its matching region with FlexGEMM. Timings use
``triton.testing.do_bench`` and are reported in microseconds.

Examples:

    python -m torchtitan.experiments.graph_trainer.benchmarks.coda_microbenchmarks \
        --list-presets

    python -m torchtitan.experiments.graph_trainer.benchmarks.coda_microbenchmarks \
        --preset dsv3-16b --pattern F_swiglu

    python -m torchtitan.experiments.graph_trainer.benchmarks.coda_microbenchmarks \
        --preset kimi-k3-16b --grid 128x256 --grid 256x256 \
        --pattern F_situ --json-output /tmp/coda.json

``--grid`` applies one complete QuACK configuration to every FlexGEMM emitted
by a pattern. ``--config`` accepts either one JSON object with the same
behavior, or a JSON list containing one object per emitted FlexGEMM.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, cast


os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import torch
from torch._inductor.utils import run_and_get_code
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from triton.testing import do_bench

from torchtitan.experiments.graph_trainer.coda_passes import (
    coda_flex_gemm_pass,
    CODA_PATTERN_NAMES,
)


aten = torch.ops.aten
flex_gemm_hop = torch.ops.higher_order.flex_gemm
EPS = 1e-5
SEQUENCE_LENGTH = 4096
TensorTree = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]


@dataclass(frozen=True)
class ModelShape:
    name: str
    family: str
    tokens: int
    model_width: int
    q_lora_rank: int
    q_output_width: int
    kv_lora_rank: int
    kv_rope_width: int
    kv_output_width: int
    attention_width: int
    routed_latent_width: int
    shared_expert_width: int
    dense_expert_width: int
    num_experts: int
    vocab_size: int
    router_activation: str
    loss_chunks: int = 8
    attn_res_width: int = 12


PRESETS: dict[str, ModelShape] = {
    "dsv3-16b": ModelShape(
        name="DeepSeek V3 16B",
        family="dsv3",
        tokens=4 * SEQUENCE_LENGTH,
        model_width=2048,
        q_lora_rank=0,
        q_output_width=16 * (128 + 64),
        kv_lora_rank=512,
        kv_rope_width=64,
        kv_output_width=16 * (128 + 128),
        attention_width=16 * 128,
        routed_latent_width=0,
        shared_expert_width=2 * 1408,
        dense_expert_width=10944,
        num_experts=64,
        vocab_size=102400,
        router_activation="softmax",
    ),
    "dsv3-671b": ModelShape(
        name="DeepSeek V3 671B",
        family="dsv3",
        tokens=4 * SEQUENCE_LENGTH,
        model_width=7168,
        q_lora_rank=1536,
        q_output_width=128 * (128 + 64),
        kv_lora_rank=512,
        kv_rope_width=64,
        kv_output_width=128 * (128 + 128),
        attention_width=128 * 128,
        routed_latent_width=0,
        shared_expert_width=2048,
        dense_expert_width=18432,
        num_experts=256,
        vocab_size=129280,
        router_activation="sigmoid",
    ),
    "kimi-k3": ModelShape(
        name="Kimi K3",
        family="kimi",
        tokens=4 * SEQUENCE_LENGTH,
        model_width=7168,
        q_lora_rank=1536,
        q_output_width=96 * (128 + 64),
        kv_lora_rank=512,
        kv_rope_width=64,
        kv_output_width=96 * (128 + 128),
        attention_width=96 * 128,
        routed_latent_width=3584,
        shared_expert_width=2 * 3072,
        dense_expert_width=33792,
        num_experts=896,
        vocab_size=163840,
        router_activation="sigmoid",
    ),
    "kimi-k3-16b": ModelShape(
        name="Kimi K3 16B",
        family="kimi",
        tokens=4 * SEQUENCE_LENGTH,
        model_width=2048,
        q_lora_rank=512,
        q_output_width=32 * (64 + 32),
        kv_lora_rank=256,
        kv_rope_width=32,
        kv_output_width=32 * (64 + 64),
        attention_width=32 * 64,
        routed_latent_width=1024,
        shared_expert_width=2 * 896,
        dense_expert_width=9728,
        num_experts=192,
        vocab_size=163840,
        router_activation="sigmoid",
    ),
}


@dataclass(frozen=True)
class BenchmarkCase:
    module: torch.nn.Module
    inputs: tuple[torch.Tensor, ...]
    module_fqns: tuple[str, ...] = ()
    upstream_patterns: tuple[str | None, ...] = ()
    backward: bool = False


CaseFactory = Callable[[ModelShape, torch.device], BenchmarkCase]


@dataclass(frozen=True)
class Microbenchmark:
    name: str
    factory: CaseFactory
    formula: str
    direction: str
    families: frozenset[str]
    atol: float = 0.15
    rtol: float = 0.05


MICROBENCHMARKS: dict[str, Microbenchmark] = {}


def register_microbenchmark(
    *,
    formula: str,
    direction: str = "forward",
    families: Sequence[str] = ("dsv3", "kimi"),
    atol: float = 0.15,
    rtol: float = 0.05,
) -> Callable[[CaseFactory], CaseFactory]:
    def register(factory: CaseFactory) -> CaseFactory:
        name = factory.__name__
        if name in MICROBENCHMARKS:
            raise ValueError(f"CODA microbenchmark {name!r} is already registered")
        MICROBENCHMARKS[name] = Microbenchmark(
            name=name,
            factory=factory,
            formula=formula,
            direction=direction,
            families=frozenset(families),
            atol=atol,
            rtol=rtol,
        )
        return factory

    return register


def _bf16(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.empty(tuple(shape), device=device, dtype=torch.bfloat16).normal_(
        std=0.02
    )


def _fp32(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.empty(tuple(shape), device=device, dtype=torch.float32).normal_(
        std=0.02
    )


class _ProjectionNorm(torch.nn.Module):
    def __init__(self, *, split_projection: bool) -> None:
        super().__init__()
        self.split_projection = split_projection

    def forward(self, x, first_weight, norm_weight, second_weight):
        projection = x @ first_weight
        if self.split_projection:
            norm_width = norm_weight.shape[0]
            norm_input, rope = torch.split(
                projection,
                [norm_width, projection.shape[-1] - norm_width],
                dim=-1,
            )
        else:
            norm_input, rope = projection, projection
        norm, rstd = aten._fused_rms_norm.default(
            norm_input,
            [norm_input.shape[-1]],
            norm_weight,
            EPS,
        )
        return projection, rope, norm @ second_weight, rstd


class _ResidualNorm(torch.nn.Module):
    def forward(self, x, weight, residual, norm_weight):
        hidden = x @ weight + residual
        norm, rstd = aten._fused_rms_norm.default(
            hidden,
            [hidden.shape[-1]],
            norm_weight,
            EPS,
        )
        return hidden, norm, rstd


class _SharedResidualNorm(torch.nn.Module):
    def forward(self, x, weight, routed, residual, norm_weight):
        hidden = x @ weight + routed + residual
        norm, rstd = aten._fused_rms_norm.default(
            hidden,
            [hidden.shape[-1]],
            norm_weight,
            EPS,
        )
        return hidden, norm, rstd


class _WeightedResidualNorm(torch.nn.Module):
    def forward(self, probabilities, values, norm_weight):
        hidden = torch.bmm(probabilities, values).squeeze(1).to(torch.bfloat16)
        norm, rstd = aten._fused_rms_norm.default(
            hidden,
            [hidden.shape[-1]],
            norm_weight,
            EPS,
        )
        return hidden, norm, rstd


class _SwiGLU(torch.nn.Module):
    def forward(self, x, gate_weight, up_weight):
        return torch.nn.functional.silu(x @ gate_weight) * (x @ up_weight)


class _SiTU(torch.nn.Module):
    def forward(self, x, gate_weight, up_weight):
        gate = (x @ gate_weight).float()
        up = (x @ up_weight).float()
        activated_gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
        transformed_up = 25.0 * torch.tanh(up / 25.0)
        return (activated_gate * transformed_up).bfloat16()


class _SigmoidEpilogue(torch.nn.Module):
    def __init__(self, *, add: bool) -> None:
        super().__init__()
        self.add = add

    def forward(self, x, weight, auxiliary):
        gate = torch.sigmoid(x @ weight)
        if self.add:
            return gate, gate + auxiliary
        return gate, gate * auxiliary


class _ProjectionNormBackward(torch.nn.Module):
    def forward(self, grad, projection_weight, norm_input, rstd, norm_weight):
        projected = grad @ projection_weight
        return aten._fused_rms_norm_backward.default(
            projected,
            norm_input,
            [norm_input.shape[-1]],
            rstd,
            norm_weight,
            [True, True],
        )


class _BackwardCast(torch.nn.Module):
    def __init__(self, *, reshape_output: bool) -> None:
        super().__init__()
        self.reshape_output = reshape_output

    def forward(self, lhs, rhs):
        output = lhs @ rhs
        if self.reshape_output:
            output = output.reshape(1, output.shape[0], output.shape[1])
        return output.float()


class _BackwardMerge(torch.nn.Module):
    def forward(self, left, left_weight, right, right_weight):
        return left @ left_weight + right @ right_weight


class _BackwardAccumulate(torch.nn.Module):
    def forward(self, x, weight, accumulated):
        return x @ weight + accumulated


class _SwiGLUBackwardActivation(torch.nn.Module):
    def forward(self, grad, down_weight, saved_silu, saved_gate, saved_w1):
        branch_grad = grad @ down_weight
        gate_grad = branch_grad * saved_silu
        silu_grad = aten.silu_backward.default(branch_grad * saved_gate, saved_w1)
        return gate_grad, silu_grad


class _SiTUBackwardActivation(torch.nn.Module):
    def forward(
        self,
        grad,
        down_weight,
        saved_tanh,
        saved_sigmoid,
    ):
        branch_grad = (grad @ down_weight).float()
        up_grad = aten.tanh_backward.default(branch_grad, saved_tanh).bfloat16()
        gate_grad = aten.sigmoid_backward.default(branch_grad, saved_sigmoid).bfloat16()
        shape = [branch_grad.shape[0], branch_grad.shape[1] * 2]
        end = torch.iinfo(torch.int64).max
        joined = aten.slice_backward.default(
            up_grad, shape, 1, branch_grad.shape[1], end, 1
        ) + aten.slice_backward.default(gate_grad, shape, 1, 0, branch_grad.shape[1], 1)
        gate = joined[:, : branch_grad.shape[1]]
        up = joined[:, branch_grad.shape[1] :]
        return gate, up


class _MlaOutputGateBackward(torch.nn.Module):
    def forward(self, grad, output_weight, attention, sigmoid_gate):
        gated_grad = grad @ output_weight
        attention_grad = gated_grad * sigmoid_gate
        gate_grad = aten.sigmoid_backward.default(
            gated_grad * attention,
            aten.alias.default(sigmoid_gate),
        )
        return attention_grad, gate_grad


def _projection_norm_case(
    shape: ModelShape,
    device: torch.device,
    *,
    q_projection: bool,
) -> BenchmarkCase:
    if q_projection:
        rank = shape.q_lora_rank
        first_width = rank
        output_width = shape.q_output_width
        prefix = "layers.0.attention.wq"
    else:
        rank = shape.kv_lora_rank
        first_width = rank + shape.kv_rope_width
        output_width = shape.kv_output_width
        prefix = "layers.0.attention.wkv"
    return BenchmarkCase(
        _ProjectionNorm(split_projection=not q_projection),
        (
            _bf16((shape.tokens, shape.model_width), device),
            _bf16((shape.model_width, first_width), device),
            torch.ones(rank, device=device, dtype=torch.bfloat16),
            _bf16((rank, output_width), device),
        ),
        (f"{prefix}_a", f"{prefix}_b"),
    )


def _residual_norm_case(
    shape: ModelShape,
    device: torch.device,
    *,
    input_width: int,
    module_fqn: str,
) -> BenchmarkCase:
    return BenchmarkCase(
        _ResidualNorm(),
        (
            _bf16((shape.tokens, input_width), device),
            _bf16((input_width, shape.model_width), device),
            _bf16((shape.tokens, shape.model_width), device),
            torch.ones(shape.model_width, device=device, dtype=torch.bfloat16),
        ),
        (module_fqn,),
    )


def _activation_case(
    shape: ModelShape,
    device: torch.device,
    *,
    hidden_width: int,
    shared: bool,
    situ: bool,
) -> BenchmarkCase:
    prefix = "layers.0.moe.shared_experts" if shared else "layers.0.feed_forward"
    return BenchmarkCase(
        _SiTU() if situ else _SwiGLU(),
        (
            _bf16((shape.tokens, shape.model_width), device),
            _bf16((shape.model_width, hidden_width), device),
            _bf16((shape.model_width, hidden_width), device),
        ),
        (f"{prefix}.w1", f"{prefix}.w3"),
    )


def _backward_activation_case(
    shape: ModelShape,
    device: torch.device,
    *,
    situ: bool,
) -> BenchmarkCase:
    width = shape.shared_expert_width
    if situ:
        return BenchmarkCase(
            _SiTUBackwardActivation(),
            (
                _bf16((shape.tokens, shape.model_width), device),
                _bf16((shape.model_width, width), device),
                torch.tanh(_fp32((shape.tokens, width), device)),
                torch.sigmoid(_fp32((shape.tokens, width), device)),
            ),
            ("layers.0.mlp.down_proj",),
            backward=True,
        )
    saved_w1 = _bf16((shape.tokens, width), device)
    return BenchmarkCase(
        _SwiGLUBackwardActivation(),
        (
            _bf16((shape.tokens, shape.model_width), device),
            _bf16((shape.model_width, width), device),
            torch.nn.functional.silu(saved_w1),
            _bf16((shape.tokens, width), device),
            saved_w1,
        ),
        ("layers.0.feed_forward.w2",),
        backward=True,
    )


def _backward_merge_case(
    shape: ModelShape,
    device: torch.device,
    *,
    situ: bool,
) -> BenchmarkCase:
    width = shape.shared_expert_width
    prefix = "layers.0.mlp" if situ else "layers.0.feed_forward"
    upstream = (
        "B_k3_situ_backward_activation" if situ else "B_swiglu_backward_activation"
    )
    return BenchmarkCase(
        _BackwardMerge(),
        (
            _bf16((shape.tokens, width), device),
            _bf16((width, shape.model_width), device),
            _bf16((shape.tokens, width), device),
            _bf16((width, shape.model_width), device),
        ),
        (f"{prefix}.gate_proj", f"{prefix}.up_proj"),
        (upstream, None),
        backward=True,
    )


def _backward_rmsnorm_case(
    shape: ModelShape,
    device: torch.device,
    *,
    output_width: int,
    norm_width: int,
    module_fqn: str,
) -> BenchmarkCase:
    return BenchmarkCase(
        _ProjectionNormBackward(),
        (
            _bf16((shape.tokens, output_width), device),
            _bf16((output_width, norm_width), device),
            _bf16((shape.tokens, norm_width), device),
            torch.empty((shape.tokens, 1), device=device, dtype=torch.float32).uniform_(
                0.5, 1.5
            ),
            _bf16((norm_width,), device),
        ),
        (module_fqn,),
        backward=True,
    )


@register_microbenchmark(
    formula="[M,D]@[D,Rq] -> RMSNorm(Rq) -> [M,Rq]@[Rq,Oq]",
)
def F_mla_qproj_rmsnorm_expand(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _projection_norm_case(shape, device, q_projection=True)


@register_microbenchmark(
    formula="[M,D]@[D,Rkv+Rrope] -> split -> RMSNorm(Rkv) -> [M,Rkv]@[Rkv,Okv]",
)
def F_mla_kvproj_rmsnorm_expand(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _projection_norm_case(shape, device, q_projection=False)


@register_microbenchmark(
    formula="[M,1,R]@[M,R,D] -> squeeze -> BF16 -> RMSNorm(D)",
    families=("kimi",),
    atol=2e-2,
    rtol=2e-2,
)
def F_weighted_residual_bmm_prenorm(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    probabilities = torch.softmax(
        _fp32((shape.tokens, 1, shape.attn_res_width), device), dim=-1
    )
    return BenchmarkCase(
        _WeightedResidualNorm(),
        (
            probabilities,
            _fp32((shape.tokens, shape.attn_res_width, shape.model_width), device),
            torch.ones(shape.model_width, device=device, dtype=torch.bfloat16),
        ),
    )


@register_microbenchmark(
    formula="[M,A]@[A,D] + residual -> RMSNorm(D)", families=("dsv3",)
)
def F_attnout_residual_ffn_prenorm(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _residual_norm_case(
        shape,
        device,
        input_width=shape.attention_width,
        module_fqn="layers.0.attention.wo",
    )


@register_microbenchmark(
    formula="[M,Hshared]@[Hshared,D] + routed + residual -> RMSNorm(D)",
    families=("dsv3",),
)
def F_sharedE_out_residual_attn_prenorm(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return BenchmarkCase(
        _SharedResidualNorm(),
        (
            _bf16((shape.tokens, shape.shared_expert_width), device),
            _bf16((shape.shared_expert_width, shape.model_width), device),
            _bf16((shape.tokens, shape.model_width), device),
            _bf16((shape.tokens, shape.model_width), device),
            torch.ones(shape.model_width, device=device, dtype=torch.bfloat16),
        ),
        ("layers.0.moe.shared_experts.w2",),
    )


@register_microbenchmark(
    formula="[M,Hdense]@[Hdense,D] + residual -> RMSNorm(D)",
    families=("dsv3",),
)
def F_dense_ffnout_residual_attn_prenorm(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _residual_norm_case(
        shape,
        device,
        input_width=shape.dense_expert_width,
        module_fqn="layers.0.feed_forward.w2",
    )


@register_microbenchmark(
    formula="X -> {GEMM gate, GEMM up} -> silu(gate) * up",
    families=("dsv3",),
)
def F_sharedE_swiglu(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _activation_case(
        shape,
        device,
        hidden_width=shape.shared_expert_width,
        shared=True,
        situ=False,
    )


@register_microbenchmark(
    formula="X -> {GEMM gate, GEMM up} -> silu(gate) * up",
    families=("dsv3",),
)
def F_dense_ffn_swiglu(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _activation_case(
        shape,
        device,
        hidden_width=shape.dense_expert_width,
        shared=False,
        situ=False,
    )


@register_microbenchmark(
    formula="X -> {GEMM gate, GEMM up} -> situ(gate) * transform(up)",
    families=("kimi",),
)
def F_k3_sharedE_situ(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _activation_case(
        shape,
        device,
        hidden_width=shape.shared_expert_width,
        shared=True,
        situ=True,
    )


@register_microbenchmark(
    formula="X -> {GEMM gate, GEMM up} -> situ(gate) * transform(up)",
    families=("kimi",),
)
def F_k3_dense_ffn_situ(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _activation_case(
        shape,
        device,
        hidden_width=shape.dense_expert_width,
        shared=False,
        situ=True,
    )


@register_microbenchmark(
    formula="[M,D]@[D,A] -> sigmoid(gate) * attention",
    families=("kimi",),
)
def F_k3_mla_output_gate(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return BenchmarkCase(
        _SigmoidEpilogue(add=False),
        (
            _bf16((shape.tokens, shape.model_width), device),
            _bf16((shape.model_width, shape.attention_width), device),
            _bf16((shape.tokens, shape.attention_width), device),
        ),
        ("layers.0.self_attn.g_proj",),
    )


@register_microbenchmark(
    formula="[M,D]@[D,E] -> sigmoid(scores) -> scores + expert_bias",
)
def F_router_sigmoid_bias(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return BenchmarkCase(
        _SigmoidEpilogue(add=True),
        (
            _bf16((shape.tokens, shape.model_width), device),
            _bf16((shape.model_width, shape.num_experts), device),
            _bf16((1, shape.num_experts), device),
        ),
        ("layers.0.moe.router.gate",),
    )


@register_microbenchmark(
    formula="[Mchunk,V]@[V,D] -> reshape -> BF16-to-FP32 cast",
    direction="backward",
    atol=5e-4,
)
def B_lmhead_dx_bf16_to_fp32(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    chunk_tokens = max(1, shape.tokens // shape.loss_chunks)
    return BenchmarkCase(
        _BackwardCast(reshape_output=True),
        (
            _bf16((chunk_tokens, shape.vocab_size), device),
            _bf16((shape.vocab_size, shape.model_width), device),
        ),
        ("model.lm_head",),
        backward=True,
    )


@register_microbenchmark(
    formula="W2 dX GEMM -> SwiGLU branch gradients",
    direction="backward",
    families=("dsv3",),
)
def B_swiglu_backward_activation(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _backward_activation_case(shape, device, situ=False)


@register_microbenchmark(
    formula="{gate dX GEMM, up dX GEMM} -> add",
    direction="backward",
    families=("dsv3",),
)
def B_swiglu_dx_merge(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _backward_merge_case(shape, device, situ=False)


@register_microbenchmark(
    formula="output-projection dX GEMM -> attention and gate gradients",
    direction="backward",
    families=("kimi",),
)
def B_k3_mla_output_gate(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return BenchmarkCase(
        _MlaOutputGateBackward(),
        (
            _bf16((shape.tokens, shape.model_width), device),
            _bf16((shape.model_width, shape.attention_width), device),
            _bf16((shape.tokens, shape.attention_width), device),
            torch.sigmoid(_bf16((shape.tokens, shape.attention_width), device)),
        ),
        ("layers.0.self_attn.o_proj",),
        backward=True,
    )


@register_microbenchmark(
    formula="down-projection dX GEMM -> SiTU branch gradients",
    direction="backward",
    families=("kimi",),
)
def B_k3_situ_backward_activation(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _backward_activation_case(shape, device, situ=True)


@register_microbenchmark(
    formula="{gate dX GEMM, up dX GEMM} -> add",
    direction="backward",
    families=("kimi",),
)
def B_k3_situ_dx_merge(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _backward_merge_case(shape, device, situ=True)


@register_microbenchmark(
    formula="router dX GEMM + expert input gradient",
    direction="backward",
)
def B_router_dx_cast_expert_merge(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return BenchmarkCase(
        _BackwardAccumulate(),
        (
            _bf16((shape.tokens, shape.num_experts), device),
            _bf16((shape.num_experts, shape.model_width), device),
            _bf16((shape.tokens, shape.model_width), device),
        ),
        ("layers.0.moe.router.gate",),
        backward=True,
    )


@register_microbenchmark(
    formula="routed-up dX GEMM -> routed RMSNorm backward",
    direction="backward",
    families=("kimi",),
)
def B_routed_up_dx_rmsnorm(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _backward_rmsnorm_case(
        shape,
        device,
        output_width=shape.model_width,
        norm_width=shape.routed_latent_width,
        module_fqn="layers.0.moe.routed_up",
    )


@register_microbenchmark(
    formula="[M,Oq]@[Oq,Rq] -> RMSNorm backward",
    direction="backward",
)
def B_mla_qproj_dx_rmsnorm(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _backward_rmsnorm_case(
        shape,
        device,
        output_width=shape.q_output_width,
        norm_width=shape.q_lora_rank,
        module_fqn="layers.0.attention.wq_b",
    )


@register_microbenchmark(
    formula="[M,Okv]@[Okv,Rkv] -> RMSNorm backward",
    direction="backward",
)
def B_mla_kvproj_dx_rmsnorm(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _backward_rmsnorm_case(
        shape,
        device,
        output_width=shape.kv_output_width,
        norm_width=shape.kv_lora_rank,
        module_fqn="layers.0.attention.wkv_b",
    )


@register_microbenchmark(
    formula="dY.T [H,M] @ X [M,D] -> BF16-to-FP32 cast",
    direction="backward",
    atol=5e-4,
)
def B_linear_dw_bf16_to_fp32(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return BenchmarkCase(
        _BackwardCast(reshape_output=False),
        (
            _bf16((shape.shared_expert_width, shape.tokens), device),
            _bf16((shape.tokens, shape.model_width), device),
        ),
        ("layers.0.moe.shared_experts.w1",),
        backward=True,
    )


@register_microbenchmark(
    formula="KV dX GEMM + Q dX GEMM -> attention input gradient",
    direction="backward",
)
def B_mla_qkv_dx_merge(  # noqa: N802
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    q_input_width = shape.q_lora_rank or shape.q_output_width
    kv_input_width = shape.kv_lora_rank + shape.kv_rope_width
    return BenchmarkCase(
        _BackwardMerge(),
        (
            _bf16((shape.tokens, kv_input_width), device),
            _bf16((kv_input_width, shape.model_width), device),
            _bf16((shape.tokens, q_input_width), device),
            _bf16((q_input_width, shape.model_width), device),
        ),
        ("layers.0.attention.wkv_a", "layers.0.attention.wq_a"),
        backward=True,
    )


def _generic_parallel_mm_dx_merge_case(
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    return _backward_merge_case(shape, device, situ=shape.family == "kimi")


def _generic_mm_dx_rmsnorm_case(
    shape: ModelShape, device: torch.device
) -> BenchmarkCase:
    if shape.family == "kimi":
        return B_routed_up_dx_rmsnorm(shape, device)
    if shape.q_lora_rank:
        return B_mla_qproj_dx_rmsnorm(shape, device)
    return B_mla_kvproj_dx_rmsnorm(shape, device)


_STRUCTURAL_BENCHMARK_VARIANTS = {
    "F_mm_residual_rmsnorm": (
        "F_attnout_residual_ffn_prenorm",
        "F_sharedE_out_residual_attn_prenorm",
        "F_dense_ffnout_residual_attn_prenorm",
    ),
    "F_swiglu": ("F_sharedE_swiglu", "F_dense_ffn_swiglu"),
    "F_situ": ("F_k3_sharedE_situ", "F_k3_dense_ffn_situ"),
    "B_reshape_bf16_to_fp32": ("B_lmhead_dx_bf16_to_fp32",),
    "B_parallel_mm_dx_merge": (
        "B_swiglu_dx_merge",
        "B_k3_situ_dx_merge",
        "B_mla_qkv_dx_merge",
    ),
    "B_mm_dx_residual_add": ("B_router_dx_cast_expert_merge",),
    "B_mm_dx_rmsnorm": (
        "B_routed_up_dx_rmsnorm",
        "B_mla_qproj_dx_rmsnorm",
        "B_mla_kvproj_dx_rmsnorm",
    ),
}
_STRUCTURAL_BENCHMARK_FACTORIES = {
    "B_parallel_mm_dx_merge": _generic_parallel_mm_dx_merge_case,
    "B_mm_dx_rmsnorm": _generic_mm_dx_rmsnorm_case,
}
for name, variants in _STRUCTURAL_BENCHMARK_VARIANTS.items():
    families = frozenset(
        family for variant in variants for family in MICROBENCHMARKS[variant].families
    )
    MICROBENCHMARKS[name] = replace(
        MICROBENCHMARKS[variants[0]],
        name=name,
        factory=_STRUCTURAL_BENCHMARK_FACTORIES.get(
            name, MICROBENCHMARKS[variants[0]].factory
        ),
        families=families,
    )


def _validate_registry() -> None:
    registered = set(CODA_PATTERN_NAMES)
    benchmarked = set(MICROBENCHMARKS)
    missing = sorted(registered - benchmarked)
    legacy = {
        variant
        for variants in _STRUCTURAL_BENCHMARK_VARIANTS.values()
        for variant in variants
    }
    extra = sorted(benchmarked - registered - legacy)
    if missing or extra:
        raise RuntimeError(
            f"CODA microbenchmark registry is stale; missing={missing}, extra={extra}"
        )


_validate_registry()


def _matmuls(gm: GraphModule) -> list[torch.fx.Node]:
    return [node for node in gm.graph.nodes if node.target is aten.mm.default]


def _trace_case(case: BenchmarkCase, *, kimi: bool) -> GraphModule:
    gm = make_fx(case.module)(*case.inputs)
    for node in gm.graph.nodes:
        if case.backward:
            node.meta["autograd_backward"] = True
        if kimi:
            node.meta.setdefault("custom", {})["coda_model"] = "kimi_k3"
    matmuls = _matmuls(gm)
    if len(matmuls) != len(case.module_fqns):
        raise AssertionError(
            f"expected {len(case.module_fqns)} matmuls, found {len(matmuls)}"
        )
    for node, module_fqn in zip(matmuls, case.module_fqns, strict=True):
        node.meta.setdefault("custom", {})["module_fqn"] = module_fqn
    if case.upstream_patterns:
        if len(matmuls) != len(case.upstream_patterns):
            raise AssertionError(
                f"expected {len(case.upstream_patterns)} upstream markers, "
                f"found {len(matmuls)} matmuls"
            )
        for node, pattern in zip(matmuls, case.upstream_patterns, strict=True):
            if pattern is not None:
                node.meta.setdefault("custom", {})["coda_pattern"] = pattern
    return gm


def _flatten_outputs(value: TensorTree) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    return [tensor for item in value for tensor in _flatten_outputs(item)]


def _check_outputs(
    actual: TensorTree,
    expected: TensorTree,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    actual_tensors = _flatten_outputs(actual)
    expected_tensors = _flatten_outputs(expected)
    max_abs = 0.0
    relative_l2 = 0.0
    for actual_tensor, expected_tensor in zip(
        actual_tensors, expected_tensors, strict=True
    ):
        difference = (actual_tensor.float() - expected_tensor.float()).norm()
        denominator = expected_tensor.float().norm().clamp_min(1e-12)
        max_abs = max(
            max_abs,
            (actual_tensor.float() - expected_tensor.float()).abs().max().item(),
        )
        relative_l2 = max(relative_l2, (difference / denominator).item())
    return max_abs, relative_l2


def _selected_quack_configs(
    generated_sources: Sequence[str],
    cache_before: dict[Path, int] | None = None,
) -> tuple[dict[str, Any], ...]:
    source = "\n".join(generated_sources)
    matches = re.findall(
        r"config_key=(\(.*?\)), config_is_lowering_validated",
        source,
    )
    if not matches:
        matches = re.findall(
            r"quack_config_key=(\(.*?\)), epilogue_arg_indices",
            source,
        )
    if matches:
        return tuple(dict(ast.literal_eval(config_key)) for config_key in matches)
    if cache_before is None:
        return ()
    cache_root = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) / "quack"
    changed = [
        path
        for path in cache_root.rglob("*.autotune.json")
        if cache_before.get(path) != path.stat().st_mtime_ns
    ]
    identifiers = re.findall(
        r"flex_gemm_runtime\([^\n]*?flex_gemm_epimod_([0-9a-f]+)", source
    )
    result = []
    for identifier in identifiers:
        candidates = sorted(
            (
                path
                for path in changed
                if path.name.startswith(f"mod_flex_gemm_epimod_{identifier}_")
            ),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not candidates:
            candidates = sorted(
                cache_root.rglob(f"mod_flex_gemm_epimod_{identifier}_*.autotune.json"),
                key=lambda path: path.stat().st_mtime_ns,
            )
        if not candidates:
            continue
        selected_path = candidates[0]
        record = json.loads(selected_path.read_text(encoding="utf-8"))
        best = min(record["configs_timings"], key=lambda item: min(item[1]))
        config = re.fullmatch(r"config: GemmConfig\((.*)\)", best[0])
        if config is None:
            continue
        result.append(
            {
                name: ast.literal_eval(value)
                for name, value in (
                    field.split("=", 1) for field in config.group(1).split(", ")
                )
            }
        )
        if selected_path in changed:
            changed.remove(selected_path)
    return tuple(result)


def _quack_autotune_cache_state() -> dict[Path, int]:
    cache_root = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) / "quack"
    if not cache_root.is_dir():
        return {}
    return {
        path: path.stat().st_mtime_ns
        for path in cache_root.rglob("*.autotune.json")
    }


def _flex_gemm_nodes(gm: GraphModule) -> list[torch.fx.Node]:
    return [node for node in gm.graph.nodes if node.target is flex_gemm_hop]


def _flex_gemm_options(node: torch.fx.Node) -> dict[str, Any]:
    options = node.args[4]
    if not isinstance(options, dict):
        raise TypeError(f"expected FlexGEMM options dict, got {type(options)}")
    return cast(dict[str, Any], options)


def _quack_config(
    tile_m: int,
    tile_n: int,
    *,
    dynamic: bool,
    cluster_m: int,
    cluster_n: int,
    swap_ab: bool,
) -> dict[str, Any]:
    device_capacity = torch.cuda.get_device_capability()[0]
    if device_capacity == 11:
        device_capacity = 10
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": None,
        "num_warps": None,
        "pingpong": False,
        "is_dynamic_persistent": dynamic,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "cluster_k": 1,
        "swap_ab": swap_ab,
        "max_swizzle_size": 8,
        "device_capacity": device_capacity,
        "use_tma_gather": False,
    }


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    configs: tuple[dict[str, Any], ...] | None = None
    autotune: bool = False


def _configure_candidate(gm: GraphModule, candidate: CandidateConfig) -> list[dict]:
    quack_nodes = [
        node
        for node in _flex_gemm_nodes(gm)
        if _flex_gemm_options(node).get("backend") == "QUACK"
    ]
    if candidate.configs is not None and not quack_nodes:
        raise ValueError("explicit QuACK config requested for a non-QuACK pattern")
    configs = candidate.configs
    if configs is not None:
        if len(configs) == 1:
            configs = configs * len(quack_nodes)
        elif len(configs) != len(quack_nodes):
            raise ValueError(
                f"candidate has {len(configs)} configs for "
                f"{len(quack_nodes)} QuACK FlexGEMMs"
            )
    for index, node in enumerate(quack_nodes):
        options: dict[str, Any] = dict(_flex_gemm_options(node))
        if os.environ.get("TORCHTITAN_CODA_DISABLE_TUNE_SPLIT_K") == "1":
            options.pop("tune_split_k", None)
        if candidate.autotune:
            options.pop("config", None)
            options["tuned"] = True
        elif configs is not None:
            options["config"] = dict(configs[index])
            options["tuned"] = False
        node.args = (*node.args[:4], options)
    gm.recompile()
    return [dict(_flex_gemm_options(node)) for node in _flex_gemm_nodes(gm)]


def _candidate_configs(args: argparse.Namespace) -> list[CandidateConfig]:
    candidates = (
        []
        if getattr(args, "only_explicit_configs", False)
        else [CandidateConfig("current")]
    )
    if args.also_autotune:
        candidates.append(CandidateConfig("autotune", autotune=True))
    for grid in args.grid:
        tile_m, tile_n = grid
        config = _quack_config(
            tile_m,
            tile_n,
            dynamic=args.dynamic_persistent,
            cluster_m=args.cluster_m,
            cluster_n=args.cluster_n,
            swap_ab=args.swap_ab,
        )
        candidates.append(
            CandidateConfig(
                (
                    f"grid-{tile_m}x{tile_n}-"
                    f"{'dynamic' if args.dynamic_persistent else 'static'}-"
                    f"cluster-{args.cluster_m}x{args.cluster_n}"
                    f"{'-swap-ab' if args.swap_ab else ''}"
                ),
                configs=(config,),
            )
        )
    for index, configs in enumerate(args.config, start=1):
        complete = []
        for config in configs:
            merged = _quack_config(
                128,
                128,
                dynamic=True,
                cluster_m=2,
                cluster_n=1,
                swap_ab=False,
            )
            merged.update(config)
            complete.append(merged)
        candidates.append(CandidateConfig(f"config-{index}", configs=tuple(complete)))
    if not candidates:
        raise ValueError(
            "--only-explicit-configs requires at least one --grid, --config, "
            "or --also-autotune candidate"
        )
    return candidates


def _pattern_applies(pattern: Microbenchmark, shape: ModelShape) -> tuple[bool, str]:
    if shape.family not in pattern.families:
        return (
            False,
            f"pattern applies to {sorted(pattern.families)}, not {shape.family}",
        )
    if pattern.name == "F_mla_qproj_rmsnorm_expand" and shape.q_lora_rank == 0:
        return False, "preset has a direct Q projection (q_lora_rank=0)"
    if pattern.name == "F_router_sigmoid_bias" and shape.router_activation != "sigmoid":
        return False, f"preset uses a {shape.router_activation} router"
    return True, ""


def _run_candidate(
    rewritten: GraphModule,
    inputs: tuple[torch.Tensor, ...],
    expected: TensorTree,
    pattern: Microbenchmark,
    candidate: CandidateConfig,
    *,
    compile_mode: str,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, Any]:
    candidate_gm = copy.deepcopy(rewritten)
    options = _configure_candidate(candidate_gm, candidate)
    compiled = None
    actual = None
    try:
        cache_before = _quack_autotune_cache_state() if candidate.autotune else None
        compile_start = time.perf_counter()
        compiled = torch.compile(
            candidate_gm,
            backend="inductor",
            fullgraph=True,
            mode=compile_mode,
        )
        actual, generated_sources = run_and_get_code(compiled, *inputs)
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - compile_start
        max_abs, relative_l2 = _check_outputs(
            actual,
            expected,
            atol=pattern.atol,
            rtol=pattern.rtol,
        )
        runtime_us = 1000.0 * cast(
            float,
            do_bench(
                lambda: compiled(*inputs),
                warmup=warmup_ms,
                rep=rep_ms,
                return_mode="median",
            ),
        )
        return {
            "name": candidate.name,
            "runtime_us": runtime_us,
            "compile_seconds": compile_seconds,
            "flex_gemm_options": options,
            "selected_quack_configs": list(
                _selected_quack_configs(generated_sources, cache_before)
            ),
            "correctness": {
                "max_abs": max_abs,
                "relative_l2": relative_l2,
                "atol": pattern.atol,
                "rtol": pattern.rtol,
            },
        }
    finally:
        del compiled, candidate_gm, actual
        torch.cuda.empty_cache()


def run_pattern(
    pattern: Microbenchmark,
    shape: ModelShape,
    candidates: Sequence[CandidateConfig],
    *,
    device: torch.device,
    compile_mode: str,
    warmup_ms: int,
    rep_ms: int,
    current_autotune: bool,
    keep_going: bool,
) -> dict[str, Any]:
    applies, reason = _pattern_applies(pattern, shape)
    if not applies:
        print(f"SKIP {shape.name}: {pattern.name}: {reason}")
        return {"pattern": pattern.name, "status": "skipped", "reason": reason}

    print(f"\n{shape.name}: {pattern.name}")
    print(f"  {pattern.formula}")
    torch.manual_seed(0)
    case = pattern.factory(shape, device)
    source = _trace_case(case, kimi=shape.family == "kimi")
    expected = source(*case.inputs)
    rewritten = coda_flex_gemm_pass(
        source,
        case.inputs,
        patterns=[pattern.name],
        compile_time_benchmark=False,
        coda_autotune=current_autotune,
    )
    count = rewritten.meta["coda_pattern_counts"][pattern.name]
    if count != 1:
        raise RuntimeError(f"expected one {pattern.name} match, found {count}")
    flex_gemm_count = len(_flex_gemm_nodes(rewritten))
    has_quack = any(
        _flex_gemm_options(node).get("backend") == "QUACK"
        for node in _flex_gemm_nodes(rewritten)
    )

    eager_us = 1000.0 * cast(
        float,
        do_bench(
            lambda: source(*case.inputs),
            warmup=warmup_ms,
            rep=rep_ms,
            return_mode="median",
        ),
    )
    print(f"  eager: {eager_us:.1f} us")
    result: dict[str, Any] = {
        "pattern": pattern.name,
        "direction": pattern.direction,
        "formula": pattern.formula,
        "status": "ok",
        "eager_us": eager_us,
        "flex_gemm_count": flex_gemm_count,
        "candidates": [],
    }
    for candidate in candidates:
        if (candidate.configs is not None or candidate.autotune) and not has_quack:
            candidate_result = {
                "name": candidate.name,
                "status": "skipped",
                "reason": "pattern uses the Triton FlexGEMM backend, not QuACK",
            }
            print(f"  {candidate.name}: SKIP: {candidate_result['reason']}")
            result["candidates"].append(candidate_result)
            continue
        try:
            candidate_result = _run_candidate(
                rewritten,
                case.inputs,
                expected,
                pattern,
                candidate,
                compile_mode=compile_mode,
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
            )
        except Exception as error:
            if not keep_going:
                raise
            candidate_result = {
                "name": candidate.name,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"  {candidate.name}: FAILED: {candidate_result['error']}")
        else:
            speedup = eager_us / candidate_result["runtime_us"]
            candidate_result["status"] = "ok"
            candidate_result["speedup_vs_eager"] = speedup
            print(
                f"  {candidate.name}: {candidate_result['runtime_us']:.1f} us, "
                f"{speedup:.3f}x vs eager"
            )
        result["candidates"].append(candidate_result)

    successful = [
        candidate for candidate in result["candidates"] if candidate["status"] == "ok"
    ]
    if successful:
        best = min(successful, key=lambda candidate: candidate["runtime_us"])
        result["best_candidate"] = best["name"]
        print(
            f"  best: {best['name']} at {best['runtime_us']:.1f} us "
            f"({best['speedup_vs_eager']:.3f}x vs eager)"
        )

    del case, expected, rewritten, source
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return result


def _parse_grid(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)[xX](\d+)", value)
    if match is None:
        raise argparse.ArgumentTypeError("grid must have the form TILE_MxTILE_N")
    tile_m, tile_n = (int(item) for item in match.groups())
    if tile_m <= 0 or tile_n <= 0:
        raise argparse.ArgumentTypeError("grid dimensions must be positive")
    return tile_m, tile_n


def _parse_config(value: str) -> tuple[dict[str, Any], ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON config: {error}") from error
    if isinstance(parsed, dict):
        return (parsed,)
    if (
        isinstance(parsed, list)
        and parsed
        and all(isinstance(item, dict) for item in parsed)
    ):
        return tuple(parsed)
    raise argparse.ArgumentTypeError(
        "config must be a JSON object or a non-empty list of JSON objects"
    )


def _parse_override(value: str) -> tuple[str, int]:
    name, separator, raw_value = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(
            "shape override must have the form FIELD=VALUE"
        )
    int_fields = {
        field.name
        for field in fields(ModelShape)
        if field.name not in {"name", "family", "router_activation"}
    }
    if name not in int_fields:
        raise argparse.ArgumentTypeError(
            f"unknown integer ModelShape field {name!r}; choices: {sorted(int_fields)}"
        )
    try:
        parsed = int(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer {raw_value!r}") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("shape values must be nonnegative")
    return name, parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark current CODA pass registrations against eager ATen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=("all", *PRESETS),
        help="Model shape to run; repeat for several presets. Default: dsv3-16b.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        help=(
            "Canonical CODA pattern, or all/forward/backward; repeat for several. "
            "Default: all."
        ),
    )
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--list-patterns", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "max-autotune", "max-autotune-no-cudagraphs"),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument(
        "--current-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use normal CODA autotuning for registrations without a pinned config.",
    )
    parser.add_argument(
        "--also-autotune",
        action="store_true",
        help="Also benchmark an unpinned QuACK autotuning candidate.",
    )
    parser.add_argument(
        "--only-explicit-configs",
        action="store_true",
        help="Skip the current registration and benchmark only explicit candidates.",
    )
    parser.add_argument(
        "--grid",
        action="append",
        type=_parse_grid,
        default=[],
        metavar="TILE_MxTILE_N",
        help="Benchmark a QuACK tile grid; repeat to sweep several grids.",
    )
    parser.add_argument(
        "--dynamic-persistent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dynamic persistent scheduling for --grid candidates.",
    )
    parser.add_argument("--cluster-m", type=int, default=2)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--swap-ab", action="store_true")
    parser.add_argument(
        "--config",
        action="append",
        type=_parse_config,
        default=[],
        help=(
            "Explicit QuACK config JSON. A single object applies to every HOP; "
            "a JSON list assigns one object per HOP."
        ),
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_override,
        default=[],
        metavar="FIELD=VALUE",
        help="Override an integer ModelShape field; repeat for several fields.",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record candidate failures and continue the sweep.",
    )
    return parser.parse_args()


def _select_presets(requested: list[str] | None) -> list[str]:
    if not requested:
        return ["dsv3-16b"]
    selected = []
    for item in requested:
        names = PRESETS if item == "all" else (item,)
        for name in names:
            if name not in selected:
                selected.append(name)
    return selected


def _select_patterns(requested: list[str] | None) -> list[str]:
    if not requested:
        return list(CODA_PATTERN_NAMES)
    selected = []
    for item in requested:
        if item == "all":
            names = CODA_PATTERN_NAMES
        elif item in {"forward", "backward"}:
            names = tuple(
                name
                for name in CODA_PATTERN_NAMES
                if MICROBENCHMARKS[name].direction == item
            )
        else:
            if item not in CODA_PATTERN_NAMES:
                raise ValueError(
                    f"unknown pattern {item!r}; choices: {sorted(CODA_PATTERN_NAMES)}"
                )
            names = (item,)
        for name in names:
            if name not in selected:
                selected.append(name)
    return selected


def _shape_with_overrides(
    shape: ModelShape, overrides: Sequence[tuple[str, int]]
) -> ModelShape:
    values: dict[str, int] = {}
    for name, value in overrides:
        values[name] = value
    return replace(shape, **values)


def _list_presets() -> None:
    for key, shape in PRESETS.items():
        print(
            f"{key:14} {shape.name:20} M={shape.tokens:<6} D={shape.model_width:<5} "
            f"Rq={shape.q_lora_rank:<4} Rkv={shape.kv_lora_rank:<4} "
            f"Hshared={shape.shared_expert_width:<5} Hdense={shape.dense_expert_width}"
        )


def _list_patterns() -> None:
    for name in CODA_PATTERN_NAMES:
        pattern = MICROBENCHMARKS[name]
        families = ",".join(sorted(pattern.families))
        print(f"{name} [{pattern.direction}; {families}]")
        print(f"  {pattern.formula}")


def main() -> None:
    args = _parse_args()
    if args.list_presets:
        _list_presets()
        return
    if args.list_patterns:
        _list_patterns()
        return
    if args.warmup_ms < 0 or args.rep_ms <= 0:
        raise ValueError("warmup-ms must be nonnegative and rep-ms must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CODA microbenchmarks require CUDA")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    candidates = _candidate_configs(args)
    preset_names = _select_presets(args.preset)
    pattern_names = _select_patterns(args.pattern)
    report: dict[str, Any] = {
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
        },
        "torch_version": torch.__version__,
        "compile_mode": args.compile_mode,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "runs": [],
    }
    for preset_name in preset_names:
        shape = _shape_with_overrides(PRESETS[preset_name], args.shape)
        preset_result = {
            "preset": preset_name,
            "shape": asdict(shape),
            "patterns": [],
        }
        print(f"\n{'=' * 80}\n{shape.name}: {asdict(shape)}\n{'=' * 80}")
        for pattern_name in pattern_names:
            pattern = MICROBENCHMARKS[pattern_name]
            try:
                pattern_result = run_pattern(
                    pattern,
                    shape,
                    candidates,
                    device=device,
                    compile_mode=args.compile_mode,
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                    current_autotune=args.current_autotune,
                    keep_going=args.keep_going,
                )
            except Exception as error:
                if not args.keep_going:
                    raise
                torch._dynamo.reset()
                torch.cuda.empty_cache()
                pattern_result = {
                    "pattern": pattern_name,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"FAILED {shape.name}: {pattern_name}: {pattern_result['error']}")
            preset_result["patterns"].append(pattern_result)
        report["runs"].append(preset_result)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {args.json_output}")


if __name__ == "__main__":
    main()
