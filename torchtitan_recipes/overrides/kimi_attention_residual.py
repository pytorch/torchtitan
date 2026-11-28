# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Fused Triton override for Kimi K3 attention residual aggregation.

The fused forward and backward algorithms are adapted from the MIT-licensed FLA
AttnRes operator introduced in
https://github.com/fla-org/flash-linear-attention/pull/878.
This implementation directly consumes TorchTitan's ``[T, N, D]`` residual bank
and optional ``[T, D]`` partial residual, avoiding a concatenation or layout
conversion.

Activate with::

    --override torchtitan_recipes.overrides.kimi_attention_residual.triton_attention_residual
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import triton
import triton.language as tl

from torchtitan.config import derive, override
from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.kimi_k3.model import AttentionResidual

__all__ = [
    "attention_residual_op",
    "TritonAttentionResidual",
    "triton_attention_residual",
]


@triton.jit
def _load_residual_tile(
    block_residual,
    partial_block,
    token,
    source,
    dim,
    num_sources,
    dim_mask,
    block_stride_t,
    block_stride_n,
    block_stride_d,
    partial_stride_t,
    partial_stride_d,
    HAS_PARTIAL: tl.constexpr,
):
    source_mask = source < num_sources
    values = tl.load(
        block_residual
        + token * block_stride_t
        + source[:, None] * block_stride_n
        + dim[None, :] * block_stride_d,
        mask=source_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if HAS_PARTIAL:
        partial_mask = source == num_sources
        values += tl.load(
            partial_block + token * partial_stride_t + dim[None, :] * partial_stride_d,
            mask=partial_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
    return values


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": block_n},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_n in [1, 2, 4, 8]
        for num_warps in [4, 8, 16]
        for num_stages in [2, 3]
    ],
    key=["SOURCE_BUCKET", "DIM", "HAS_PARTIAL"],
    cache_results=True,
)
@triton.jit(do_not_specialize=["NUM_SOURCES"])
def _attention_residual_forward_kernel(
    partial_block,
    block_residual,
    projection_weight,
    norm_weight,
    output,
    rstd,
    logits,
    lse,
    NUM_TOKENS,
    NUM_SOURCES,
    SOURCE_BUCKET: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    block_stride_t,
    block_stride_n,
    block_stride_d,
    partial_stride_t,
    partial_stride_d,
    HAS_PARTIAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    token = tl.program_id(0).to(tl.int64)
    total_sources = NUM_SOURCES + HAS_PARTIAL
    dim = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_D), BLOCK_D),
        BLOCK_D,
    )
    dim_mask = dim < DIM
    score_weight = tl.load(norm_weight + dim, mask=dim_mask, other=0.0).to(
        tl.float32
    ) * tl.load(projection_weight + dim, mask=dim_mask, other=0.0).to(tl.float32)

    max_score = tl.full([], float("-inf"), dtype=tl.float32)
    score_sum = tl.zeros([], dtype=tl.float32)
    mixed = tl.zeros([BLOCK_D], dtype=tl.float32)
    for source_block in range(tl.cdiv(total_sources, BLOCK_N)):
        source = source_block * BLOCK_N + tl.arange(0, BLOCK_N)
        source_mask = source < total_sources
        values = _load_residual_tile(
            block_residual,
            partial_block,
            token,
            source,
            dim,
            NUM_SOURCES,
            dim_mask,
            block_stride_t,
            block_stride_n,
            block_stride_d,
            partial_stride_t,
            partial_stride_d,
            HAS_PARTIAL,
        )
        source_rstd = tl.rsqrt(tl.sum(values * values, axis=1) / DIM + EPS)
        source_logits = tl.sum(values * score_weight[None, :], axis=1) * source_rstd
        source_scores = tl.where(source_mask, source_logits, float("-inf"))

        new_max_score = tl.maximum(max_score, tl.max(source_scores, axis=0))
        previous_scale = tl.exp(max_score - new_max_score)
        unnormalized = tl.exp(source_scores - new_max_score)
        score_sum = score_sum * previous_scale + tl.sum(unnormalized, axis=0)
        mixed = mixed * previous_scale + tl.sum(
            unnormalized[:, None] * values,
            axis=0,
        )
        max_score = new_max_score

        stats_offset = source * NUM_TOKENS + token
        tl.store(rstd + stats_offset, source_rstd, mask=source_mask)
        tl.store(logits + stats_offset, source_logits, mask=source_mask)

    tl.store(lse + token, max_score + tl.log(score_sum))
    tl.store(output + token * DIM + dim, mixed / score_sum, mask=dim_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": block_n},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_n in [1, 2, 4, 8]
        for num_warps in [4, 8, 16]
        for num_stages in [2, 3]
    ],
    key=["SOURCE_BUCKET", "DIM", "HAS_PARTIAL"],
    cache_results=True,
)
@triton.jit(do_not_specialize=["NUM_SOURCES"])
def _attention_residual_backward_values_kernel(
    grad_output,
    partial_block,
    block_residual,
    projection_weight,
    norm_weight,
    rstd,
    logits,
    lse,
    grad_partial,
    grad_block_residual,
    grad_score_weight,
    NUM_TOKENS,
    NUM_SOURCES,
    SOURCE_BUCKET: tl.constexpr,
    DIM: tl.constexpr,
    block_stride_t,
    block_stride_n,
    block_stride_d,
    partial_stride_t,
    partial_stride_d,
    grad_block_stride_t,
    grad_block_stride_n,
    grad_block_stride_d,
    grad_output_stride_t,
    grad_output_stride_d,
    grad_partial_stride_t,
    grad_partial_stride_d,
    HAS_PARTIAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    token = tl.program_id(0).to(tl.int64)
    total_sources = NUM_SOURCES + HAS_PARTIAL
    dim = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_D), BLOCK_D),
        BLOCK_D,
    )
    dim_mask = dim < DIM
    score_weight = tl.load(norm_weight + dim, mask=dim_mask, other=0.0).to(
        tl.float32
    ) * tl.load(projection_weight + dim, mask=dim_mask, other=0.0).to(tl.float32)
    logsumexp = tl.load(lse + token).to(tl.float32)
    grad = tl.load(
        grad_output + token * grad_output_stride_t + dim * grad_output_stride_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    mixed = tl.zeros([BLOCK_D], dtype=tl.float32)
    for source_block in range(tl.cdiv(total_sources, BLOCK_N)):
        source = source_block * BLOCK_N + tl.arange(0, BLOCK_N)
        source_mask = source < total_sources
        values = _load_residual_tile(
            block_residual,
            partial_block,
            token,
            source,
            dim,
            NUM_SOURCES,
            dim_mask,
            block_stride_t,
            block_stride_n,
            block_stride_d,
            partial_stride_t,
            partial_stride_d,
            HAS_PARTIAL,
        )
        stats_offset = source * NUM_TOKENS + token
        source_logits = tl.load(
            logits + stats_offset,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
        source_probabilities = tl.where(
            source_mask,
            tl.exp(source_logits - logsumexp),
            0.0,
        )
        mixed += tl.sum(source_probabilities[:, None] * values, axis=0)

    softmax_delta = tl.sum(tl.where(dim_mask, grad * mixed, 0.0), axis=0)
    score_weight_grad = tl.zeros([BLOCK_D], dtype=tl.float32)
    partial_grad = tl.zeros([BLOCK_D], dtype=tl.float32)
    for source_block in range(tl.cdiv(total_sources, BLOCK_N)):
        source = source_block * BLOCK_N + tl.arange(0, BLOCK_N)
        source_mask = source < total_sources
        values = _load_residual_tile(
            block_residual,
            partial_block,
            token,
            source,
            dim,
            NUM_SOURCES,
            dim_mask,
            block_stride_t,
            block_stride_n,
            block_stride_d,
            partial_stride_t,
            partial_stride_d,
            HAS_PARTIAL,
        )
        stats_offset = source * NUM_TOKENS + token
        source_rstd = tl.load(
            rstd + stats_offset,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
        source_logits = tl.load(
            logits + stats_offset,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
        source_probabilities = tl.where(
            source_mask,
            tl.exp(source_logits - logsumexp),
            0.0,
        )
        probability_grad = tl.sum(values * grad[None, :], axis=1)
        score_grad = source_probabilities * (probability_grad - softmax_delta)
        normalized = values * source_rstd[:, None]
        value_grad = source_probabilities[:, None] * grad[None, :] + (
            score_grad * source_rstd
        )[:, None] * (
            score_weight[None, :] - normalized * (source_logits / DIM)[:, None]
        )

        tl.store(
            grad_block_residual
            + token * grad_block_stride_t
            + source[:, None] * grad_block_stride_n
            + dim[None, :] * grad_block_stride_d,
            value_grad,
            mask=(source < NUM_SOURCES)[:, None] & dim_mask[None, :],
        )
        if HAS_PARTIAL:
            partial_grad += tl.sum(
                tl.where(
                    (source == NUM_SOURCES)[:, None],
                    value_grad,
                    0.0,
                ),
                axis=0,
            )
        score_weight_grad += tl.sum(score_grad[:, None] * normalized, axis=0)

    if HAS_PARTIAL:
        tl.store(
            grad_partial + token * grad_partial_stride_t + dim * grad_partial_stride_d,
            partial_grad,
            mask=dim_mask,
        )
    tl.store(grad_score_weight + token * DIM + dim, score_weight_grad, mask=dim_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_T": block_t, "BLOCK_D": block_d},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_t, block_d, num_warps in [
            (1024, 16, 4),
            (2048, 32, 4),
            (2048, 32, 8),
            (4096, 32, 8),
            (4096, 64, 8),
        ]
        for num_stages in [3, 4]
    ],
    key=["NUM_TOKENS", "DIM"],
    cache_results=True,
)
@triton.jit
def _attention_residual_backward_weights_kernel(
    projection_weight,
    norm_weight,
    grad_score_weight,
    grad_projection_weight,
    grad_norm_weight,
    NUM_TOKENS,
    DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    dim = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim < DIM
    score_weight_grad = tl.zeros([BLOCK_D], dtype=tl.float32)
    for token_block in range(0, NUM_TOKENS, BLOCK_T):
        token = token_block.to(tl.int64) + tl.arange(0, BLOCK_T)
        mask = (token[:, None] < NUM_TOKENS) & dim_mask[None, :]
        score_weight_grad += tl.sum(
            tl.load(
                grad_score_weight + token[:, None] * DIM + dim[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32),
            axis=0,
        )

    projection = tl.load(projection_weight + dim, mask=dim_mask, other=0.0).to(
        tl.float32
    )
    norm = tl.load(norm_weight + dim, mask=dim_mask, other=0.0).to(tl.float32)
    tl.store(
        grad_projection_weight + dim,
        score_weight_grad * norm,
        mask=dim_mask,
    )
    tl.store(
        grad_norm_weight + dim,
        score_weight_grad * projection,
        mask=dim_mask,
    )


def attention_residual_forward_kernel(
    partial_block_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection_weight_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_sources, dim = block_residual_TND.shape
    has_partial = partial_block_TD.numel() != 0
    total_sources = num_sources + int(has_partial)
    output_TD = torch.empty(
        (num_tokens, dim),
        device=block_residual_TND.device,
        dtype=block_residual_TND.dtype,
    )
    stats_shape = (total_sources, num_tokens)
    rstd_NT = torch.empty(
        stats_shape,
        device=block_residual_TND.device,
        dtype=torch.float32,
    )
    logits_NT = torch.empty_like(rstd_NT)
    lse_T = torch.empty(
        (num_tokens,),
        device=block_residual_TND.device,
        dtype=torch.float32,
    )
    # Bucket source counts like FLA's padded pointer tuple so nearby counts
    # reuse one autotune result while NUM_SOURCES remains a runtime value.
    source_bucket = max(8, triton.next_power_of_2(total_sources))
    _attention_residual_forward_kernel[(num_tokens,)](
        partial_block_TD,
        block_residual_TND,
        projection_weight_1D,
        norm_weight_D,
        output_TD,
        rstd_NT,
        logits_NT,
        lse_T,
        NUM_TOKENS=num_tokens,
        NUM_SOURCES=num_sources,
        SOURCE_BUCKET=source_bucket,
        DIM=dim,
        EPS=eps,
        block_stride_t=block_residual_TND.stride(0),
        block_stride_n=block_residual_TND.stride(1),
        block_stride_d=block_residual_TND.stride(2),
        partial_stride_t=partial_block_TD.stride(0),
        partial_stride_d=partial_block_TD.stride(1),
        HAS_PARTIAL=has_partial,
        BLOCK_D=triton.next_power_of_2(dim),
    )
    return output_TD, rstd_NT, logits_NT, lse_T


def attention_residual_backward_kernel(
    grad_output_TD: torch.Tensor,
    partial_block_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection_weight_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
    rstd_NT: torch.Tensor,
    logits_NT: torch.Tensor,
    lse_T: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_sources, dim = block_residual_TND.shape
    has_partial = partial_block_TD.numel() != 0
    total_sources = num_sources + int(has_partial)
    grad_partial_TD = torch.empty_like(partial_block_TD)
    grad_block_residual_TND = torch.empty_like(block_residual_TND)
    grad_score_weight_TD = torch.empty(
        (num_tokens, dim),
        device=block_residual_TND.device,
        dtype=torch.float32,
    )
    grad_projection_weight_1D = torch.empty_like(projection_weight_1D)
    grad_norm_weight_D = torch.empty_like(norm_weight_D)
    source_bucket = max(8, triton.next_power_of_2(total_sources))
    _attention_residual_backward_values_kernel[(num_tokens,)](
        grad_output_TD,
        partial_block_TD,
        block_residual_TND,
        projection_weight_1D,
        norm_weight_D,
        rstd_NT,
        logits_NT,
        lse_T,
        grad_partial_TD,
        grad_block_residual_TND,
        grad_score_weight_TD,
        NUM_TOKENS=num_tokens,
        NUM_SOURCES=num_sources,
        SOURCE_BUCKET=source_bucket,
        DIM=dim,
        block_stride_t=block_residual_TND.stride(0),
        block_stride_n=block_residual_TND.stride(1),
        block_stride_d=block_residual_TND.stride(2),
        partial_stride_t=partial_block_TD.stride(0),
        partial_stride_d=partial_block_TD.stride(1),
        grad_block_stride_t=grad_block_residual_TND.stride(0),
        grad_block_stride_n=grad_block_residual_TND.stride(1),
        grad_block_stride_d=grad_block_residual_TND.stride(2),
        grad_output_stride_t=grad_output_TD.stride(0),
        grad_output_stride_d=grad_output_TD.stride(1),
        grad_partial_stride_t=grad_partial_TD.stride(0),
        grad_partial_stride_d=grad_partial_TD.stride(1),
        HAS_PARTIAL=has_partial,
        BLOCK_D=triton.next_power_of_2(dim),
    )

    def grid(meta):
        return (triton.cdiv(dim, meta["BLOCK_D"]),)

    _attention_residual_backward_weights_kernel[grid](
        projection_weight_1D,
        norm_weight_D,
        grad_score_weight_TD,
        grad_projection_weight_1D,
        grad_norm_weight_D,
        NUM_TOKENS=num_tokens,
        DIM=dim,
    )
    return (
        grad_partial_TD,
        grad_block_residual_TND,
        grad_projection_weight_1D,
        grad_norm_weight_D,
    )


@torch.library.custom_op(
    "torchtitan::kimi_attention_residual_forward",
    mutates_args=(),
    device_types="cuda",
)
def _attention_residual_forward_op(
    partial_block_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection_weight_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return attention_residual_forward_kernel(
        partial_block_TD,
        block_residual_TND,
        projection_weight_1D,
        norm_weight_D,
        eps,
    )


@_attention_residual_forward_op.register_fake
def _attention_residual_forward_op_fake(
    partial_block_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection_weight_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del projection_weight_1D, norm_weight_D, eps
    num_tokens, num_sources, dim = block_residual_TND.shape
    total_sources = num_sources + int(partial_block_TD.numel() != 0)
    output_TD = block_residual_TND.new_empty((num_tokens, dim))
    rstd_NT = block_residual_TND.new_empty(
        (total_sources, num_tokens), dtype=torch.float32
    )
    logits_NT = torch.empty_like(rstd_NT)
    lse_T = block_residual_TND.new_empty((num_tokens,), dtype=torch.float32)
    return output_TD, rstd_NT, logits_NT, lse_T


@torch.library.custom_op(
    "torchtitan::kimi_attention_residual_backward",
    mutates_args=(),
    device_types="cuda",
)
def _attention_residual_backward_op(
    grad_output_TD: torch.Tensor,
    partial_block_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection_weight_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
    rstd_NT: torch.Tensor,
    logits_NT: torch.Tensor,
    lse_T: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return attention_residual_backward_kernel(
        grad_output_TD,
        partial_block_TD,
        block_residual_TND,
        projection_weight_1D,
        norm_weight_D,
        rstd_NT,
        logits_NT,
        lse_T,
    )


@_attention_residual_backward_op.register_fake
def _attention_residual_backward_op_fake(
    grad_output_TD: torch.Tensor,
    partial_block_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection_weight_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
    rstd_NT: torch.Tensor,
    logits_NT: torch.Tensor,
    lse_T: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del grad_output_TD, rstd_NT, logits_NT, lse_T
    return (
        torch.empty_like(partial_block_TD),
        torch.empty_like(block_residual_TND),
        torch.empty_like(projection_weight_1D),
        torch.empty_like(norm_weight_D),
    )


def _attention_residual_setup_context(ctx, inputs, output) -> None:
    (
        partial_block_TD,
        block_residual_TND,
        projection_weight_1D,
        norm_weight_D,
        _,
    ) = inputs
    _, rstd_NT, logits_NT, lse_T = output
    ctx.save_for_backward(
        partial_block_TD,
        block_residual_TND,
        projection_weight_1D,
        norm_weight_D,
        rstd_NT,
        logits_NT,
        lse_T,
    )
    ctx.mark_non_differentiable(rstd_NT, logits_NT, lse_T)


def _attention_residual_autograd_backward(
    ctx,
    grad_output_TD: torch.Tensor,
    grad_rstd_NT: torch.Tensor | None,
    grad_logits_NT: torch.Tensor | None,
    grad_lse_T: torch.Tensor | None,
):
    del grad_rstd_NT, grad_logits_NT, grad_lse_T
    (
        partial_block_TD,
        block_residual_TND,
        projection_weight_1D,
        norm_weight_D,
        rstd_NT,
        logits_NT,
        lse_T,
    ) = ctx.saved_tensors
    grads = _attention_residual_backward_op(
        grad_output_TD,
        partial_block_TD,
        block_residual_TND,
        projection_weight_1D,
        norm_weight_D,
        rstd_NT,
        logits_NT,
        lse_T,
    )
    return (*grads, None)


_attention_residual_forward_op.register_autograd(
    _attention_residual_autograd_backward,
    setup_context=_attention_residual_setup_context,
)


_TOKEN_AXES = ("dp", "cp", "tp")
_ACTIVATION_LOCAL_TYPE = {axis: spmd.V for axis in _TOKEN_AXES}
_ACTIVATION_TD = (
    _ACTIVATION_LOCAL_TYPE,
    spmd.PartitionSpec(_TOKEN_AXES, None),
)
_ACTIVATION_TND = (
    _ACTIVATION_LOCAL_TYPE,
    spmd.PartitionSpec(_TOKEN_AXES, None, None),
)
_ACTIVATION_NT = (
    _ACTIVATION_LOCAL_TYPE,
    spmd.PartitionSpec(None, _TOKEN_AXES),
)
_ACTIVATION_T = (
    _ACTIVATION_LOCAL_TYPE,
    spmd.PartitionSpec(_TOKEN_AXES),
)
_REPLICATED = {axis: spmd.R for axis in _TOKEN_AXES}


# TODO(pianpwk): Replace this local_map workaround with a custom-op SPMD
# propagation rule when that registration API is available.
@spmd.local_map(
    in_types=(
        _ACTIVATION_TD,
        _ACTIVATION_TND,
        _REPLICATED,
        _REPLICATED,
        None,
    ),
    out_types=(
        _ACTIVATION_TD,
        _ACTIVATION_NT,
        _ACTIVATION_NT,
        _ACTIVATION_T,
    ),
)
def _attention_residual_forward_local_with_partial(
    partial_block_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection_weight_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _attention_residual_forward_op(
        partial_block_TD,
        block_residual_TND,
        projection_weight_1D,
        norm_weight_D,
        eps,
    )


@spmd.local_map(
    in_types=(
        _ACTIVATION_TND,
        _REPLICATED,
        _REPLICATED,
        None,
    ),
    out_types=(
        _ACTIVATION_TD,
        _ACTIVATION_NT,
        _ACTIVATION_NT,
        _ACTIVATION_T,
    ),
)
def _attention_residual_forward_local_without_partial(
    block_residual_TND: torch.Tensor,
    projection_weight_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    partial_block_TD = block_residual_TND.new_empty((0, block_residual_TND.shape[-1]))
    return _attention_residual_forward_op(
        partial_block_TD,
        block_residual_TND,
        projection_weight_1D,
        norm_weight_D,
        eps,
    )


def attention_residual_op(
    partial_block_TD: torch.Tensor | None,
    block_residual_TND: torch.Tensor,
    projection_weight_1D: torch.Tensor,
    norm_weight_D: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if partial_block_TD is None:
        output_TD, _, _, _ = _attention_residual_forward_local_without_partial(
            block_residual_TND,
            projection_weight_1D,
            norm_weight_D,
            eps,
        )
    else:
        output_TD, _, _, _ = _attention_residual_forward_local_with_partial(
            partial_block_TD,
            block_residual_TND,
            projection_weight_1D,
            norm_weight_D,
            eps,
        )
    return output_TD


class TritonAttentionResidual(AttentionResidual):
    """Kimi K3 attention residual aggregation implemented with Triton."""

    @dataclass(kw_only=True, slots=True)
    class Config(AttentionResidual.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(
        self,
        partial_block_TD: torch.Tensor | None,
        block_residual_TND: torch.Tensor,
        projection: Linear,
        norm: RMSNorm,
    ) -> torch.Tensor:
        assert projection.bias is None
        assert norm.eps is not None
        return attention_residual_op(
            partial_block_TD,
            block_residual_TND,
            projection.weight,
            norm.weight,
            norm.eps,
        )


@override(
    target=AttentionResidual.Config,
    exact=True,
    description="Fuse Kimi K3 attention residual aggregation with Triton.",
)
def triton_attention_residual(
    cfg: AttentionResidual.Config,
) -> TritonAttentionResidual.Config:
    return derive(cfg, TritonAttentionResidual.Config)
