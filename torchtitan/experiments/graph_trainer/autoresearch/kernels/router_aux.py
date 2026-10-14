# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exact FP32 L1 normalization for DeepSeek V3 auxiliary router loss."""

import torch
import triton
import triton.language as tl

# Shape suffixes: T = tokens, E = experts (256), G = reduction lane groups.


@triton.jit
def _fold_halves(values_TG, BLOCK_T: tl.constexpr, HALF: tl.constexpr):
    lower_TG, upper_TG = tl.split(
        tl.permute(values_TG.reshape((BLOCK_T, 2, HALF)), (0, 2, 1))
    )
    return lower_TG + upper_TG


@triton.jit
def _sum_256(values_TE, BLOCK_T: tl.constexpr, WIDE_REDUCTION: tl.constexpr):
    # ATen vectorizes in groups of four. Small token counts use 64 lanes;
    # production uses 32 lanes with two vectors per lane.
    if WIDE_REDUCTION:
        NUM_GROUPS: tl.constexpr = 64
        values_TG4 = values_TE.reshape((BLOCK_T, 64, 4))
    else:
        NUM_GROUPS: tl.constexpr = 32
        values_TG4 = _fold_halves(values_TE, BLOCK_T, 128).reshape((BLOCK_T, 32, 4))
    even_TG2, odd_TG2 = tl.split(values_TG4.reshape((BLOCK_T, NUM_GROUPS, 2, 2)))
    value_0_TG, value_2_TG = tl.split(even_TG2)
    value_1_TG, value_3_TG = tl.split(odd_TG2)
    values_TG = ((value_0_TG + value_1_TG) + value_2_TG) + value_3_TG
    if WIDE_REDUCTION:
        values_TG = _fold_halves(values_TG, BLOCK_T, 32)
    values_TG = _fold_halves(values_TG, BLOCK_T, 16)
    values_TG = _fold_halves(values_TG, BLOCK_T, 8)
    values_TG = _fold_halves(values_TG, BLOCK_T, 4)
    values_TG = _fold_halves(values_TG, BLOCK_T, 2)
    values_TG = _fold_halves(values_TG, BLOCK_T, 1)
    reduced_T = values_TG.reshape((BLOCK_T,))
    # ATen seeds every accumulator with positive zero.
    return tl.where(reduced_T == 0.0, 0.0, reduced_T)


@triton.jit
def _router_aux_forward_kernel(
    scores,
    normalized_scores,
    raw_norm,
    clamped_norm,
    num_tokens: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    token_T = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    expert_E = tl.arange(0, 256)
    offset_TE = token_T[:, None] * 256 + expert_E[None, :]
    scores_TE = tl.load(scores + offset_TE, token_T[:, None] < num_tokens, 0.0)
    raw_norm_T = _sum_256(tl.abs(scores_TE), BLOCK_T, num_tokens < 16)
    clamped_norm_T = tl.maximum(raw_norm_T, epsilon)
    normalized_TE = tl.div_rn(scores_TE, clamped_norm_T[:, None])
    tl.store(
        normalized_scores + offset_TE, normalized_TE, token_T[:, None] < num_tokens
    )
    tl.store(raw_norm + token_T, raw_norm_T, token_T < num_tokens)
    tl.store(clamped_norm + token_T, clamped_norm_T, token_T < num_tokens)


@triton.jit
def _router_aux_backward_kernel(
    grad_per_expert,
    scores,
    raw_norm,
    clamped_norm,
    grad_scores,
    num_tokens: tl.constexpr,
    grad_expert_stride: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    token_T = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    expert_E = tl.arange(0, 256)
    offset_TE = token_T[:, None] * 256 + expert_E[None, :]
    scores_TE = tl.load(scores + offset_TE, token_T[:, None] < num_tokens, 0.0)
    grad_E = tl.load(grad_per_expert + expert_E * grad_expert_stride)
    raw_norm_T = tl.load(raw_norm + token_T, token_T < num_tokens, 0.0)
    clamped_norm_T = tl.load(clamped_norm + token_T, token_T < num_tokens, 1.0)
    normalized_TE = tl.div_rn(scores_TE, clamped_norm_T[:, None])
    grad_norm_TE = -grad_E[None, :] * tl.div_rn(normalized_TE, clamped_norm_T[:, None])
    grad_norm_T = _sum_256(grad_norm_TE, BLOCK_T, num_tokens < 16)
    grad_norm_T = tl.where(raw_norm_T > epsilon, grad_norm_T, 0.0)
    sign_TE = tl.where(scores_TE > 0.0, 1.0, tl.where(scores_TE < 0.0, -1.0, 0.0))
    grad_TE = tl.div_rn(grad_E[None, :], clamped_norm_T[:, None]) + (
        sign_TE * grad_norm_T[:, None]
    )
    tl.store(grad_scores + offset_TE, grad_TE, token_T[:, None] < num_tokens)


def router_aux_forward(
    scores_TE: torch.Tensor, *, epsilon: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized scores, raw L1 norm, and clamped norm.

    The input is a contiguous, 16-byte-aligned CUDA FP32 tensor [T, 256].
    The reduction over tokens and auxiliary-loss scaling remain outside.
    """
    if (
        scores_TE.ndim != 2
        or scores_TE.shape[1] != 256
        or scores_TE.dtype != torch.float32
        or not scores_TE.is_cuda
        or not scores_TE.is_contiguous()
        or scores_TE.data_ptr() % 16 != 0
    ):
        raise ValueError(
            "Expected contiguous, 16-byte-aligned CUDA FP32 scores [T, 256]."
        )
    normalized_scores_TE = torch.empty_like(scores_TE)
    raw_norm_T1 = scores_TE.new_empty((scores_TE.shape[0], 1))
    clamped_norm_T1 = torch.empty_like(raw_norm_T1)
    _router_aux_forward_kernel[(triton.cdiv(scores_TE.shape[0], 2),)](
        scores_TE,
        normalized_scores_TE,
        raw_norm_T1,
        clamped_norm_T1,
        scores_TE.shape[0],
        epsilon,
        2,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return normalized_scores_TE, raw_norm_T1, clamped_norm_T1


def router_aux_backward(
    grad_per_expert_E: torch.Tensor,
    scores_TE: torch.Tensor,
    raw_norm_T1: torch.Tensor,
    clamped_norm_T1: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Evaluate the traced auxiliary normalization gradient for finite scores."""
    grad_scores_TE = torch.empty_like(scores_TE)
    _router_aux_backward_kernel[(triton.cdiv(scores_TE.shape[0], 2),)](
        grad_per_expert_E,
        scores_TE,
        raw_norm_T1,
        clamped_norm_T1,
        grad_scores_TE,
        scores_TE.shape[0],
        grad_per_expert_E.stride(0),
        epsilon,
        2,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return grad_scores_TE
