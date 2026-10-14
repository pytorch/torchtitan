# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exact post-selection weights and gate gradients for the DeepSeek V3 router."""

import torch
import triton
import triton.language as tl

# Shape suffixes: T = tokens, E = experts, K = selected experts (8), P = BF16 pieces (3).


@triton.jit
def _sum_eight(
    values_TK, BLOCK_T: tl.constexpr, STRIDED_REDUCTION: tl.constexpr = False
):
    value_0 = tl.gather(values_TK, tl.full((BLOCK_T, 1), 0, tl.int32), 1)
    value_1 = tl.gather(values_TK, tl.full((BLOCK_T, 1), 1, tl.int32), 1)
    value_2 = tl.gather(values_TK, tl.full((BLOCK_T, 1), 2, tl.int32), 1)
    value_3 = tl.gather(values_TK, tl.full((BLOCK_T, 1), 3, tl.int32), 1)
    value_4 = tl.gather(values_TK, tl.full((BLOCK_T, 1), 4, tl.int32), 1)
    value_5 = tl.gather(values_TK, tl.full((BLOCK_T, 1), 5, tl.int32), 1)
    value_6 = tl.gather(values_TK, tl.full((BLOCK_T, 1), 6, tl.int32), 1)
    value_7 = tl.gather(values_TK, tl.full((BLOCK_T, 1), 7, tl.int32), 1)
    if STRIDED_REDUCTION:
        # ATen combines four thread-local accumulators for a strided reduction.
        return (
            (((value_0 + value_4) + (value_1 + value_5)) + (value_2 + value_6))
            + (value_3 + value_7)
        ).reshape((BLOCK_T,))
    return (
        ((value_0 + value_4) + (value_2 + value_6))
        + ((value_1 + value_5) + (value_3 + value_7))
    ).reshape((BLOCK_T,))


@triton.jit
def _normalized_selected_gradient(
    selected_TK,
    denominator_T,
    grad_weight_TK,
    route_scale: tl.constexpr,
    BLOCK_T: tl.constexpr,
    STRIDED_REDUCTION: tl.constexpr,
):
    grad_scaled_TK = grad_weight_TK * route_scale
    normalized_TK = tl.div_rn(selected_TK, denominator_T[:, None])
    grad_denominator_TK = -grad_scaled_TK * tl.div_rn(
        normalized_TK, denominator_T[:, None]
    )
    grad_selected_TK = (
        tl.div_rn(grad_scaled_TK, denominator_T[:, None])
        + _sum_eight(grad_denominator_TK, BLOCK_T, STRIDED_REDUCTION)[:, None]
    )
    # Top-k indices are unique. Match scatter_add's addition to positive zero.
    return tl.where(grad_selected_TK == 0.0, 0.0, grad_selected_TK)


@triton.jit
def _split_bf16(grad_TE):
    hi_TE = grad_TE.to(tl.bfloat16)
    hi_fp32_TE = (hi_TE.to(tl.int16, bitcast=True).to(tl.int32) << 16).to(
        tl.float32, bitcast=True
    )
    rest_TE = grad_TE - hi_fp32_TE
    mid_TE = rest_TE.to(tl.bfloat16)
    mid_fp32_TE = (mid_TE.to(tl.int16, bitcast=True).to(tl.int32) << 16).to(
        tl.float32, bitcast=True
    )
    lo_TE = (rest_TE - mid_fp32_TE).to(tl.bfloat16)
    return hi_TE, mid_TE, lo_TE


@triton.jit
def _router_weights_forward_kernel(
    scores,
    expert_ids,
    selected_scores,
    denominator,
    weights,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    route_scale: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    token_T = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    selected_K = tl.arange(0, 8)
    offset_TK = token_T[:, None] * 8 + selected_K[None, :]
    expert_id_TK = tl.load(expert_ids + offset_TK, token_T[:, None] < num_tokens, 0)
    selected_TK = tl.load(
        scores + token_T[:, None] * num_experts + expert_id_TK,
        token_T[:, None] < num_tokens,
        0.0,
    )
    # ATen's eight-element reduction uses the same offset-4/2/1 tree.
    denominator_T = _sum_eight(selected_TK, BLOCK_T) + epsilon
    weight_TK = tl.div_rn(selected_TK, denominator_T[:, None]) * route_scale
    tl.store(selected_scores + offset_TK, selected_TK, token_T[:, None] < num_tokens)
    tl.store(denominator + token_T, denominator_T, token_T < num_tokens)
    tl.store(weights + offset_TK, weight_TK, token_T[:, None] < num_tokens)


@triton.jit
def _router_weights_backward_kernel(
    selected_scores,
    denominator,
    expert_ids,
    grad_weights,
    grad_scores,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    route_scale: tl.constexpr,
    grad_token_stride: tl.constexpr,
    grad_selected_stride: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token_T = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    selected_K = tl.arange(0, 8)
    expert_E = tl.arange(0, BLOCK_E)
    offset_TK = token_T[:, None] * 8 + selected_K[None, :]
    selected_TK = tl.load(
        selected_scores + offset_TK, token_T[:, None] < num_tokens, 0.0
    )
    denominator_T = tl.load(denominator + token_T, token_T < num_tokens, 1.0)
    grad_weight_TK = tl.load(
        grad_weights
        + token_T[:, None] * grad_token_stride
        + selected_K[None, :] * grad_selected_stride,
        token_T[:, None] < num_tokens,
        0.0,
    )
    expert_id_TK = tl.load(expert_ids + offset_TK, token_T[:, None] < num_tokens, 0)
    grad_selected_TK = _normalized_selected_gradient(
        selected_TK,
        denominator_T,
        grad_weight_TK,
        route_scale,
        BLOCK_T,
        grad_token_stride > 0 and grad_token_stride < grad_selected_stride,
    )
    tl.store(
        grad_scores + token_T[:, None] * num_experts + expert_E[None, :],
        0.0,
        (token_T[:, None] < num_tokens) & (expert_E[None, :] < num_experts),
    )
    # Each CTA owns complete rows, so initialization must precede its scatter.
    tl.debug_barrier()
    tl.store(
        grad_scores + token_T[:, None] * num_experts + expert_id_TK,
        grad_selected_TK,
        token_T[:, None] < num_tokens,
    )


@triton.jit
def _router_gate_backward_split_kernel(
    aux_grad,
    router_grad,
    scores,
    stacked_grad,
    num_values: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    aux_grad_TE = tl.load(aux_grad + offset, offset < num_values, 0.0)
    router_grad_TE = tl.load(router_grad + offset, offset < num_values, 0.0)
    scores_TE = tl.load(scores + offset, offset < num_values, 0.0)
    score_grad_TE = aux_grad_TE + router_grad_TE
    # Match ATen's left-associated sigmoid backward and the BF16 split.
    gate_grad_TE = (score_grad_TE * (1.0 - scores_TE)) * scores_TE
    hi_TE, mid_TE, lo_TE = _split_bf16(gate_grad_TE)
    stacked_offset = (offset // num_experts) * (3 * num_experts) + offset % num_experts
    tl.store(stacked_grad + stacked_offset, hi_TE, offset < num_values)
    tl.store(stacked_grad + stacked_offset + num_experts, mid_TE, offset < num_values)
    tl.store(
        stacked_grad + stacked_offset + 2 * num_experts, lo_TE, offset < num_values
    )


@triton.jit
def _router_weights_gate_backward_kernel(
    selected_scores,
    denominator,
    expert_ids,
    grad_weights,
    aux_grad,
    scores,
    stacked_grad,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    route_scale: tl.constexpr,
    grad_token_stride: tl.constexpr,
    grad_selected_stride: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token_T = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    selected_K = tl.arange(0, 8)
    expert_E = tl.arange(0, BLOCK_E)
    offset_TK = token_T[:, None] * 8 + selected_K[None, :]
    selected_TK = tl.load(
        selected_scores + offset_TK, token_T[:, None] < num_tokens, 0.0
    )
    denominator_T = tl.load(denominator + token_T, token_T < num_tokens, 1.0)
    grad_weight_TK = tl.load(
        grad_weights
        + token_T[:, None] * grad_token_stride
        + selected_K[None, :] * grad_selected_stride,
        token_T[:, None] < num_tokens,
        0.0,
    )
    expert_id_TK = tl.load(expert_ids + offset_TK, token_T[:, None] < num_tokens, 0)
    grad_selected_TK = _normalized_selected_gradient(
        selected_TK,
        denominator_T,
        grad_weight_TK,
        route_scale,
        BLOCK_T,
        grad_token_stride > 0 and grad_token_stride < grad_selected_stride,
    )
    router_grad_TE = tl.full((BLOCK_T, BLOCK_E), 0.0, tl.float32)
    for selected_id in tl.static_range(8):
        selected_index_T1 = tl.full((BLOCK_T, 1), selected_id, tl.int32)
        expert_id_T1 = tl.gather(expert_id_TK, selected_index_T1, 1)
        grad_selected_T1 = tl.gather(grad_selected_TK, selected_index_T1, 1)
        router_grad_TE = tl.where(
            expert_E[None, :] == expert_id_T1, grad_selected_T1, router_grad_TE
        )
    offset_TE = token_T[:, None] * num_experts + expert_E[None, :]
    mask_TE = (token_T[:, None] < num_tokens) & (expert_E[None, :] < num_experts)
    aux_grad_TE = tl.load(aux_grad + offset_TE, mask_TE, 0.0)
    scores_TE = tl.load(scores + offset_TE, mask_TE, 0.0)
    score_grad_TE = aux_grad_TE + router_grad_TE
    gate_grad_TE = (score_grad_TE * (1.0 - scores_TE)) * scores_TE
    hi_TE, mid_TE, lo_TE = _split_bf16(gate_grad_TE)
    stacked_offset_TE = token_T[:, None] * (3 * num_experts) + expert_E[None, :]
    tl.store(stacked_grad + stacked_offset_TE, hi_TE, mask_TE)
    tl.store(stacked_grad + stacked_offset_TE + num_experts, mid_TE, mask_TE)
    tl.store(stacked_grad + stacked_offset_TE + 2 * num_experts, lo_TE, mask_TE)


def router_weights_forward(
    scores_TE: torch.Tensor,
    expert_ids_TK: torch.Tensor,
    *,
    route_scale: float = 2.5,
    epsilon: float = 1e-20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weights, selected scores, and denominator with eager saved state.

    Inputs are contiguous CUDA FP32 scores and CUDA int64 top-8 indices.
    Selection and auxiliary-loss evaluation remain outside this region.
    """
    if (
        scores_TE.ndim != 2
        or scores_TE.dtype != torch.float32
        or not scores_TE.is_cuda
        or not scores_TE.is_contiguous()
        or expert_ids_TK.shape != (scores_TE.shape[0], 8)
        or expert_ids_TK.dtype != torch.int64
        or expert_ids_TK.device != scores_TE.device
        or not expert_ids_TK.is_contiguous()
    ):
        raise ValueError(
            "Expected contiguous CUDA FP32 [T, E] scores and int64 [T, 8] indices."
        )
    selected_scores_TK = torch.empty_like(expert_ids_TK, dtype=scores_TE.dtype)
    denominator_T1 = scores_TE.new_empty((scores_TE.shape[0], 1))
    weights_TK = torch.empty_like(selected_scores_TK)
    _router_weights_forward_kernel[(triton.cdiv(scores_TE.shape[0], 16),)](
        scores_TE,
        expert_ids_TK,
        selected_scores_TK,
        denominator_T1,
        weights_TK,
        scores_TE.shape[0],
        scores_TE.shape[1],
        route_scale,
        epsilon,
        16,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return weights_TK, selected_scores_TK, denominator_T1


def router_weights_backward(
    grad_weights_TK: torch.Tensor,
    selected_scores_TK: torch.Tensor,
    denominator_T1: torch.Tensor,
    expert_ids_TK: torch.Tensor,
    *,
    num_experts: int,
    route_scale: float = 2.5,
) -> torch.Tensor:
    """Differentiate the normalized weights and scatter to unique expert IDs."""
    grad_scores_TE = selected_scores_TK.new_empty(
        (selected_scores_TK.shape[0], num_experts)
    )
    _router_weights_backward_kernel[(triton.cdiv(selected_scores_TK.shape[0], 16),)](
        selected_scores_TK,
        denominator_T1,
        expert_ids_TK,
        grad_weights_TK,
        grad_scores_TE,
        selected_scores_TK.shape[0],
        num_experts,
        route_scale,
        grad_weights_TK.stride(0),
        grad_weights_TK.stride(1),
        16,
        triton.next_power_of_2(num_experts),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return grad_scores_TE


def router_gate_backward_split(
    aux_grad_TE: torch.Tensor,
    router_grad_TE: torch.Tensor,
    scores_TE: torch.Tensor,
) -> torch.Tensor:
    """Fuse gradient accumulation, sigmoid backward, and the three BF16 pieces.

    Inputs are contiguous CUDA FP32 tensors with shape [T, E]. The output
    has shape [T, 3 * E] and the original hi, mid, lo concatenation order.
    """
    stacked_grad_TPE = scores_TE.new_empty(
        (scores_TE.shape[0], scores_TE.shape[1] * 3), dtype=torch.bfloat16
    )
    _router_gate_backward_split_kernel[(triton.cdiv(scores_TE.numel(), 512),)](
        aux_grad_TE,
        router_grad_TE,
        scores_TE,
        stacked_grad_TPE,
        scores_TE.numel(),
        scores_TE.shape[1],
        512,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return stacked_grad_TPE


def router_weights_gate_backward(
    grad_weights_TK: torch.Tensor,
    selected_scores_TK: torch.Tensor,
    denominator_T1: torch.Tensor,
    expert_ids_TK: torch.Tensor,
    aux_grad_TE: torch.Tensor,
    scores_TE: torch.Tensor,
    *,
    route_scale: float = 2.5,
) -> torch.Tensor:
    """Fuse normalized-weight and gate backward without a dense route gradient."""
    stacked_grad_TPE = scores_TE.new_empty(
        (scores_TE.shape[0], scores_TE.shape[1] * 3), dtype=torch.bfloat16
    )
    _router_weights_gate_backward_kernel[(scores_TE.shape[0],)](
        selected_scores_TK,
        denominator_T1,
        expert_ids_TK,
        grad_weights_TK,
        aux_grad_TE,
        scores_TE,
        stacked_grad_TPE,
        scores_TE.shape[0],
        scores_TE.shape[1],
        route_scale,
        grad_weights_TK.stride(0),
        grad_weights_TK.stride(1),
        1,
        triton.next_power_of_2(scores_TE.shape[1]),
        num_warps=1,
        enable_fp_fusion=False,
    )
    return stacked_grad_TPE


class _RouterWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores_TE, expert_ids_TK, route_scale, epsilon):
        weights_TK, selected_scores_TK, denominator_T1 = router_weights_forward(
            scores_TE, expert_ids_TK, route_scale=route_scale, epsilon=epsilon
        )
        ctx.save_for_backward(selected_scores_TK, denominator_T1, expert_ids_TK)
        ctx.num_experts = scores_TE.shape[1]
        ctx.route_scale = route_scale
        return weights_TK

    @staticmethod
    def backward(ctx, grad_weights_TK):
        selected_scores_TK, denominator_T1, expert_ids_TK = ctx.saved_tensors
        grad_scores_TE = router_weights_backward(
            grad_weights_TK,
            selected_scores_TK,
            denominator_T1,
            expert_ids_TK,
            num_experts=ctx.num_experts,
            route_scale=ctx.route_scale,
        )
        return grad_scores_TE, None, None, None


def normalized_router_weights(
    scores_TE: torch.Tensor,
    expert_ids_TK: torch.Tensor,
    *,
    route_scale: float = 2.5,
    epsilon: float = 1e-20,
) -> torch.Tensor:
    """Evaluate exact top-8 router weights with a first-order backward."""
    return _RouterWeights.apply(scores_TE, expert_ids_TK, route_scale, epsilon)
