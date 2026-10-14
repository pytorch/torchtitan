# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exact combined auxiliary and selected-weight backward for DeepSeek V3."""

import torch
import triton
import triton.language as tl

from .router_aux import _sum_256
from .router_weights import _normalized_selected_gradient, _split_bf16

# Shape suffixes: T = tokens, E = experts (256), K = selections (8), P = BF16 pieces (3).


@triton.jit
def _router_backward_kernel(
    grad_weights,
    selected_scores,
    denominator,
    expert_ids,
    grad_per_expert,
    scores,
    raw_norm,
    clamped_norm,
    stacked_grad,
    num_tokens: tl.constexpr,
    grad_token_stride: tl.constexpr,
    grad_selected_stride: tl.constexpr,
    grad_expert_stride: tl.constexpr,
    route_scale: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    token_T = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    expert_E = tl.arange(0, 256)
    selected_K = tl.arange(0, 8)
    offset_TE = token_T[:, None] * 256 + expert_E[None, :]
    offset_TK = token_T[:, None] * 8 + selected_K[None, :]
    scores_TE = tl.load(scores + offset_TE, token_T[:, None] < num_tokens, 0.0)
    grad_E = tl.load(grad_per_expert + expert_E * grad_expert_stride)
    raw_norm_T = tl.load(raw_norm + token_T, token_T < num_tokens, 0.0)
    clamped_norm_T = tl.load(clamped_norm + token_T, token_T < num_tokens, 1.0)
    normalized_TE = tl.div_rn(scores_TE, clamped_norm_T[:, None])
    grad_norm_TE = -grad_E[None, :] * tl.div_rn(normalized_TE, clamped_norm_T[:, None])
    grad_norm_T = _sum_256(grad_norm_TE, BLOCK_T, num_tokens < 16)
    grad_norm_T = tl.where(raw_norm_T > epsilon, grad_norm_T, 0.0)
    sign_TE = tl.where(scores_TE > 0.0, 1.0, tl.where(scores_TE < 0.0, -1.0, 0.0))
    aux_grad_TE = tl.div_rn(grad_E[None, :], clamped_norm_T[:, None]) + (
        sign_TE * grad_norm_T[:, None]
    )
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
    router_grad_TE = tl.full((BLOCK_T, 256), 0.0, tl.float32)
    for selected_id in tl.static_range(8):
        selected_index_T1 = tl.full((BLOCK_T, 1), selected_id, tl.int32)
        expert_id_T1 = tl.gather(expert_id_TK, selected_index_T1, 1)
        grad_selected_T1 = tl.gather(grad_selected_TK, selected_index_T1, 1)
        router_grad_TE = tl.where(
            expert_E[None, :] == expert_id_T1, grad_selected_T1, router_grad_TE
        )
    score_grad_TE = aux_grad_TE + router_grad_TE
    gate_grad_TE = (score_grad_TE * (1.0 - scores_TE)) * scores_TE
    hi_TE, mid_TE, lo_TE = _split_bf16(gate_grad_TE)
    stacked_offset_TE = token_T[:, None] * 768 + expert_E[None, :]
    tl.store(stacked_grad + stacked_offset_TE, hi_TE, token_T[:, None] < num_tokens)
    tl.store(
        stacked_grad + stacked_offset_TE + 256, mid_TE, token_T[:, None] < num_tokens
    )
    tl.store(
        stacked_grad + stacked_offset_TE + 512, lo_TE, token_T[:, None] < num_tokens
    )


def router_backward(
    grad_weights_TK: torch.Tensor,
    selected_scores_TK: torch.Tensor,
    denominator_T1: torch.Tensor,
    expert_ids_TK: torch.Tensor,
    grad_per_expert_E: torch.Tensor,
    scores_TE: torch.Tensor,
    raw_norm_T1: torch.Tensor,
    clamped_norm_T1: torch.Tensor,
    *,
    route_scale: float = 2.5,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return the BF16 gate-gradient pieces without intermediate score gradients.

    Scores are contiguous CUDA FP32 [T, 256], with the saved forward tensors
    from ``router_weights_forward`` and ``router_aux_forward``. The expert
    gradient has shape [256]; selected-weight gradients have shape [T, 8].
    """
    stacked_grad_TPE = scores_TE.new_empty(
        (scores_TE.shape[0], 768), dtype=torch.bfloat16
    )
    _router_backward_kernel[(triton.cdiv(scores_TE.shape[0], 8),)](
        grad_weights_TK,
        selected_scores_TK,
        denominator_T1,
        expert_ids_TK,
        grad_per_expert_E,
        scores_TE,
        raw_norm_T1,
        clamped_norm_T1,
        stacked_grad_TPE,
        scores_TE.shape[0],
        grad_weights_TK.stride(0),
        grad_weights_TK.stride(1),
        grad_per_expert_E.stride(0),
        route_scale,
        epsilon,
        8,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return stacked_grad_TPE
