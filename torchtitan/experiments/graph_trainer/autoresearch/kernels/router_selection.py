# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exact group-limited selection for the DeepSeek V3 production router."""

import torch
import triton
import triton.language as tl

# Shape suffix legend:
#   T = num tokens, E = num experts, G = num expert groups,
#   L = num selected groups, K = num selected experts,
#   P = num experts per group, Q = two experts, C = group expert candidates


@triton.jit
def _radix_key(value):
    bits = value.to(tl.uint32, bitcast=True)
    mask = tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
    return tl.where(value == value, bits ^ mask, 0xFFFFFFFF).to(tl.uint32)


@triton.jit
def _radix_value(key):
    mask = tl.where((key & 0x80000000) != 0, 0x80000000, 0xFFFFFFFF).to(tl.uint32)
    return (key ^ mask).to(tl.float32, bitcast=True)


@triton.jit
def _select_experts_fused_kernel(
    scores,
    expert_bias,
    selected_expert_ids,
    num_tokens: tl.constexpr,
    has_expert_bias: tl.constexpr,
    block_tokens: tl.constexpr,
):
    token_ids = tl.program_id(0) * block_tokens + tl.arange(0, block_tokens)
    expert_ids = tl.arange(0, 256)
    scores_TE = tl.load(
        scores + token_ids[:, None] * 256 + expert_ids[None, :],
        token_ids[:, None] < num_tokens,
        other=0.0,
    )
    if has_expert_bias:
        expert_bias_E = tl.load(expert_bias + expert_ids)
        scores_TE = scores_TE + expert_bias_E[None, :]
    score_keys_TE = _radix_key(scores_TE)
    score_keys_TGP = tl.reshape(score_keys_TE, (block_tokens, 8, 32))
    # A group's ninth expert cannot be in the global top eight. Reuse this
    # local top eight for both the group decision and the final cutoff.
    top_group_expert_keys_TGK = tl.topk(score_keys_TGP, 8, dim=2)
    top2_positions_TGQ = tl.broadcast_to(
        tl.arange(0, 2)[None, None, :], (block_tokens, 8, 2)
    )
    top2_keys_TGQ = tl.gather(top_group_expert_keys_TGK, top2_positions_TGQ, axis=2)
    top2_scores_TGQ = _radix_value(top2_keys_TGQ)
    group_scores_TG = tl.sum(top2_scores_TGQ, 2)
    # ATen sum starts from +0, including when both selected values are -0.
    group_scores_TG = tl.where(group_scores_TG == 0.0, 0.0, group_scores_TG)
    group_keys_TG = _radix_key(group_scores_TG)
    top_group_keys_TL = tl.topk(group_keys_TG, 4, dim=1)
    kth_group_keys_T = tl.min(top_group_keys_TL, 1)
    groups_above_TG = group_keys_TG > kth_group_keys_T[:, None]
    groups_equal_TG = group_keys_TG == kth_group_keys_T[:, None]
    num_groups_above_T = tl.sum(groups_above_TG.to(tl.int32), 1)
    equal_group_positions_TG = tl.cumsum(groups_equal_TG.to(tl.int32), 1)
    selected_groups_TG = groups_above_TG | (
        groups_equal_TG & (equal_group_positions_TG <= 4 - num_groups_above_T[:, None])
    )
    selected_experts_TGP = tl.broadcast_to(
        selected_groups_TG[:, :, None], (block_tokens, 8, 32)
    )
    selected_experts_TE = tl.reshape(selected_experts_TGP, (block_tokens, 256))
    # 0x007FFFFF is the ordered radix key of negative infinity.
    masked_keys_TE = tl.where(selected_experts_TE, score_keys_TE, 0x007FFFFF)
    masked_group_expert_keys_TGK = tl.where(
        selected_groups_TG[:, :, None], top_group_expert_keys_TGK, 0x007FFFFF
    )
    candidate_keys_TC = tl.reshape(masked_group_expert_keys_TGK, (block_tokens, 64))
    top_keys_TK = tl.topk(candidate_keys_TC, 8, dim=1)
    kth_score_keys_T = tl.min(top_keys_TK, 1)
    experts_above_TE = masked_keys_TE > kth_score_keys_T[:, None]
    experts_equal_TE = masked_keys_TE == kth_score_keys_T[:, None]
    # Match the installed PyTorch CUDA single-block radix topk order:
    # strictly greater keys in index order, then equal keys in index order.
    # See TensorTopK.cu and SortingRadixSelect.cuh in PyTorch 2.16.
    above_positions_TE = tl.cumsum(experts_above_TE.to(tl.int32), 1) - 1
    num_experts_above_T = tl.sum(experts_above_TE.to(tl.int32), 1)
    equal_positions_TE = (
        tl.cumsum(experts_equal_TE.to(tl.int32), 1) - 1 + num_experts_above_T[:, None]
    )
    output_positions_TE = tl.where(
        experts_above_TE, above_positions_TE, equal_positions_TE
    )
    output_mask_TE = (
        (token_ids[:, None] < num_tokens)
        & (experts_above_TE | experts_equal_TE)
        & (output_positions_TE < 8)
    )
    tl.store(
        selected_expert_ids + token_ids[:, None] * 8 + output_positions_TE,
        expert_ids[None, :].to(tl.int64),
        output_mask_TE,
    )


def select_experts_fused(
    scores_TE: torch.Tensor,
    expert_bias_E: torch.Tensor | None,
) -> torch.Tensor:
    """Fuse selection with the installed PyTorch 2.16 CUDA topk ordering.

    Supports contiguous FP32 CUDA scores [T, 256], 0 < T <= 2**23, and
    optional FP32 [256] expert bias. The production grouping is eight
    groups of 32 experts, selecting four groups and eight experts.
    These slice sizes use PyTorch's CUDA single-block radix topk path.
    The integer decision output has no autograd derivative.
    """
    if (
        scores_TE.ndim != 2
        or scores_TE.shape[1] != 256
        or not 0 < scores_TE.shape[0] <= 2**23
        or scores_TE.dtype != torch.float32
        or not scores_TE.is_cuda
        or not scores_TE.is_contiguous()
    ):
        raise ValueError(
            "Router input must be contiguous FP32 CUDA [T, 256], "
            "with 0 < T <= 2**23."
        )
    if expert_bias_E is not None and (
        expert_bias_E.shape != (256,)
        or expert_bias_E.dtype != torch.float32
        or expert_bias_E.device != scores_TE.device
        or not expert_bias_E.is_contiguous()
    ):
        raise ValueError("Expert bias must be contiguous FP32 CUDA [256].")
    num_tokens = scores_TE.shape[0]
    selected_expert_ids_TK = torch.empty(
        (num_tokens, 8), device=scores_TE.device, dtype=torch.int64
    )
    _select_experts_fused_kernel[(triton.cdiv(num_tokens, 4),)](
        scores_TE,
        expert_bias_E,
        selected_expert_ids_TK,
        num_tokens,
        expert_bias_E is not None,
        4,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return selected_expert_ids_TK
