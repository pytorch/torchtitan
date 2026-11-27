# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from torchtitan.models.common.activation import Softmax
from torchtitan.models.common.moe import TokenChoiceTopKRouter

# T = tokens, E = experts, K = top-k slots, L = softmax reduction lanes,
# N = elements of a tree sum.

# PyTorch's CUDA softmax uses its warp-per-row kernel up to this row size in
# FP32. The kernel below reproduces that kernel's reduction order.
_MAX_NUM_EXPERTS = 2048
# PyTorch reduces K <= 64 scores with one tree over at most 32 threads.
_MAX_TOP_K = 32


@triton.jit
def _tree_sum(v_TN, BLOCK_T: tl.constexpr, N: tl.constexpr, LOG2_N: tl.constexpr):
    for i in tl.static_range(LOG2_N):
        v_TN = tl.sum(tl.reshape(v_TN, (BLOCK_T, 2, N // (2 << i))), axis=1)
    return tl.reshape(v_TN, (BLOCK_T,))


@triton.jit
def _softmax_topk_kernel(
    gate_ptr,
    expert_bias_ptr,
    scores_ptr,
    expert_ids_ptr,
    routing_map_ptr,
    num_tokens,
    route_norm_epsilon,
    route_scale,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    L: tl.constexpr,
    LOG2_L: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOG2_BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    ROUTE_NORM: tl.constexpr,
    HAS_EXPERT_BIAS: tl.constexpr,
):
    rows_T = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    row_valid_T = rows_T < num_tokens
    lanes_L = tl.arange(0, L)
    max_T = tl.full((BLOCK_T,), float("-inf"), tl.float32)
    for it in tl.static_range(BLOCK_E // L):
        cols_L = it * L + lanes_L
        x_TL = tl.load(
            gate_ptr + rows_T[:, None] * E + cols_L[None, :],
            mask=row_valid_T[:, None] & (cols_L < E)[None, :],
            other=float("-inf"),
        )
        max_T = tl.maximum(max_T, tl.max(x_TL, axis=1))
    lane_sum_TL = tl.zeros((BLOCK_T, L), tl.float32)
    for it in tl.static_range(BLOCK_E // L):
        cols_L = it * L + lanes_L
        x_TL = tl.load(
            gate_ptr + rows_T[:, None] * E + cols_L[None, :],
            mask=row_valid_T[:, None] & (cols_L < E)[None, :],
            other=float("-inf"),
        )
        lane_sum_TL = lane_sum_TL + libdevice.exp(x_TL - max_T[:, None])
    sum_T = _tree_sum(lane_sum_TL, BLOCK_T, L, LOG2_L)

    cols_E = tl.arange(0, BLOCK_E)
    col_valid_E = cols_E < E
    x_TE = tl.load(
        gate_ptr + rows_T[:, None] * E + cols_E[None, :],
        mask=row_valid_T[:, None] & col_valid_E[None, :],
        other=float("-inf"),
    )
    probs_TE = tl.math.div_rn(libdevice.exp(x_TE - max_T[:, None]), sum_T[:, None])
    if HAS_EXPERT_BIAS:
        choice_TE = (
            probs_TE + tl.load(expert_bias_ptr + cols_E, mask=col_valid_E)[None, :]
        )
    else:
        choice_TE = probs_TE
    choice_TE = tl.where(choice_TE != choice_TE, float("inf"), choice_TE)
    choice_TE = tl.where(col_valid_E[None, :], choice_TE, float("-inf"))

    slots_K = tl.arange(0, BLOCK_K)
    picked_scores_TK = tl.zeros((BLOCK_T, BLOCK_K), tl.float32)
    picked_choices_TK = tl.zeros((BLOCK_T, BLOCK_K), tl.float32)
    picked_ids_TK = tl.zeros((BLOCK_T, BLOCK_K), tl.int32)
    best_T = max_T
    for k in tl.static_range(K):
        best_T = tl.max(choice_TE, axis=1)
        best_id_T = tl.argmax(choice_TE, axis=1, tie_break_left=True)
        is_best_TE = cols_E[None, :] == best_id_T[:, None]
        if HAS_EXPERT_BIAS:
            best_score_T = tl.sum(tl.where(is_best_TE, probs_TE, 0.0), axis=1)
        else:
            best_score_T = tl.where(best_T == float("inf"), float("nan"), best_T)
        is_slot_K = slots_K[None, :] == k
        picked_scores_TK = tl.where(is_slot_K, best_score_T[:, None], picked_scores_TK)
        picked_choices_TK = tl.where(is_slot_K, best_T[:, None], picked_choices_TK)
        picked_ids_TK = tl.where(is_slot_K, best_id_T[:, None], picked_ids_TK)
        choice_TE = tl.where(is_best_TE, float("-inf"), choice_TE)
    tl.store(
        routing_map_ptr + rows_T[:, None] * E + cols_E[None, :],
        (choice_TE == float("-inf")).to(tl.int8),
        mask=row_valid_T[:, None] & col_valid_E[None, :],
    )

    slot_valid_K = slots_K < K
    order_key_TK = picked_ids_TK + tl.where(
        picked_choices_TK == best_T[:, None], BLOCK_E, 0
    )
    order_key_TK = tl.where(slot_valid_K[None, :], order_key_TK, 2 * BLOCK_E + slots_K)
    position_TK = tl.sum(
        (order_key_TK[:, None, :] < order_key_TK[:, :, None]).to(tl.int32), axis=2
    )
    at_position_TKK = position_TK[:, :, None] == slots_K[None, None, :]
    scores_TK = tl.sum(
        tl.where(at_position_TKK, picked_scores_TK[:, :, None], 0.0), axis=1
    )
    expert_ids_TK = tl.sum(
        tl.where(at_position_TKK, picked_ids_TK[:, :, None], 0), axis=1
    )
    if ROUTE_NORM:
        denominator_T = (
            _tree_sum(
                tl.where(slot_valid_K[None, :], scores_TK, 0.0),
                BLOCK_T,
                BLOCK_K,
                LOG2_BLOCK_K,
            )
            + route_norm_epsilon
        )
        scores_TK = tl.math.div_rn(scores_TK, denominator_T[:, None])
    scores_TK = scores_TK * route_scale
    out_mask_TK = row_valid_T[:, None] & slot_valid_K[None, :]
    out_offsets_TK = rows_T[:, None] * K + slots_K[None, :]
    tl.store(scores_ptr + out_offsets_TK, scores_TK, mask=out_mask_TK)
    tl.store(
        expert_ids_ptr + out_offsets_TK, expert_ids_TK.to(tl.int64), mask=out_mask_TK
    )


def _softmax_topk(
    gate_TE: torch.Tensor,
    expert_bias_E: torch.Tensor | None,
    *,
    top_k: int,
    route_norm: bool,
    route_norm_epsilon: float,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return what ``TokenChoiceTopKRouter`` derives from FP32 gate logits with a
    softmax score, bitwise: scores ``(T, K)``, expert IDs ``(T, K)`` in
    ``torch.topk(sorted=False)`` order, and the routing map ``(T, E)``.
    """
    assert gate_TE.dtype == torch.float32 and gate_TE.is_contiguous()
    assert expert_bias_E is None or (
        expert_bias_E.dtype == torch.float32 and expert_bias_E.is_contiguous()
    )
    num_tokens, num_experts = gate_TE.shape
    scores_TK = gate_TE.new_empty((num_tokens, top_k))
    expert_ids_TK = gate_TE.new_empty((num_tokens, top_k), dtype=torch.int64)
    routing_map_TE = gate_TE.new_empty((num_tokens, num_experts), dtype=torch.bool)
    block_e = triton.next_power_of_2(num_experts)
    lanes = min(32, block_e)
    block_k = triton.next_power_of_2(top_k)
    block_t = 1 if num_tokens <= 1024 else 4
    _softmax_topk_kernel[(triton.cdiv(num_tokens, block_t),)](
        gate_TE,
        expert_bias_E,
        scores_TK,
        expert_ids_TK,
        routing_map_TE,
        num_tokens,
        route_norm_epsilon,
        route_scale,
        E=num_experts,
        K=top_k,
        BLOCK_E=block_e,
        L=lanes,
        LOG2_L=lanes.bit_length() - 1,
        BLOCK_K=block_k,
        LOG2_BLOCK_K=block_k.bit_length() - 1,
        BLOCK_T=block_t,
        ROUTE_NORM=route_norm,
        HAS_EXPERT_BIAS=expert_bias_E is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return scores_TK, expert_ids_TK, routing_map_TE


class FusedSoftmaxTopKRouter(TokenChoiceTopKRouter):
    """Inference-only ``TokenChoiceTopKRouter`` with a softmax score.

    Runs everything after the gate projection in one kernel and returns the
    stock router's outputs bitwise.

    TODO: The kernel reproduces undocumented ATen CUDA internals: the warp
    softmax's reduction order and its 2048-element threshold, the Reduce.cuh
    warp-shuffle order of the K-score sum, and the ``torch.topk(sorted=False)``
    output order, so a PyTorch update can break bitwise equality; the GPU test
    guards it. Upstream it (e.g. into Dist-MoE or a meta-pytorch kernel library)
    instead of keeping it here. No upstream issue is filed yet.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        def __post_init__(self) -> None:
            if torch.version.hip is not None:
                raise ValueError(
                    "FusedSoftmaxTopKRouter reproduces CUDA reduction orders only"
                )
            if type(self.score_func) is not Softmax.Config or self.score_func.dim != -1:
                raise ValueError(
                    "FusedSoftmaxTopKRouter requires a softmax over the last dim"
                )
            if self.num_experts > _MAX_NUM_EXPERTS:
                raise ValueError(
                    f"FusedSoftmaxTopKRouter supports at most {_MAX_NUM_EXPERTS} "
                    f"experts, got {self.num_experts}"
                )
            if not 1 <= self.top_k <= min(_MAX_TOP_K, self.num_experts):
                raise ValueError(
                    f"FusedSoftmaxTopKRouter supports 1 <= top_k <= "
                    f"min({_MAX_TOP_K}, num_experts), got {self.top_k}"
                )

    def forward(
        self,
        x_TD: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        *,
        padding_mask_T: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
        **router_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del padding_mask_T, aux_loss_denominator, router_kwargs
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("FusedSoftmaxTopKRouter is inference-only")
        return _softmax_topk(
            self.gate(x_TD),
            expert_bias_E,
            top_k=self.top_k,
            route_norm=self.route_norm,
            route_norm_epsilon=self.route_norm_epsilon,
            route_scale=self.route_scale,
        )
