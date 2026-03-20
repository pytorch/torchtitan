# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Literal

import torch

from torchtitan.components.quantization import MXFP8_GROUP_ALIGNMENT_SIZE
from torchtitan.tools.utils import _round_up

TOKEN_GROUP_ALIGN_SIZE_M = 0
ValidTokenGroupAlignmentSize = Literal[0, 16, 32]


def set_token_group_alignment_size(
    alignment_size: ValidTokenGroupAlignmentSize,
) -> None:
    """
    Set the token group alignment size for token groups in MoE. This is implemented by
    padding each token group size to the next multiple of TOKEN_GROUP_ALIGN_SIZE_M.

    Valid values are: 0, 16, or 32.
    Different values are needed for different cases:

    * For bf16, no padding is needed.
    * For fp8, 16 byte alignment / 1 byte per elem = 16 elements.
    * For mxfp8, we need 32 (or block_size) because scaling block size is (1 x 32),
      so when doing per-token-group quantization on each logically distinct subtensor,
      we need to ensure the contracting dim is divisible by block_size.
      In the backward pass, grad_weight = (grad_output_t @ input).t() has gemm dims
      of (N, M) @ (M, K) so M is the contracting dim, and group offsets are along M,
      so we need 32 element alignment.
    """
    global TOKEN_GROUP_ALIGN_SIZE_M
    TOKEN_GROUP_ALIGN_SIZE_M = alignment_size


def get_token_group_alignment_size() -> ValidTokenGroupAlignmentSize:
    """Return the token group alignment size for FP8/MXFP8 grouped GEMMs."""
    return TOKEN_GROUP_ALIGN_SIZE_M


def get_mxfp8_pad_multiple() -> int | None:
    """Return the pad_multiple needed for MXFP8 grouped GEMMs, or None if not active.

    When TOKEN_GROUP_ALIGN_SIZE_M has been set to the MXFP8 block size (32),
    dispatch kernels must pad per-expert token groups to that multiple so the
    quantisation kernel's row-count requirement is satisfied.
    """
    return (
        MXFP8_GROUP_ALIGNMENT_SIZE
        if TOKEN_GROUP_ALIGN_SIZE_M == MXFP8_GROUP_ALIGNMENT_SIZE
        else None
    )


def maybe_align_num_tokens_for_mxfp8(num_tokens: int) -> int:
    """Round up token count only when MXFP8 group alignment is active."""
    if TOKEN_GROUP_ALIGN_SIZE_M != MXFP8_GROUP_ALIGNMENT_SIZE:
        return num_tokens
    return _round_up(num_tokens, MXFP8_GROUP_ALIGNMENT_SIZE)


def indices_padding_wrapper(func: Callable) -> Callable:
    """
    In order to use torch._grouped_mm, we need to make sure the number of
    tokens each expert gets is a multiple of TOKEN_GROUP_ALIGN_SIZE_M. The
    generate_permute_indices kernel also helps achieve this via padding,
    without incurring synchronization between device and host.
    """

    def wrapper(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        num_local_experts = w1.shape[0]
        ep_degree = num_tokens_per_expert.shape[0] // num_local_experts

        input_shape, x, permuted_indices, num_tokens_per_expert = _permute(
            x, num_tokens_per_expert, ep_degree, num_local_experts
        )

        out = func(w1, w2, w3, x, num_tokens_per_expert)

        out = _unpermute(out, input_shape, permuted_indices)

        return out

    return wrapper


def _permute(x, num_tokens_per_expert, ep_degree, num_local_experts):
    """
    Permute token groups from rank-major to expert-major, to prepare for a Grouped GEMM.
    """
    max_len = x.shape[0]

    with torch.no_grad():
        (permuted_indices, num_tokens_per_expert,) = _generate_permute_indices(
            num_tokens_per_expert,
            num_local_experts,
            ep_degree,
            max_len,
        )

    input_shape = x.shape
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, num_tokens_per_expert


# Source: https://github.com/pytorch/torchtitan/pull/2255
def _generate_permute_indices(
    tokens_per_expert_group: torch.Tensor,  # shape [num_ranks * experts_per_rank]
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
):
    """Generate permute indices for permute_rank_major_to_expert_major (no padding)."""
    device = tokens_per_expert_group.device

    # start_index_values[j] = where group j starts in the ORIGINAL
    # Here j is in rank-major order: j = r*E + e.
    # cumsum = [4, 6, 7, 8, 11, 13]
    # result = [0, 4, 6, 7, 8, 11]
    start_index_values = tokens_per_expert_group.cumsum(0) - tokens_per_expert_group

    t_mat = tokens_per_expert_group.view(num_ranks, experts_per_rank)
    m_sizes = t_mat.sum(0)
    lens = t_mat.t().reshape(-1)  # (R, E) -> (E, R) -> (E * R,)

    starts = start_index_values.view(num_ranks, experts_per_rank).t().reshape(-1)
    # = [0, 7, 4, 8, 6, 11]  where each segment starts in the ORIGINAL x

    out_starts = lens.cumsum(0) - lens
    out_ends = out_starts + lens

    bias = starts - out_starts
    p = torch.arange(max_len, device=device)
    seg = torch.searchsorted(out_ends, p, right=True)

    permuted_indices = p + bias[seg]

    return permuted_indices, m_sizes


def _unpermute(out, input_shape, permuted_indices, remove_padding_row=False):
    """Unpermute tokens from expert-major to rank-major order."""
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    if remove_padding_row:
        out_unpermuted = out_unpermuted[:-1]
    return out_unpermuted
