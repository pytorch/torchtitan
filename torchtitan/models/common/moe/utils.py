# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import torch


def indices_padding_wrapper(func: Callable) -> Callable:
    """
    Wrapper to permute tokens from rank-major to expert-major order for
    torch._grouped_mm. The permutation is done without incurring
    synchronization between device and host.
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
