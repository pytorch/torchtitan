# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


def generate_permute_indices(
    tokens_per_expert_group: torch.Tensor,  # shape [num_ranks * experts_per_rank]
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
):
    """
    Input tokens_per_expert_group is flattened as (e0,r0),(e1,r0),...,(e0,r1),(e1,r1),... (rank-major)
    i = r * E + e
      where E = experts_per_rank, R = num_ranks.

    Output segments are ordered as (e0,r0),(e0,r1),...,(e1,r0),(e1,r1),... (expert-major):
      This matches the original kernel loop nesting:
        for expert_id in [0..E):
          for r in [0..R):
            write a contiguous segment of length t_mat[r,e]

    so we need to
    reorder from
    (e0,r0),(e1,r0),...,(e0,r1),(e1,r1) ...

    into
    (e0,r0),(e0,r1),...,(e1,r0),(e1,r1) ...
    """
    device = tokens_per_expert_group.device
    # start_index_values[j] = where group j starts in the ORIGINAL
    # Here j is in rank-major order: j = r*E + e.
    start_index_values = tokens_per_expert_group.cumsum(0) - tokens_per_expert_group

    t_mat = tokens_per_expert_group.view(num_ranks, experts_per_rank)
    m_sizes = t_mat.sum(0)

    lens = t_mat.t().reshape(-1)
    # (experts_per_rank * num_ranks,) segment lengths in expert-major order
    starts = start_index_values.view(num_ranks, experts_per_rank).t().reshape(-1)
    # (experts_per_rank * num_ranks,) where that segment begins in the ORIGINAL

    out_starts = lens.cumsum(0) - lens
    # where the segment begins in the Permuted
    out_ends = out_starts + lens
    # where the segment ends in the Permuted token stream

    bias = starts - out_starts
    p = torch.arange(max_len, device=device)
    seg = torch.searchsorted(out_ends, p, right=True)

    permuted_indices = p + bias[seg]

    # m_offsets = m_sizes.cumsum(0)
    return permuted_indices, m_sizes


def _permute(x, num_tokens_per_expert, ep_degree, num_local_experts):
    max_len = x.shape[0]

    with torch.no_grad():
        (
            permuted_indices,
            num_tokens_per_expert,
        ) = generate_permute_indices(
            num_tokens_per_expert,
            num_local_experts,
            ep_degree,
            max_len,
        )

    input_shape = x.shape
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, num_tokens_per_expert


def _unpermute(
    out: torch.Tensor, input_shape, permuted_indices: torch.Tensor
) -> torch.Tensor:
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    return out_unpermuted
