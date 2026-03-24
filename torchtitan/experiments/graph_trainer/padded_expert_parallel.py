# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PaddedExpertParallel: graph-compilation-compatible Expert Parallel.

Temporary workaround that will be removed when the compiler supports
non-even routing (data-dependent split sizes in all-to-all).
"""

import torch
from torch import Tensor
from torch.distributed._functional_collectives import all_to_all_single_autograd
from torch.distributed.tensor import DeviceMesh
from torch.nn import Module

from torchtitan.distributed.expert_parallel import ExpertParallel


class PaddedExpertParallel(ExpertParallel):
    """Expert Parallel with fixed-size all-to-all for graph compilation compatibility.

    Standard ExpertParallel uses data-dependent split sizes for all-to-all
    (via .tolist()) and dynamic permutation indices (via generate_permute_indices),
    both of which introduce _local_scalar_dense ops that Inductor cannot compile.

    This variant:
    1. Pads routed_input so each rank sends/receives equal-sized chunks
    2. Uses equal-split all-to-all (split_sizes=None)
    3. Replaces dynamic _permute with a static reshape+transpose for
       rank-grouped → expert-grouped reordering

    Requires balanced routing (e.g. _debug_force_load_balance=True) so that
    each EP rank sends the same number of tokens to every other rank.
    """

    def _token_dispatch(
        self, mod: Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        total_routed_tokens = routed_input.shape[0]
        dim = routed_input.shape[1]
        # Pad to be divisible by ep_degree for equal-split all-to-all
        tokens_per_rank = (total_routed_tokens + ep_degree - 1) // ep_degree
        padded_total = tokens_per_rank * ep_degree
        if padded_total > total_routed_tokens:
            pad = routed_input.new_zeros(
                padded_total - total_routed_tokens, dim
            )
            routed_input = torch.cat([routed_input, pad], dim=0)

        # Save for combine
        self._unpadded_total = total_routed_tokens
        self._tokens_per_rank = tokens_per_rank
        self._ep_degree = ep_degree
        self._num_local_experts = num_local_experts

        # Equal-split all-to-all for the actual token data
        routed_input = all_to_all_single_autograd(
            routed_input,
            None,  # equal output splits
            None,  # equal input splits
            device_mesh.get_group(),
        )

        # After all-to-all, tokens are arranged as:
        #   [rank0_expert0..., rank0_expert1..., rank1_expert0..., rank1_expert1..., ...]
        # With balanced routing, each (rank, expert) chunk has exactly
        # tokens_per_rank // num_local_experts tokens.
        # We want expert-grouped layout:
        #   [expert0_from_rank0, expert0_from_rank1, ..., expert1_from_rank0, ...]
        # This is a reshape(ep_degree, num_local_experts, tpe, dim) + transpose(0,1).
        tpe = tokens_per_rank // num_local_experts  # tokens per expert per rank
        self._tokens_per_expert_per_rank = tpe

        routed_input = (
            routed_input.view(ep_degree, num_local_experts, tpe, dim)
            .transpose(0, 1)
            .contiguous()
            .view(-1, dim)
        )

        # Compute static num_tokens_per_expert for grouped_mm offsets.
        # Each local expert gets tpe * ep_degree tokens total.
        # Return pre-computed cumsum offsets (int32) directly.
        # This avoids an Inductor bug where cumsum(..., dtype=int32) produces
        # int64 during post-grad fake tensor propagation.
        # _run_experts_grouped_mm computes cumsum on this; since it's already
        # the cumsum, we store the per-expert counts but mark them for direct use.
        tokens_per_expert = tpe * ep_degree
        num_tokens_per_expert_local = torch.full(
            (num_local_experts,),
            tokens_per_expert,
            dtype=torch.int32,
            device=routed_input.device,
        )

        return routed_input, num_tokens_per_expert_local

    def _token_combine(
        self, mod: Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        dim = routed_output.shape[1]
        ep_degree = self._ep_degree
        num_local_experts = self._num_local_experts
        tpe = self._tokens_per_expert_per_rank

        # Reverse the expert-grouped → rank-grouped transpose
        routed_output = (
            routed_output.view(num_local_experts, ep_degree, tpe, dim)
            .transpose(0, 1)
            .contiguous()
            .view(-1, dim)
        )

        # Reverse all-to-all with equal splits
        routed_output = all_to_all_single_autograd(
            routed_output,
            None,  # equal output splits
            None,  # equal input splits
            device_mesh.get_group(),
        )

        # Remove padding added during dispatch
        if routed_output.shape[0] > self._unpadded_total:
            routed_output = routed_output[: self._unpadded_total]

        return routed_output
