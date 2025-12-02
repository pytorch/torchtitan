# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fix: torch._grouped_mm backward produces garbage gradients for experts with 0 tokens.
Solution: Pad empty experts to 8 tokens (like Standard EP does in kernels.py:181).
"""

import torch

_ALIGNMENT = 8


def _patched_get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor):
    """Patched permute with padding for empty experts."""
    from torchtitan.distributed.deepep.fused_indices_converter import (
        fused_indices_to_multihot,
    )

    routing_map, probs = fused_indices_to_multihot(
        self.dispatched_indices, self.dispatched_probs, self.num_local_experts
    )
    self.dispatched_indices = None
    self.hidden_shape_before_permute = hidden_states.shape

    num_tokens, hidden = hidden_states.shape
    num_experts = routing_map.shape[1]
    device = hidden_states.device

    routing_map_t = routing_map.bool().T.contiguous()
    tokens_per_expert = routing_map_t.sum(dim=1)

    # Pad empty experts to alignment (matches Standard EP behavior in kernels.py:181)
    padded_tokens_per_expert = tokens_per_expert.clone()
    needs_padding = (tokens_per_expert < _ALIGNMENT).any().item()

    if needs_padding:
        # Append zero row for padding tokens to index
        hidden_with_zero = torch.cat(
            [hidden_states, hidden_states.new_zeros(1, hidden)]
        )

        all_sorted_indices = []
        all_probs = []

        for expert_idx in range(num_experts):
            expert_mask = routing_map_t[expert_idx]
            expert_token_indices = torch.arange(num_tokens, device=device)[expert_mask]
            expert_probs = probs[:, expert_idx][expert_mask]
            expert_count = expert_token_indices.shape[0]

            if expert_count < _ALIGNMENT:
                # Pad with zero-index tokens
                pad_count = _ALIGNMENT - expert_count
                pad_indices = torch.full(
                    (pad_count,), num_tokens, dtype=torch.long, device=device
                )
                pad_probs = torch.zeros(
                    pad_count, dtype=expert_probs.dtype, device=device
                )
                expert_token_indices = torch.cat([expert_token_indices, pad_indices])
                expert_probs = torch.cat([expert_probs, pad_probs])
                padded_tokens_per_expert[expert_idx] = _ALIGNMENT

            all_sorted_indices.append(expert_token_indices)
            all_probs.append(expert_probs)

        sorted_indices = torch.cat(all_sorted_indices)
        permuted_probs = torch.cat(all_probs)
        permuted_input = hidden_with_zero.index_select(0, sorted_indices)

        self._has_padding = True
    else:
        # Fast path - original logic, no padding needed
        token_indices = (
            torch.arange(num_tokens, device=device).unsqueeze(0).expand(num_experts, -1)
        )
        sorted_indices = token_indices.masked_select(routing_map_t)
        permuted_probs = probs.T.contiguous().masked_select(routing_map_t)
        permuted_input = hidden_states.index_select(0, sorted_indices)

        self._has_padding = False

    self.reversed_mapping_for_combine = sorted_indices
    self.tokens_per_expert = padded_tokens_per_expert

    if self.router_dtype == "fp64":
        permuted_probs = permuted_probs.to(torch.float64)

    return permuted_input, permuted_probs


def _patched_get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor):
    """Patched unpermute handling padding."""
    num_tokens, hidden = self.hidden_shape_before_permute
    device = hidden_states.device

    if getattr(self, "_has_padding", False):
        # Extra row for padding tokens
        output = torch.zeros(
            num_tokens + 1, hidden, dtype=hidden_states.dtype, device=device
        )
        output.scatter_add_(
            0,
            self.reversed_mapping_for_combine.unsqueeze(1).expand(-1, hidden),
            hidden_states,
        )
        return output[:-1]  # Remove padding row
    else:
        output = torch.zeros(
            num_tokens, hidden, dtype=hidden_states.dtype, device=device
        )
        output.scatter_add_(
            0,
            self.reversed_mapping_for_combine.unsqueeze(1).expand(-1, hidden),
            hidden_states,
        )
        return output


def apply_deepep_empty_expert_fix():
    """Apply monkey patches."""
    from torchtitan.distributed.deepep import utils

    utils.PrimusTurboDeepepManager.get_permuted_hidden_states_by_experts = (
        _patched_get_permuted_hidden_states_by_experts
    )
    utils.PrimusTurboDeepepManager.get_restored_hidden_states_by_experts = (
        _patched_get_restored_hidden_states_by_experts
    )
