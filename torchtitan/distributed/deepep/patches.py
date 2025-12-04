# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fix: torch._grouped_mm backward produces garbage gradients for experts with 0 tokens.
Solution: Pad empty experts to 8 tokens (like Standard EP does in kernels.py:181).

Strategy:
- Uses ORIGINAL kernel (no overhead) + compute tokens_per_expert from routing_map
- Fast path (no padding): When all experts have >= 8 tokens, use original logic
- Padding path: When some experts have < 8 tokens, do explicit padding
- With load balancing enabled, skip the expensive check (assumes all experts >= 8 tokens)
"""

import torch

from torchtitan.distributed.deepep.fused_indices_converter import (
    ALIGNMENT_M,
    fused_indices_to_multihot,  # Use ORIGINAL kernel - no overhead
)

# Global flag to skip padding check when load balancing is enabled
# Set this to True when using --training.debug_moe_force_load_balance
_ASSUME_LOAD_BALANCED = False


def set_assume_load_balanced(value: bool):
    """Set whether to assume load balancing is enabled (skip padding check)."""
    global _ASSUME_LOAD_BALANCED
    _ASSUME_LOAD_BALANCED = value


def _patched_get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor):
    """Patched permute with padding for empty experts.

    Uses ORIGINAL fused_indices_to_multihot kernel (no overhead) and computes
    tokens_per_expert from routing_map to check if padding is needed.
    """
    # Use ORIGINAL kernel - no overhead
    routing_map, probs = fused_indices_to_multihot(
        self.dispatched_indices, self.dispatched_probs, self.num_local_experts
    )

    num_tokens, hidden = hidden_states.shape
    num_experts = routing_map.shape[1]
    topk = self.dispatched_indices.shape[1]
    device = hidden_states.device

    self.hidden_shape_before_permute = hidden_states.shape

    # Compute tokens_per_expert (needed for grouped_mm later)
    # Keep as int64 from sum, convert to int32 only if needed for grouped_mm
    tokens_per_expert = routing_map.sum(dim=0)

    # Check if padding is needed
    if _ASSUME_LOAD_BALANCED:
        # Fast path: assume load balancing ensures all experts have >= 8 tokens
        # Skip the expensive GPU->CPU sync
        needs_padding = False
    else:
        # Full check with GPU->CPU sync (slower but correct for all cases)
        min_tokens = tokens_per_expert.min().item()
        needs_padding = min_tokens < ALIGNMENT_M

    # Convert to int32 (grouped_mm requirement) - do after check to avoid unnecessary conversion
    tokens_per_expert = tokens_per_expert.to(torch.int32)

    if not needs_padding:
        # Fast path - no padding needed (common case with load balancing)
        routing_map_t = routing_map.bool().T.contiguous()
        token_indices = (
            torch.arange(num_tokens, device=device).unsqueeze(0).expand(num_experts, -1)
        )
        sorted_indices = token_indices.masked_select(routing_map_t)
        permuted_probs = probs.T.contiguous().masked_select(routing_map_t)
        permuted_input = hidden_states.index_select(0, sorted_indices)

        self.reversed_mapping_for_combine = sorted_indices
        self.tokens_per_expert = tokens_per_expert
        self._has_padding = False
    else:
        # Padding path - slower but correct for experts with < 8 tokens
        # Compute padded sizes: clamp to min 8, round up to multiple of 8
        tokens_clamped = torch.clamp_min(tokens_per_expert, ALIGNMENT_M)
        m_sizes = ((tokens_clamped + ALIGNMENT_M - 1) // ALIGNMENT_M * ALIGNMENT_M).to(
            torch.int32
        )
        m_offsets = torch.cumsum(m_sizes, dim=0).to(torch.int32)

        total_padded = m_offsets[-1].item()
        write_offsets = m_offsets - m_sizes  # start of each expert segment

        # For real tokens: use routing_map to find which tokens go to which experts
        routing_map_t = routing_map.bool().T.contiguous()
        token_indices = (
            torch.arange(num_tokens, device=device).unsqueeze(0).expand(num_experts, -1)
        )

        # Get the real token indices per expert (flattened)
        real_sorted_indices = token_indices.masked_select(routing_map_t)
        real_probs = probs.T.contiguous().masked_select(routing_map_t)

        # Build position within each expert using cumcount
        expert_ids_for_tokens = torch.repeat_interleave(
            torch.arange(num_experts, device=device), tokens_per_expert.long()
        )

        # Vectorized cumcount within each segment:
        # positions_in_expert[i] = count of same expert_id before position i
        raw_counts_cumsum = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                torch.cumsum(tokens_per_expert.long(), dim=0)[:-1],
            ]
        )
        global_indices = torch.arange(len(expert_ids_for_tokens), device=device)
        positions_in_expert = global_indices - raw_counts_cumsum[expert_ids_for_tokens]

        # Write positions in padded layout
        write_pos = write_offsets[expert_ids_for_tokens] + positions_in_expert

        # Build padded sorted_indices (padding positions point to num_tokens = zero row)
        new_sorted_indices = torch.full(
            (total_padded,), num_tokens, dtype=torch.long, device=device
        )
        new_sorted_indices.scatter_(0, write_pos, real_sorted_indices)

        # Build padded probs (padding has 0 prob)
        new_probs = torch.zeros(total_padded, dtype=probs.dtype, device=device)
        new_probs.scatter_(0, write_pos, real_probs)

        # Permute with zero row appended for padding
        hidden_with_zero = torch.cat(
            [hidden_states, hidden_states.new_zeros(1, hidden)]
        )
        permuted_input = hidden_with_zero.index_select(0, new_sorted_indices)

        self.reversed_mapping_for_combine = new_sorted_indices
        self.tokens_per_expert = m_sizes  # Use padded counts for grouped_mm
        self._has_padding = True
        permuted_probs = new_probs

    self.dispatched_indices = None

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


def apply_deepep_empty_expert_fix(assume_load_balanced: bool = False):
    """Apply monkey patches to fix empty expert gradient bug.

    Args:
        assume_load_balanced: If True, skip the expensive padding check (assumes all
            experts have >= 8 tokens due to load balancing). This gives ~6% overhead
            instead of ~12% overhead. Set to True when using
            --training.debug_moe_force_load_balance.
    """
    global _ASSUME_LOAD_BALANCED
    _ASSUME_LOAD_BALANCED = assume_load_balanced

    from torchtitan.distributed.deepep import utils

    utils.PrimusTurboDeepepManager.get_permuted_hidden_states_by_experts = (
        _patched_get_permuted_hidden_states_by_experts
    )
    utils.PrimusTurboDeepepManager.get_restored_hidden_states_by_experts = (
        _patched_get_restored_hidden_states_by_experts
    )
