# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Expert Parallel Communication Backends for MoE Training.

- DeepEP: Support for NVLink8 (H100 / B200)
- HybridEP: Optimized for GB200/NVLink72

Backend is selected via job_config.parallelism.expert_parallel_comm_backend.
HybridEP config is in job_config.parallelism.hybridep.
"""

from typing import Any, Literal, Optional, Tuple

import torch
from torch.distributed import ProcessGroup


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    group: ProcessGroup,
    score_before_experts: bool = True,
    backend: Literal["deepep", "hybridep"] = "deepep",
    # HybridEP-specific (ignored for DeepEP)
    num_permuted_tokens: Optional[int] = None,
    capacity_factor: float = 1.0,
    num_sms_dispatch: int = 16,
    num_sms_combine: int = 16,
    pad_multiple: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """Dispatch tokens to experts via specified backend.
    
    Returns: (permuted_hidden, tokens_per_expert, state)
    """
    if backend == "hybridep":
        from . import hybridep
        return hybridep.dispatch_tokens(
            hidden_states=hidden_states,
            selected_experts_indices=selected_experts_indices,
            top_scores=top_scores,
            num_local_experts=num_local_experts,
            num_experts=num_experts,
            group=group,
            score_before_experts=score_before_experts,
            num_permuted_tokens=num_permuted_tokens,
            capacity_factor=capacity_factor,
            num_sms_dispatch=num_sms_dispatch,
            num_sms_combine=num_sms_combine,
            pad_multiple=pad_multiple,
        )
    else:
        from .deepep import dispatch_tokens as _dispatch
        return _dispatch(
            hidden_states=hidden_states,
            selected_experts_indices=selected_experts_indices,
            top_scores=top_scores,
            num_local_experts=num_local_experts,
            num_experts=num_experts,
            group=group,
            score_before_experts=score_before_experts,
        )


def combine_tokens(
    hidden_states: torch.Tensor,
    state: Any,
    backend: Literal["deepep", "hybridep"] = "deepep",
) -> torch.Tensor:
    """Combine expert outputs via specified backend."""
    if backend == "hybridep":
        from . import hybridep
        return hybridep.combine_tokens(hidden_states, state)
    else:
        from .deepep import combine_tokens as _combine
        return _combine(hidden_states, state)


__all__ = ["dispatch_tokens", "combine_tokens"]
