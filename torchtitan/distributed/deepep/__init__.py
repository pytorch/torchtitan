# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Expert Parallel Communication Backends for MoE Training.

This module provides two backends for expert-parallel all-to-all communication:

- **DeepEP**: Optimized for H100 with NVLink Switch (sparse routing format)
- **HybridEP**: Optimized for GB200 with NVLink72 (dense routing, TMA-accelerated)
"""

from typing import Any, Literal, Optional, Tuple

import torch
from torch.distributed import ProcessGroup


# ============================================================================
# Backend Selection for Unified Interface
# ============================================================================

_backend_mode: Literal["deepep", "hybridep"] = "deepep"


def configure_backend(
    backend: Literal["deepep", "hybridep"] = "deepep",
) -> None:
    """Configure which backend to use for the unified interface.
    
    Args:
        backend: "deepep" for H100/NVLink Switch, "hybridep" for GB200/NVLink72
    
    For HybridEP, SM configuration is read from environment variables:
        - HYBRIDEP_NUM_SMS_DISPATCH (default: 16)
        - HYBRIDEP_NUM_SMS_COMBINE (default: 16)
    """
    global _backend_mode
    _backend_mode = backend
    
    if backend == "hybridep":
        from . import hybridep
        hybridep.configure()


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    group: ProcessGroup,
    score_before_experts: bool = True,
    num_permuted_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """Dispatch tokens to experts via the configured backend.
    
    Returns:
        Tuple of (hidden_states, tokens_per_expert, state).
        The state object is backend-specific and should be passed to combine_tokens.
    """
    if _backend_mode == "hybridep":
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
        )
    else:
        from .deepep import dispatch_tokens as _dispatch
        # Original deepep doesn't have num_permuted_tokens parameter
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
) -> torch.Tensor:
    """Combine expert outputs via the configured backend.
    
    Args:
        hidden_states: Expert outputs
        state: State object returned by dispatch_tokens (backend-specific)
    """
    if _backend_mode == "hybridep":
        from . import hybridep
        return hybridep.combine_tokens(hidden_states, state)
    else:
        from .deepep import combine_tokens as _combine
        return _combine(hidden_states, state)


# ============================================================================
# Legacy Exports (backward compatibility with existing code)
# ============================================================================

from .deepep import (
    dispatch_tokens as deepep_dispatch_tokens,
    combine_tokens as deepep_combine_tokens,
)


__all__ = [
    "configure_backend",
    "dispatch_tokens",
    "combine_tokens",
    "hybridep",
    "deepep_dispatch_tokens",
    "deepep_combine_tokens",
]
