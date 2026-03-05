# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE with adaptive DeepEP + LLEP backend.

Uses DeepEP for balanced steps (fast NVL/RDMA transport) and falls back to
LLEP (LPT-based load balancing + NCCL all-to-all) for imbalanced steps.

The switching is handled by DeepEPLLEPExpertParallel hooks installed on the
GroupedExperts module. This MoE class provides the correct forward() flow:
- Passes 5-tuple routing info to experts (DeepEP-style)
- Overlaps shared_experts with combine communication (DeepEP path only)
- Calls sync_combine() which is a no-op when LLEP path was used
"""

import torch

from torchtitan.distributed.deepep import sync_combine

from .moe import MoE, MoEArgs


class DeepEPLLEPMoE(MoE):
    """
    Mixture of Experts with adaptive DeepEP + LLEP communication.

    Inherits from MoE but overrides forward() to:
    1. Pass 5-tuple routing info directly to experts (like DeepEPMoE)
    2. Let DeepEPLLEPExpertParallel hooks decide per-step which path to use
    3. Overlap shared_experts with combine when DeepEP path is active
    4. sync_combine() safely handles both paths (no-op for LLEP)

    The hooks return (bs*slen, dim) from both paths, so the MoE module
    doesn't need to know which path was used.
    """

    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__(moe_args, dim, hidden_dim)
        # DeepEP+LLEP doesn't use reorderer - routing handled by hooks
        self.reorderer = None  # pyrefly: ignore [bad-assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive DeepEP/LLEP communication.

        DeepEPLLEPExpertParallel hooks intercept experts() call and handle
        dispatch/combine. When DeepEP path is used, combine runs async,
        allowing shared_experts to overlap. When LLEP path is used, combine
        is synchronous and completes before shared_experts.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x, self.expert_bias
        )

        if self.load_balance_coeff is not None:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        # Call experts with routing info - hooks handle dispatch/combine.
        # DeepEP path: combine returns asynchronously
        # LLEP path: combine is synchronous, returns (bs*slen, dim) directly
        routed_output = self.experts(
            x,
            num_tokens_per_expert,
            selected_experts_indices,
            top_scores,
            self.experts.num_experts,
        )

        # shared_experts can overlap with DeepEP async combine.
        # For LLEP path, this just runs after combine is already done.
        out = self.shared_experts(x) if self.shared_experts is not None else None

        # Sync DeepEP combine if pending. No-op if LLEP path was used
        # (no pending event) or if no shared_experts ran.
        sync_combine()

        if out is None:
            return routed_output.reshape(bs, slen, dim)
        return (out + routed_output).reshape(bs, slen, dim)
