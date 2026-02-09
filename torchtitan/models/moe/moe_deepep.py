# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE with DeepEP backend for efficient expert-parallel communication."""

import torch

from .moe import MoE, MoEArgs


class DeepEPMoE(MoE):
    """
    Mixture of Experts with DeepEP communication.

    Inherits from MoE but overrides forward() to pass routing info to experts,
    letting DeepEPExpertParallel hooks handle dispatch/combine.
    """

    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__(moe_args, dim, hidden_dim)
        # DeepEP doesn't use reorderer - routing handled by DeepEPExpertParallel
        self.reorderer = None  # pyrefly: ignore [bad-assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with DeepEP communication.

        DeepEPExpertParallel hooks intercept experts() call and handle
        dispatch/combine via deepep functions.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x, self.expert_bias
        )

        if self.load_balance_coeff is not None:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        # Call experts with routing info - hooks handle DeepEP dispatch/combine
        routed_output = self.experts(
            x,
            num_tokens_per_expert,
            selected_experts_indices,
            top_scores,
            self.experts.num_experts,
        )

        out = self.shared_experts(x) if self.shared_experts is not None else None

        if out is None:
            return routed_output.reshape(bs, slen, dim)
        return (out + routed_output).reshape(bs, slen, dim)
