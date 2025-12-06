# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE with DeepEP backend for efficient expert-parallel communication."""

import torch
import torch.nn as nn

from .moe import MoEArgs, GroupedExperts, TokenChoiceTopKRouter, FeedForward
from torchtitan.tools.logging import logger


class MoEWithDeepEP(nn.Module):
    """
    Mixture of Experts with DeepEP communication.
    
    FSDP manages all parameters (router, experts, shared_experts).
    DeepEP handles expert-parallel token communication only.
    
    Note: ep_group is passed at runtime during forward pass (via hooks),
    not stored during initialization.
    """
    
    def __init__(
        self,
        moe_args: MoEArgs,
        dim: int,
        hidden_dim: int,
        score_before_experts: bool = False,
    ):
        """
        Initialize MoEWithDeepEP.
        
        Args:
            moe_args: MoE configuration
            dim: Input/output dimension
            hidden_dim: Hidden dimension for expert feed-forward networks
            score_before_experts: Whether to apply scores before or after experts
        """
        super().__init__()
        
        # Store configuration
        num_experts = moe_args.num_experts
        self.num_experts = num_experts
        self.router_topk = moe_args.top_k
        self.hidden_dim = dim
        self.score_before_experts = score_before_experts
        
        # Create router
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
        )
        
        # Create experts
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=moe_args.use_grouped_mm,
        )
        
        # Create shared experts if specified
        self.shared_experts = None
        if moe_args.num_shared_experts > 0:
            self.shared_experts = FeedForward(
                dim=dim,
                hidden_dim=hidden_dim * moe_args.num_shared_experts,
            )
        
        # Create dispatcher (without ep_group - passed at runtime)
        from torchtitan.distributed.deepep import MoEFlexTokenDispatcher
        
        # Calculate num_local_experts (will be correct after EP sharding)
        # For now, assume it's the total - will work correctly after parallelization
        num_local_experts = num_experts  # Will be sharded by EP
        
        self.deepep_dispatcher = MoEFlexTokenDispatcher(
            num_local_experts=num_local_experts,
            router_topk=moe_args.top_k,
            num_experts=num_experts,
            hidden_dim=dim,
        )
        
        # Attach dispatcher to experts so ExpertParallelDeepEP can access it
        self.experts.deepep_dispatcher = self.deepep_dispatcher
        
        # Setup load balancing
        self.load_balance_coeff = moe_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None

        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )
        
        logger.info(
            f"MoEWithDeepEP initialized: num_experts={num_experts}, "
            f"router_topk={moe_args.top_k}, dim={dim}, hidden_dim={hidden_dim}"
        )
        
        
    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        """Initialize weights for experts and router."""
        import torch.distributed as dist
        import os
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)

        if buffer_device != self.tokens_per_expert.device:
            self.tokens_per_expert = self.tokens_per_expert.to(buffer_device)
            if self.expert_bias is not None:
                self.expert_bias = self.expert_bias.to(buffer_device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE with DeepEP communication.
        
        This method does routing and prepares tokens, then calls self.experts(). 
        When ExpertParallelDeepEP hooks are applied, they intercept the call to 
        self.experts() and handle all DeepEP dispatch/combine communication via
        the attached dispatcher (_DeepepManager handles all DeepEP calls).
        
        Args:
            x: Input tokens [bs, slen, hidden_dim] or [bs*slen, hidden_dim]
        
        Returns:
            Output tokens - same shape as input
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)  # Flatten to [bs*slen, dim]
        
        original_dtype = x.dtype
        
        # Route tokens to experts
        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(x, self.expert_bias)
        
        if self.load_balance_coeff is not None:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)
        
        # Setup dispatcher metadata (routing information) for hooks to use
        # The hooks will call token_dispatch/token_combine which need this metadata
        x_prep, probs_prep = self.deepep_dispatcher.dispatch_preprocess(
            x, selected_experts_indices, top_scores
        )
        
        # Call experts - ExpertParallelDeepEP hooks intercept here
        # Hooks use _DeepepManager (via dispatcher) to handle:
        #   1. token_dispatch: fused permute + all-to-all
        #   2. expert forward: run local experts
        #   3. token_combine: fused all-to-all + unpermute
        routed_output = self.experts(x_prep, num_tokens_per_expert)
        
        # Restore original shape (hooks don't call combine_postprocess)
        routed_output = self.deepep_dispatcher.combine_postprocess(routed_output)
        
        # Shared expert (execute to overlap with communication if needed)
        if self.shared_experts is not None:
            out = self.shared_experts(x)
        else:
            out = torch.zeros_like(x)
        
        # Combine routed expert output with shared expert output
        out = out + routed_output.to(original_dtype)
        out = out.reshape(bs, slen, dim)
        return out
