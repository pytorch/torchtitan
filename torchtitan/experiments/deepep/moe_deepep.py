# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
MoE with DeepEP Integration

This module provides a MoE class that uses DeepEP for high-performance
expert-parallel communication.

Clean architecture:
- DeepEPDispatch: Minimal autograd wrapper for dispatch() only
- DeepEPCombine: Minimal autograd wrapper for combine() only
- MoEWithDeepEP: Normal PyTorch module - all operations are differentiable!
"""

import os
import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from typing import Optional, Tuple, List

from deep_ep import Buffer, EventOverlap
from torchtitan.models.moe.moe import MoEArgs, GroupedExperts, TokenChoiceTopKRouter, FeedForward
from torchtitan.tools.logging import logger

# Global buffer management
_deepep_buffers: dict[ProcessGroup, Buffer] = {}


def get_deepep_buffer(group: ProcessGroup, hidden_bytes: int) -> Buffer:
    """
    Get or create the DeepEP communication buffer.
    
    Args:
        group: The process group for expert parallelism
        hidden_bytes: Size of hidden dimension in bytes
    
    Returns:
        Buffer: The DeepEP communication buffer
    """
    global _deepep_buffers
    
    # Check if we already have a buffer for this EP group
    if group in _deepep_buffers:
        existing_buffer = _deepep_buffers[group]
        if existing_buffer.num_nvl_bytes >= hidden_bytes and existing_buffer.num_rdma_bytes >= hidden_bytes:
            return existing_buffer
    
    import torch.distributed as dist
    is_multinode = False
    local_world_size = 0
    num_nodes = 1
    rank = 0
    
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', torch.cuda.device_count()))
        is_multinode = world_size > local_world_size
        num_nodes = world_size // local_world_size if local_world_size > 0 else 1

    num_nvl_bytes, num_rdma_bytes = 0, 0
    
    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)
    
    # For multi-node with >8 ranks:
    # - internode_dispatch is used
    # - NVL buffers are for INTRA-node communication (within same node via NVLink)
    # - RDMA buffers are for INTER-node communication (across nodes via network)
    if is_multinode:
        if num_rdma_bytes == 0:
            num_rdma_bytes = hidden_bytes * group.size() * 8
            if rank == 0:
                logger.info(f"Allocated fallback RDMA buffer: {num_rdma_bytes} bytes")

    low_latency_mode = is_multinode or group.size() > 8
    
    ep_rank = dist.get_rank(group) if group else 0

    buffer = Buffer(
        group=group,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=low_latency_mode
    )
    
    _deepep_buffers[group] = buffer
    
    return buffer


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate hidden dimension size in bytes."""
    t = x[0] if isinstance(x, tuple) else x
    return t.size(-1) * max(t.element_size(), 2)


class DeepEPDispatch(torch.autograd.Function):
    """
    Minimal autograd wrapper for DeepEP's dispatch() operation.
    
    Forward: buffer.dispatch() - scatter tokens to expert ranks
    Backward: buffer.combine() - gather gradients back (reverses dispatch)
    """
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        buffer: Buffer,
        num_tokens_per_rank: torch.Tensor,
        num_tokens_per_rdma_rank: torch.Tensor,
        is_token_in_rank: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ):
        """
        Dispatch tokens to expert ranks.
        
        Args:
            x: Input tokens [num_tokens, hidden_dim]
            topk_idx: Expert indices [num_tokens, top_k]
            topk_weights: Router weights [num_tokens, top_k]
            buffer: DeepEP buffer
            (rest): Dispatch layout tensors from get_dispatch_layout()
        
        Returns:
            recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle
        """
        # DeepEP requires: x=bfloat16, topk_weights=float32
        x_bfloat16 = x.to(torch.bfloat16)
        topk_weights_float32 = topk_weights.to(torch.float32)
        
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, _ = \
            buffer.dispatch(
                x=x_bfloat16,
                topk_idx=topk_idx,
                topk_weights=topk_weights_float32,
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                async_finish=False,  # Async requires event state management in C++
                allocate_on_comm_stream=False,
            )
        
        # Save for backward
        ctx.handle = handle
        ctx.buffer = buffer
        ctx.input_dtype = x.dtype
        ctx.hidden_dim = x.shape[1]
        ctx.top_k = topk_weights.shape[1]
        
        return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle
    
    @staticmethod
    def backward(ctx, grad_recv_x, grad_recv_topk_idx, grad_recv_topk_weights, grad_num_recv, grad_handle):
        """
        Reverse dispatch using combine().
        
        Args:
            grad_recv_x: Gradient w.r.t. received tokens [num_recv_tokens, hidden_dim]
            grad_recv_topk_weights: Gradient w.r.t. received weights [num_recv_tokens, top_k]
        
        Returns:
            Gradients for (x, topk_idx, topk_weights, buffer, ...)
        """
        handle = ctx.handle
        buffer = ctx.buffer
        input_dtype = ctx.input_dtype
        hidden_dim = ctx.hidden_dim
        top_k = ctx.top_k
        
        if grad_recv_x is not None:
            grad_x_bfloat16 = grad_recv_x.to(torch.bfloat16)
            grad_x_combined, _, _ = buffer.combine(
                x=grad_x_bfloat16,
                handle=handle,
                async_finish=False,  # Async requires event state management in C++
                allocate_on_comm_stream=False,
            )
            grad_x = grad_x_combined.to(input_dtype)
        else:
            grad_x = None
        
        if grad_recv_topk_weights is not None:
            grad_recv_topk_weights_padded = torch.zeros(
                grad_recv_topk_weights.shape[0], hidden_dim,
                dtype=torch.bfloat16,
                device=grad_recv_topk_weights.device
            )
            grad_recv_topk_weights_padded[:, :top_k] = grad_recv_topk_weights.to(torch.bfloat16)
            
            grad_topk_weights_combined, _, _ = buffer.combine(
                x=grad_recv_topk_weights_padded,
                handle=handle,
                async_finish=False,
                allocate_on_comm_stream=False,
            )
            grad_topk_weights = grad_topk_weights_combined[:, :top_k].to(input_dtype)
        else:
            grad_topk_weights = None
        
        return grad_x, None, grad_topk_weights, None, None, None, None, None


class DeepEPCombine(torch.autograd.Function):
    """
    Minimal autograd wrapper for DeepEP's combine() operation.
    
    Forward: buffer.combine() - gather tokens back to original ranks
    Backward: buffer.dispatch() - scatter gradients (reverses combine)
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, handle, buffer: Buffer,
                topk_idx, topk_weights, num_tokens_per_rank,
                num_tokens_per_rdma_rank, is_token_in_rank, num_tokens_per_expert):
        """
        Combine tokens back to original ranks.
        
        Args:
            x: Tokens to combine [num_recv_tokens, hidden_dim]
            handle: Communication handle from dispatch
            buffer: DeepEP buffer
            (rest): Layout information (not used - handle contains comm pattern)
        
        Returns:
            combined: Combined tokens [num_original_tokens, hidden_dim]
        """
        #  Only supports bfloat16 for now
        x_bfloat16 = x.to(torch.bfloat16)
        
        combined, _, _ = buffer.combine(
            x=x_bfloat16,
            handle=handle,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        
        # Save for backward
        ctx.handle = handle
        ctx.buffer = buffer
        ctx.input_dtype = x.dtype
        # No need to save layout - handle contains the comm pattern
        
        return combined
    
    @staticmethod
    def backward(ctx, grad_combined):
        """
        Reverse combine using dispatch().
        
        Args:
            grad_combined: Gradient w.r.t. combined output [num_original_tokens, hidden_dim]
        
        Returns:
            Gradients for (x, handle, buffer, ...)
        """
        handle = ctx.handle
        buffer = ctx.buffer
        input_dtype = ctx.input_dtype
        
        grad_combined_bfloat16 = grad_combined.to(torch.bfloat16)
        
        grad_x, _, _, _, _, _ = buffer.dispatch(
            x=grad_combined_bfloat16,
            topk_idx=None,  # Must be None when handle is provided
            topk_weights=None,  # Must be None when handle is provided
            num_tokens_per_rank=None,
            num_tokens_per_rdma_rank=None,
            is_token_in_rank=None,
            num_tokens_per_expert=None,
            handle=handle,  # Reuse forward comm pattern
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        grad_x = grad_x.to(input_dtype)
        
        return grad_x, None, None, None, None, None, None, None, None


class MoEWithDeepEP(nn.Module):
    """
    Mixture of Experts with DeepEP communication.
    
    DeepEP parameters are excluded from FSDP wrapping (handled in parallelize.py).
    """
    
    def __init__(
        self, 
        router: nn.Module,
        experts: nn.Module,
        buffer: Buffer,
        num_experts: int,
        score_before_experts: bool = False,
        load_balance_coeff: float | None = None,
        ep_group: ProcessGroup | None = None,
        shared_experts: nn.Module | None = None,
    ):
        super().__init__()
        self.router = router
        self.experts = experts
        self.buffer = buffer
        self.num_experts = num_experts
        self.score_before_experts = score_before_experts
        self.ep_group = ep_group
        self.shared_experts = shared_experts
        
        self.load_balance_coeff = load_balance_coeff
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
        
        All intermediate operations use standard PyTorch so that autograd just works
        
        Args:
            x: Input tokens [bs, slen, hidden_dim] or [bs*slen, hidden_dim]
        
        Returns:
            Output tokens - same shape as input
        """
        input_shape = x.shape
        if x.dim() == 3:
            bs, slen, dim = x.shape
            x = x.view(-1, dim)  # Flatten to [bs*slen, dim]
        
        original_dtype = x.dtype
        
        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(x, self.expert_bias)
        
        if self.load_balance_coeff is not None:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)
        
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert_dispatch, is_token_in_rank, _ = \
            self.buffer.get_dispatch_layout(
                topk_idx=selected_experts_indices,
                num_experts=self.num_experts,
            )
        
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle = \
            DeepEPDispatch.apply(
                x,
                selected_experts_indices,
                top_scores,
                self.buffer,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                is_token_in_rank,
                num_tokens_per_expert_dispatch,
            )
        
        expert_output_combined = self._process_experts(
            recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list
        )
        
        if self.shared_experts is not None:
            output = self.shared_experts(x)  # x is still flattened [bs*slen, dim]
        else:
            output = torch.zeros_like(x)

        routed_output = DeepEPCombine.apply(
            expert_output_combined, 
            handle, 
            self.buffer,
            selected_experts_indices,
            top_scores,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert_dispatch,
        )
        output = output + routed_output.to(original_dtype)
        
        if len(input_shape) == 3:
            output = output.view(input_shape)
        
        return output
    
    def _process_experts(
        self,
        recv_x: torch.Tensor,
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        num_recv_tokens_per_expert_list: List[int],
    ) -> torch.Tensor:
        """
        Process tokens through local experts - all standard PyTorch ops.
        
        PyTorch autograd automatically handles all gradients here, including:
        - Sorting/unsorting
        - Expert forward/backward
        - Score multiplication
        - Per-token combination
        
        Args:
            recv_x: Received tokens [num_recv_tokens, hidden_dim]
            recv_topk_idx: Expert indices [num_recv_tokens, top_k]
            recv_topk_weights: Router weights [num_recv_tokens, top_k]
            num_recv_tokens_per_expert_list: Tokens per expert
        
        Returns:
            Combined expert outputs [num_recv_tokens, hidden_dim]
        """
        recv_topk_idx_flat = recv_topk_idx.view(-1)
        recv_topk_weights_flat = recv_topk_weights.view(-1)
        
        valid_mask = recv_topk_idx_flat >= 0
        valid_expert_ids = recv_topk_idx_flat[valid_mask]
        valid_weights = recv_topk_weights_flat[valid_mask]
        
        token_indices = torch.arange(
            recv_x.shape[0], device=recv_x.device
        ).unsqueeze(1).expand(-1, recv_topk_idx.shape[1]).reshape(-1)
        token_indices = token_indices[valid_mask]
        
        sorted_indices = torch.argsort(valid_expert_ids, stable=True)
        token_indices_sorted = token_indices[sorted_indices]
        valid_weights_sorted = valid_weights[sorted_indices]
        valid_expert_ids_sorted = valid_expert_ids[sorted_indices]
        
        recv_x_sorted = recv_x[token_indices_sorted]
        
        num_local_experts = self.experts.w1.shape[0]
        
        valid_expert_ids_local = valid_expert_ids_sorted
        
        # Count tokens only for LOCAL experts (using LOCAL IDs: 0-7)
        token_counts = torch.stack([
            (valid_expert_ids_local == i).sum()
            for i in range(num_local_experts)
        ]).to(torch.int32)
        
        if self.score_before_experts:
            recv_x_sorted = (recv_x_sorted.to(torch.float32) * valid_weights_sorted.unsqueeze(-1)).to(recv_x_sorted.dtype)

        # Run experts using GroupedExperts.forward() (PyTorch autograd handles backward automatically)
        expert_output = self.experts.forward(recv_x_sorted, token_counts)
        
        if not self.score_before_experts:
            expert_output = (expert_output.to(torch.float32) * valid_weights_sorted.unsqueeze(-1)).to(expert_output.dtype)
        
        unsorted_indices = torch.argsort(sorted_indices)
        expert_output_unsorted = expert_output[unsorted_indices]
        
        num_recv_tokens = recv_x.shape[0]
        hidden_dim = recv_x.shape[1]
        
        expert_output_combined = torch.zeros(
            num_recv_tokens, hidden_dim,
            dtype=recv_x.dtype, device=recv_x.device
        )
        
        expert_output_combined = expert_output_combined.scatter_add(
            0, 
            token_indices_sorted.unsqueeze(1).expand(-1, hidden_dim),
            expert_output_unsorted.to(recv_x.dtype)
        )
        
        return expert_output_combined


def create_deepep_moe(
    args: MoEArgs,
    ep_group: ProcessGroup,
    score_before_experts: bool = False,
) -> MoEWithDeepEP:
    """
    Create a MoEWithDeepEP module from MoEArgs.
    
    Args:
        args: MoE configuration
        ep_group: Expert parallelism process group
        score_before_experts: Whether to apply scores before or after experts
    
    Returns:
        MoEWithDeepEP module
    """
    router = TokenChoiceTopKRouter(
        dim=args.dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        score_func=args.score_func,
        route_norm=args.route_norm,
        route_scale=args.route_scale,
    )
    
    experts = GroupedExperts(
        dim=args.dim,
        hidden_dim=args.ffn_dim_multiplier * args.dim if args.ffn_dim_multiplier else args.dim * 4,
        num_experts=args.num_experts,
        use_grouped_mm=True,
    )
    
    hidden_bytes = args.dim * 2  # Assuming bfloat16
    buffer = get_deepep_buffer(ep_group, hidden_bytes)
    
    return MoEWithDeepEP(
        router=router,
        experts=experts,
        buffer=buffer,
        num_experts=args.num_experts,
        score_before_experts=score_before_experts,
    )
