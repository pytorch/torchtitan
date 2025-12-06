# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepEP Flexible Token Dispatcher for MoE.

This module provides a clean dispatcher architecture following Megatron-LM patterns:
- _DeepepManager: Low-level DeepEP communication manager
- MoEFlexTokenDispatcher: High-level flexible token dispatcher interface
"""

import os
from typing import Optional, Tuple
import torch
from torch.distributed import ProcessGroup
from torchtitan.tools.logging import logger

# Import DeepEP primitives
try:
    from deep_ep import Buffer
    HAS_DEEPEP = True
except ImportError:
    HAS_DEEPEP = False
    Buffer = None

# Global buffer cache
_deepep_buffers: dict[ProcessGroup, Buffer] = {}


def get_deepep_buffer(group: ProcessGroup, hidden_bytes: int) -> Buffer:
    """Get or create cached DeepEP buffer for the given process group."""
    if not HAS_DEEPEP:
        raise ImportError("DeepEP not installed. Install from https://github.com/deepseek-ai/deepep")
    
    global _deepep_buffers
    if group in _deepep_buffers:
        existing = _deepep_buffers[group]
        if existing.num_nvl_bytes >= hidden_bytes and existing.num_rdma_bytes >= hidden_bytes:
            return existing
    
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', torch.cuda.device_count()))
    is_multinode = world_size > local_world_size

    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)
    
    if is_multinode and num_rdma_bytes == 0:
        num_rdma_bytes = hidden_bytes * group.size() * 8
        if rank == 0:
            logger.info(f"Allocated fallback RDMA buffer: {num_rdma_bytes} bytes")

    low_latency_mode = is_multinode or group.size() > 8
    buffer = Buffer(group=group, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=num_rdma_bytes, low_latency_mode=low_latency_mode)
    _deepep_buffers[group] = buffer
    
    ep_rank = dist.get_rank(group) if group else 0
    if ep_rank == 0:
        logger.info(f"DeepEP Buffer: NVL={num_nvl_bytes/(1024**3):.2f}GB, RDMA={num_rdma_bytes/(1024**3):.2f}GB, "
                   f"mode={'low-latency' if low_latency_mode else 'high-throughput'}, multinode={is_multinode}")
    
    return buffer


def deepep_permute(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
):
    """Permute tokens by expert for grouped_mm. Returns (permuted_tokens, permuted_probs, sorted_indices)."""
    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    
    routing_map = routing_map.bool().T.contiguous()
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)
    permuted_probs = probs.T.contiguous().masked_select(routing_map) if probs is not None else None
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices


def deepep_unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    """Reverse permutation applied by deepep_permute using scatter_add."""
    _, hidden = restore_shape
    output_tokens = torch.zeros(restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device)
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output_tokens


class DeepEPDispatch(torch.autograd.Function):
    """Autograd wrapper for DeepEP dispatch (forward: scatter tokens, backward: gather gradients)."""
    
    @staticmethod
    def forward(ctx, x, topk_idx, topk_weights, buffer, num_tokens_per_rank,
                num_tokens_per_rdma_rank, is_token_in_rank, num_tokens_per_expert):
        """Dispatch tokens to expert ranks via buffer.dispatch()."""
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, _ = \
            buffer.dispatch(
                x=x.to(torch.bfloat16), topk_idx=topk_idx, topk_weights=topk_weights.to(torch.float32),
                num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
                async_finish=False, allocate_on_comm_stream=False,
            )
        
        num_recv_tokens_per_expert_tensor = torch.tensor(
            num_recv_tokens_per_expert_list, dtype=torch.int64, device='cpu'
        ).to(recv_x.device, non_blocking=True)
        
        ctx.handle, ctx.buffer, ctx.input_dtype, ctx.hidden_dim, ctx.top_k = handle, buffer, x.dtype, x.shape[1], topk_weights.shape[1]
        return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_tensor, handle
    
    @staticmethod
    def backward(ctx, grad_recv_x, grad_recv_topk_idx, grad_recv_topk_weights, grad_num_recv, grad_handle):
        """Reverse dispatch using buffer.combine()."""
        grad_x = None
        grad_topk_weights = None
        
        if grad_recv_x is not None:
            grad_x_combined, grad_token_probs, _ = ctx.buffer.combine(
                x=grad_recv_x.to(torch.bfloat16), handle=ctx.handle,
                topk_weights=grad_recv_topk_weights.float(),
                async_finish=False, allocate_on_comm_stream=False
            )
            grad_x = grad_x_combined.to(ctx.input_dtype)
            
            # If DeepEP returns gradients for token probs, use them
            if grad_token_probs is not None:
                grad_topk_weights = grad_token_probs.to(ctx.input_dtype)
        
        return grad_x, None, grad_topk_weights, None, None, None, None, None


class DeepEPCombine(torch.autograd.Function):
    """Autograd wrapper for DeepEP combine (forward: gather tokens, backward: scatter gradients)."""
    
    @staticmethod
    def forward(ctx, x, handle, buffer, topk_idx, topk_weights, num_tokens_per_rank,
                num_tokens_per_rdma_rank, is_token_in_rank, num_tokens_per_expert):
        """Combine tokens back to original ranks via buffer.combine()."""
        combined, _, _ = buffer.combine(x=x.to(torch.bfloat16), handle=handle, async_finish=False, allocate_on_comm_stream=False)
        ctx.handle, ctx.buffer, ctx.input_dtype = handle, buffer, x.dtype
        return combined
    
    @staticmethod
    def backward(ctx, grad_combined):
        """Reverse combine using buffer.dispatch()."""
        grad_x, _, _, _, _, _ = ctx.buffer.dispatch(
            x=grad_combined.to(torch.bfloat16), topk_idx=None, topk_weights=None,
            num_tokens_per_rank=None, num_tokens_per_rdma_rank=None, is_token_in_rank=None,
            num_tokens_per_expert=None, handle=ctx.handle, async_finish=False, allocate_on_comm_stream=False
        )
        return grad_x.to(ctx.input_dtype), None, None, None, None, None, None, None, None


class _DeepepManager:
    """Low-level manager for DeepEP communication (dispatch/combine with permutation)."""

    def __init__(
        self,
        num_local_experts: int,
        router_topk: int,
        num_experts: int,
        hidden_dim: int,
    ):
        if Buffer is None:
            raise ImportError("DeepEP is not installed. Install from https://github.com/deepseek-ai/deepep")
        
        self.num_local_experts = num_local_experts
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.token_indices: Optional[torch.Tensor] = None
        self.token_probs: Optional[torch.Tensor] = None
        self.handle = None
        self.dispatched_indices: Optional[torch.Tensor] = None
        self.dispatched_probs: Optional[torch.Tensor] = None
        self.tokens_per_expert: Optional[torch.Tensor] = None
        self.num_tokens_per_rank: Optional[torch.Tensor] = None
        self.num_tokens_per_rdma_rank: Optional[torch.Tensor] = None
        self.is_token_in_rank: Optional[torch.Tensor] = None
        self.num_tokens_per_expert_dispatch: Optional[torch.Tensor] = None

    def setup_metadata(self, token_indices: torch.Tensor, token_probs: torch.Tensor):
        """Setup routing metadata for dispatch."""
        if token_indices.dim() == 2:
            self.token_indices = token_indices.contiguous()
            self.token_probs = token_probs.contiguous()
        else:
            self.token_indices = token_indices.view(-1, self.router_topk).contiguous()
            self.token_probs = token_probs.view(-1, self.router_topk).contiguous()
        
        self.token_indices = self.token_indices.masked_fill(self.token_probs == 0, -1)

    def dispatch(self, hidden_states: torch.Tensor, group: ProcessGroup, 
                 async_finish: bool = False, allocate_on_comm_stream: bool = False) -> torch.Tensor:
        """Execute DeepEP dispatch (fused permute + all-to-all)."""
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                logger.warning("DeepEP requires float32 probs, set --moe-router-dtype=fp32")
            self.token_probs = self.token_probs.float()
        
        buffer = get_deepep_buffer(group, self.hidden_dim * 2)
        
        self.num_tokens_per_rank, self.num_tokens_per_rdma_rank, self.num_tokens_per_expert_dispatch, self.is_token_in_rank, _ = \
            buffer.get_dispatch_layout(topk_idx=self.token_indices, num_experts=self.num_experts)
        
        hidden_states, self.dispatched_indices, self.dispatched_probs, self.tokens_per_expert, self.handle = \
            DeepEPDispatch.apply(
                hidden_states, self.token_indices, self.token_probs, buffer,
                self.num_tokens_per_rank, self.num_tokens_per_rdma_rank,
                self.is_token_in_rank, self.num_tokens_per_expert_dispatch,
            )
        
        return hidden_states

    def _indices_to_multihot(self, indices: torch.Tensor, probs: torch.Tensor):
        """Convert topk indices to multihot format for permutation."""
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros((batch_size, self.num_local_experts), dtype=torch.long, device=indices.device)
        multihot_probs = torch.zeros((batch_size, self.num_local_experts), dtype=probs.dtype, device=indices.device)
        
        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(mask.sum(dim=1))
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        
        return multihot_routing_map.bool(), multihot_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        return self.tokens_per_expert

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Permute dispatched tokens into per-expert layout for grouped_mm."""
        self.dispatched_routing_map, self.dispatched_probs = self._indices_to_multihot(
            self.dispatched_indices, self.dispatched_probs
        )
        self.hidden_shape_before_permute = hidden_states.shape
        assert self.dispatched_probs.dtype == torch.float32, "DeepEP requires float32 probs"
        
        hidden_states, permuted_probs, self.reversed_mapping_for_combine = deepep_permute(
            hidden_states, self.dispatched_routing_map, probs=self.dispatched_probs
        )
        return hidden_states, permuted_probs

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reverse permutation to restore dispatch order."""
        return deepep_unpermute(hidden_states, self.reversed_mapping_for_combine, restore_shape=self.hidden_shape_before_permute)

    def combine(self, hidden_states: torch.Tensor, group: ProcessGroup,
                async_finish: bool = False, allocate_on_comm_stream: bool = False) -> torch.Tensor:
        """Execute DeepEP combine (fused unpermute + all-to-all)."""
        buffer = get_deepep_buffer(group, self.hidden_dim * 2)
        hidden_states = DeepEPCombine.apply(
            hidden_states, self.handle, buffer, self.token_indices, self.token_probs,
            self.num_tokens_per_rank, self.num_tokens_per_rdma_rank,
            self.is_token_in_rank, self.num_tokens_per_expert_dispatch,
        )
        self.handle = None
        return hidden_states


class MoEFlexTokenDispatcher:
    """High-level token dispatcher interface using DeepEP for efficient MoE communication."""

    def __init__(self, num_local_experts: int, router_topk: int, num_experts: int, hidden_dim: int):
        self.num_local_experts = num_local_experts
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self._comm_manager = _DeepepManager(num_local_experts, router_topk, num_experts, hidden_dim)
        self.hidden_shape: Optional[Tuple] = None

    def dispatch_preprocess(self, hidden_states: torch.Tensor, token_indices: torch.Tensor, 
                           token_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Setup routing metadata and flatten input."""
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self._comm_manager.setup_metadata(token_indices, token_probs)
        return hidden_states, self._comm_manager.token_probs

    def token_dispatch(self, hidden_states: torch.Tensor, group: ProcessGroup, probs: torch.Tensor = None,
                      async_finish: bool = False, allocate_on_comm_stream: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute fused dispatch (permute + all-to-all)."""
        dispatched_states = self._comm_manager.dispatch(hidden_states, group, async_finish, allocate_on_comm_stream)
        return dispatched_states, self._comm_manager.dispatched_probs

    def dispatch_postprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Organize dispatched tokens into per-expert layout for grouped_mm."""
        global_input_tokens, permuted_probs = self._comm_manager.get_permuted_hidden_states_by_experts(hidden_states)
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()
        return global_input_tokens, tokens_per_expert, permuted_probs

    def combine_preprocess(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Restore dispatch order before combine."""
        return self._comm_manager.get_restored_hidden_states_by_experts(hidden_states)

    def token_combine(self, hidden_states: torch.Tensor, group: ProcessGroup,
                     async_finish: bool = False, allocate_on_comm_stream: bool = False) -> torch.Tensor:
        """Execute fused combine (unpermute + all-to-all)."""
        return self._comm_manager.combine(hidden_states, group, async_finish, allocate_on_comm_stream)

    def combine_postprocess(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Restore original shape."""
        return hidden_states.view(self.hidden_shape)

