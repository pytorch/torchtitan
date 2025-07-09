# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed import all_reduce, get_world_size


class TensorParallelExpand(torch.autograd.Function):
    """
    Custom tensor parallel operation for expanding k_pe in DeepSeek V3 attention.
    
    This operation handles the gradient flow issue where k_pe.expand(-1, -1, n_local_heads, -1)
    needs proper gradient aggregation across tensor parallel ranks during backward pass.
    
    Forward: Performs standard expand operation
    Backward: Aggregates gradients across the expanded dimension and across TP ranks
    """
    
    @staticmethod
    def forward(ctx, input_tensor, expand_shape):
        """
        Forward pass: expand the input tensor to the target shape.
        
        Args:
            input_tensor: Input tensor to expand, shape (bsz, seqlen, 1, qk_rope_head_dim)
            expand_shape: Target shape tuple, e.g., (-1, -1, n_local_heads, -1)
        
        Returns:
            Expanded tensor with shape (bsz, seqlen, n_local_heads, qk_rope_head_dim)
        """
        # Save the original shape and expand dimension for backward pass
        ctx.original_shape = input_tensor.shape
        ctx.expand_shape = expand_shape
        
        # Find which dimension was expanded (should be dim=2 for n_local_heads)
        ctx.expanded_dim = None
        for i, (orig_size, expand_size) in enumerate(zip(ctx.original_shape, expand_shape)):
            if orig_size == 1 and expand_size != 1 and expand_size != -1:
                ctx.expanded_dim = i
                break
        
        # If expand_size is -1, infer from input shape
        if ctx.expanded_dim is None:
            for i, expand_size in enumerate(expand_shape):
                if expand_size == -1:
                    continue
                if i < len(ctx.original_shape) and ctx.original_shape[i] == 1:
                    ctx.expanded_dim = i
                    break
        
        # Perform the expand operation
        expanded_tensor = input_tensor.expand(expand_shape)
        return expanded_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: aggregate gradients across the expanded dimension and TP ranks.
        
        Args:
            grad_output: Gradient from the next layer, shape (bsz, seqlen, n_local_heads, qk_rope_head_dim)
        
        Returns:
            Aggregated gradient for the original input tensor
        """
        # Sum gradients across the expanded dimension (n_local_heads)
        if ctx.expanded_dim is not None:
            grad_input = torch.sum(grad_output, dim=ctx.expanded_dim, keepdim=True)
        else:
            grad_input = grad_output
        
        # All-reduce across tensor parallel ranks to aggregate gradients
        # This ensures that gradients from all TP ranks are properly combined
        # In torchtitan, we'll use a simple all_reduce without specifying the group
        # as the tensor parallel operations are handled by the DTensor framework
        if torch.distributed.is_initialized() and get_world_size() > 1:
            # Only perform all_reduce if we're in a distributed setting
            # The DTensor framework will handle the proper group communication
            all_reduce(grad_input)
        
        return grad_input, None


def tensor_parallel_expand(input_tensor, expand_shape):
    """
    Tensor parallel-aware expand operation for k_pe in DeepSeek V3 attention.
    
    This function should be used instead of the standard tensor.expand() when
    expanding k_pe to match n_local_heads in tensor parallel settings.
    
    Args:
        input_tensor: Input tensor to expand, typically k_pe with shape (bsz, seqlen, 1, qk_rope_head_dim)
        expand_shape: Target shape tuple, e.g., (-1, -1, n_local_heads, -1)
    
    Returns:
        Expanded tensor with proper gradient flow for tensor parallelism
    
    Example:
        # Instead of: k_pe.expand(-1, -1, n_local_heads, -1)
        # Use: tensor_parallel_expand(k_pe, (-1, -1, n_local_heads, -1))
    """
    return TensorParallelExpand.apply(input_tensor, expand_shape)
