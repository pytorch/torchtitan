"""
Batch-invariant operations with backward pass support.

This module adds gradient support to vLLM's deterministic batch_invariant mode
by registering backward operations that also use vLLM's deterministic kernels.

Key architecture:
- Forward: Uses vLLM's batch_invariant Triton kernels (deterministic)
- Backward: Also uses vLLM's batch_invariant kernels (deterministic)

This achieves bitwise-deterministic RL training where both rollouts (forward)
and training (forward + backward) produce identical results.

Usage:
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance
    from batch_invariant_backward import patch_batch_invariant_with_gradients

    # Initialize vLLM's deterministic mode first
    init_batch_invariance()

    # Then patch in gradient support
    patch_batch_invariant_with_gradients()

    # Now all operations are deterministic AND support gradients
    model = MyModel()
    output = model(input)  # deterministic forward
    loss = compute_loss(output)
    loss.backward()  # gradients work with deterministic backward!
"""

import torch
from torch.autograd import Function


# ============================================================================
# Custom autograd Functions for vLLM operations
# ============================================================================

class FlashAttn3Function(Function):
    """
    Autograd function for Flash Attention 3 with proper backward support.
    """

    @staticmethod
    def forward(
        ctx,
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_left,
        window_right,
        softcap,
        scheduler_metadata,
        num_splits,
    ):
        """
        Forward pass using vLLM's FA3 CUDA kernel.
        """
        out, softmax_lse, _, _ = torch.ops._vllm_fa3_C.fwd(
            q, k, v,
            None, None,       # k_new, v_new
            None,             # q_v
            None,             # out
            cu_seqlens_q,
            cu_seqlens_k,
            None,             # cu_seqlens_k_new
            None, None,       # seqused_q, seqused_k
            max_seqlen_q, max_seqlen_k,
            None,             # block_table
            None,             # kv_batch_idx
            None,             # leftpad_k
            None, None, None, # rotary_cos, rotary_sin, seqlens_rotary
            None, None, None, # q_descale, k_descale, v_descale
            softmax_scale,
            causal,
            window_left, window_right,
            softcap,
            True,             # rotary_interleaved
            scheduler_metadata,
        )

        # Save tensors needed for backward
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_left = window_left
        ctx.window_right = window_right
        ctx.softcap = softcap
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.scheduler_metadata = scheduler_metadata

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using vLLM's FA3 CUDA backward kernel.
        """
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors

        # Allocate gradient tensors
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)

        # Call FA3 backward kernel
        torch.ops._vllm_fa3_C.bwd(
            grad_output,
            q, k, v,
            out,
            softmax_lse,
            grad_q, grad_k, grad_v,
            cu_seqlens_q,
            cu_seqlens_k,
            None,             # cu_seqlens_k_new
            None, None,       # seqused_q, seqused_k
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            None,             # block_table
            None,             # kv_batch_idx
            None,             # leftpad_k
            None, None, None, # rotary_cos, rotary_sin, seqlens_rotary
            None, None, None, # dq_accum, q_descale, k_descale, v_descale
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_left, ctx.window_right,
            ctx.softcap,
            False,            # deterministic
            True,             # rotary_interleaved
            ctx.scheduler_metadata,
        )

        # Return gradients for all forward inputs (None for non-tensor args)
        return grad_q, grad_k, grad_v, None, None, None, None, None, None, None, None, None, None, None


class RMSNormFunction(Function):
    """
    Autograd function for RMS normalization using vLLM's Triton kernel in forward
    and batch-invariant operations in backward.
    """

    @staticmethod
    def forward(ctx, input, weight, eps):
        """
        Forward pass using vLLM's rms_norm Triton kernel.

        Args:
            input: Input tensor [*, hidden_size]
            weight: Weight tensor [hidden_size]
            eps: Epsilon for numerical stability

        Returns:
            output: Normalized and scaled tensor [*, hidden_size]
        """
        from vllm.model_executor.layers.batch_invariant import rms_norm as vllm_rms_norm

        # Use vLLM's Triton kernel for forward (deterministic)
        output = vllm_rms_norm(input, weight, eps)

        # Save for backward
        ctx.save_for_backward(input, weight)
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using batch-invariant PyTorch operations.

        Returns:
            (grad_input, grad_weight, None)
        """
        input, weight = ctx.saved_tensors
        eps = ctx.eps

        # Compute forward pass values needed for backward
        # variance = mean(x^2) along last dim
        variance = (input * input).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(variance + eps)
        x_norm = input / rms

        # Gradient w.r.t. weight
        # grad_weight = sum(grad_output * x_norm) over all dims except last
        grad_weight = (grad_output * x_norm).sum(dim=tuple(range(grad_output.ndim - 1)))

        # Gradient w.r.t. input
        # grad_x_norm = grad_output * weight
        grad_x_norm = grad_output * weight

        # grad_x = (grad_x_norm - mean(grad_x_norm * x_norm) * x_norm) / rms
        mean_term = (grad_x_norm * x_norm).mean(dim=-1, keepdim=True)
        grad_input = (grad_x_norm - mean_term * x_norm) / rms

        return grad_input, grad_weight, None


# ============================================================================
# Backward operation implementations for autograd
# ============================================================================

def matmul_backward_impl(grad_output, self, other, output_mask):
    """
    Backward pass for matmul: y = matmul(a, b)
    Returns: (grad_a, grad_b)

    Args:
        grad_output: Gradient from downstream
        self: First input tensor (a)
        other: Second input tensor (b)
        output_mask: List of bools indicating which gradients to compute [self, other]

    grad_a = grad_output @ b.T
    grad_b = a.T @ grad_output

    Uses torch.matmul which is overridden by vLLM's batch_invariant mode!
    """
    grad_self = grad_other = None

    # output_mask is a list [compute_grad_self, compute_grad_other]
    compute_grad_self = output_mask[0] if len(output_mask) > 0 else True
    compute_grad_other = output_mask[1] if len(output_mask) > 1 else True

    if compute_grad_self:
        # grad_self = grad_output @ other.T
        if other.ndim == 2:
            grad_self = torch.matmul(grad_output, other.t())
        elif other.ndim == 3:
            grad_self = torch.matmul(grad_output, other.transpose(-2, -1))
        else:
            grad_self = torch.matmul(grad_output, other.transpose(-2, -1))

    if compute_grad_other:
        # grad_other = self.T @ grad_output
        if self.ndim == 2:
            grad_other = torch.matmul(self.t(), grad_output)
        elif self.ndim == 3:
            grad_other = torch.matmul(self.transpose(-2, -1), grad_output)
        else:
            grad_other = torch.matmul(self.transpose(-2, -1), grad_output)

    return grad_self, grad_other


def linear_backward_impl(grad_output, input, weight, output_mask):
    """
    Backward pass for linear: y = input @ weight.T + bias
    Returns: (grad_input, grad_weight, grad_bias)

    Args:
        grad_output: Gradient from downstream (actually the saved input!)
        input: Input tensor (actually grad_output!)
        weight: Weight tensor
        output_mask: List of bools indicating which gradients to compute [input, weight, bias]

    PyTorch passes args in weird order: (saved_input, grad_output, weight, output_mask)
    So we swap the first two args in our implementation.
    """
    # Swap: PyTorch passes (saved_input, grad_output, ...) but we want (grad_output, input, ...)
    input, grad_output = grad_output, input

    grad_input = grad_weight = grad_bias = None

    # output_mask is a list [compute_grad_input, compute_grad_weight, compute_grad_bias]
    compute_grad_input = output_mask[0] if len(output_mask) > 0 else True
    compute_grad_weight = output_mask[1] if len(output_mask) > 1 else True
    compute_grad_bias = output_mask[2] if len(output_mask) > 2 else True

    if compute_grad_input:
        # grad_input = grad_output @ weight
        grad_input = torch.matmul(grad_output, weight)

    if compute_grad_weight:
        # PyTorch linear: y = x @ W.T + b where W is [out, in]
        # Backward: grad_W = grad_y.T @ x
        # grad_output: (batch, out), input: (batch, in)
        # grad_output.T @ input: (out, batch) @ (batch, in) = (out, in) ✓

        # Handle multi-dimensional inputs
        if input.ndim == 3:
            # Reshape for matmul: (batch, seq, in) -> (batch*seq, in)
            input_2d = input.reshape(-1, input.shape[-1])
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
            # grad_output_2d: (batch*seq, out), input_2d: (batch*seq, in)
            # grad_output_2d.T @ input_2d: (out, batch*seq) @ (batch*seq, in) = (out, in) ✓
            grad_weight = torch.matmul(grad_output_2d.transpose(0, 1), input_2d)
        else:
            # input: (batch, in), grad_output: (batch, out)
            # grad_output.T @ input: (out, batch) @ (batch, in) = (out, in) ✓
            grad_weight = torch.matmul(grad_output.transpose(0, 1), input)

    if compute_grad_bias:
        # grad_bias = sum(grad_output) along all dims except last
        grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))

    return grad_input, grad_weight, grad_bias


def rms_norm_backward_impl(grad_output, input, weight, eps):
    """
    Backward pass for RMS normalization.

    Forward: y = x / sqrt(mean(x^2) + eps) * weight

    Let:
        variance = mean(x^2)
        rms = sqrt(variance + eps)
        x_norm = x / rms
        y = x_norm * weight

    Gradients:
        grad_weight = sum(grad_output * x_norm) over all dims except last
        grad_x_norm = grad_output * weight
        grad_x = (grad_x_norm - mean(grad_x_norm * x_norm) * x_norm) / rms

    Args:
        grad_output: Gradient from downstream [*, hidden_size]
        input: Input tensor [*, hidden_size]
        weight: Weight tensor [hidden_size]
        eps: Epsilon for numerical stability

    Returns:
        (grad_input, grad_weight)
    """
    # Compute forward pass values needed for backward
    # variance = mean(x^2) along last dim
    variance = (input * input).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(variance + eps)
    x_norm = input / rms

    # Gradient w.r.t. weight
    # grad_weight = sum(grad_output * x_norm) over all dims except last
    grad_weight = (grad_output * x_norm).sum(dim=tuple(range(grad_output.ndim - 1)))

    # Gradient w.r.t. input
    # grad_x_norm = grad_output * weight
    grad_x_norm = grad_output * weight

    # grad_x = (grad_x_norm - mean(grad_x_norm * x_norm) * x_norm) / rms
    mean_term = (grad_x_norm * x_norm).mean(dim=-1, keepdim=True)
    grad_input = (grad_x_norm - mean_term * x_norm) / rms

    return grad_input, grad_weight


# ============================================================================
# Registration
# ============================================================================

_batch_invariant_backward_MODE = False
_batch_invariant_backward_LIB = None


def patch_batch_invariant_with_gradients():
    """Patch vLLM's batch_invariant mode to support gradients.

    This function adds backward pass support to vLLM's existing batch_invariant
    implementations by registering the backward operations. vLLM handles all the
    forward passes, we just add gradient support.
    """
    global _batch_invariant_backward_MODE, _batch_invariant_backward_LIB

    if _batch_invariant_backward_MODE:
        return

    # Get vLLM's batch_invariant library (already created by init_batch_invariance)
    from vllm.model_executor.layers import batch_invariant as vllm_bi

    if not hasattr(vllm_bi, '_batch_invariant_LIB') or vllm_bi._batch_invariant_LIB is None:
        raise RuntimeError(
            "vLLM's batch_invariant mode is not initialized. "
            "Call init_batch_invariance() first."
        )

    # Use vLLM's existing library - don't destroy it!
    _batch_invariant_backward_LIB = vllm_bi._batch_invariant_LIB

    # Just add the backward operations - everything else is already handled by vLLM
    _batch_invariant_backward_LIB.impl("aten::matmul_backward", matmul_backward_impl, "CUDA")
    _batch_invariant_backward_LIB.impl("aten::linear_backward", linear_backward_impl, "CUDA")

    # Monkey-patch vLLM's flash_attn_varlen_func to use our autograd wrapper for FA3
    import vllm.vllm_flash_attn.flash_attn_interface as fa_interface
    _original_flash_attn_varlen_func = fa_interface.flash_attn_varlen_func

    def patched_flash_attn_varlen_func(*args, **kwargs):
        # Only patch FA3 calls
        fa_version = kwargs.get('fa_version', fa_interface.DEFAULT_FA_VERSION)
        if fa_version == 3:
            # Extract the args needed for our autograd function
            q = args[0]
            k = args[1]
            v = args[2]
            max_seqlen_q = args[3]
            cu_seqlens_q = args[4]
            max_seqlen_k = args[5]
            cu_seqlens_k = args[6] if len(args) > 6 else kwargs.get('cu_seqlens_k')
            softmax_scale = kwargs.get('softmax_scale')
            causal = kwargs.get('causal', False)
            window_size = kwargs.get('window_size', (-1, -1))
            softcap = kwargs.get('softcap', 0.0)
            scheduler_metadata = kwargs.get('scheduler_metadata')
            num_splits = kwargs.get('num_splits', 0)

            if window_size is None:
                window_size = (-1, -1)
            window_left, window_right = window_size

            # Use our autograd wrapper
            return FlashAttn3Function.apply(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                softmax_scale, causal,
                window_left, window_right,
                softcap, scheduler_metadata, num_splits
            )
        else:
            # Fall through to original implementation for FA2
            return _original_flash_attn_varlen_func(*args, **kwargs)

    fa_interface.flash_attn_varlen_func = patched_flash_attn_varlen_func

    _batch_invariant_backward_MODE = True


def enable_batch_invariant_backward_mode():
    """Legacy name for patch_batch_invariant_with_gradients()."""
    patch_batch_invariant_with_gradients()


def disable_batch_invariant_backward_mode():
    """Disable batch invariant backward mode."""
    global _batch_invariant_backward_MODE, _batch_invariant_backward_LIB

    if _batch_invariant_backward_LIB is not None:
        _batch_invariant_backward_LIB._destroy()

    _batch_invariant_backward_MODE = False
    _batch_invariant_backward_LIB = None


def is_batch_invariant_backward_mode_enabled():
    """Check if batch invariant backward mode is enabled."""
    return _batch_invariant_backward_MODE


# ============================================================================
# Public API for gradient-enabled vLLM operations
# ============================================================================

def rms_norm_with_gradients(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    RMS normalization with gradient support.

    Uses vLLM's Triton kernel for forward pass (deterministic) and
    batch-invariant PyTorch operations for backward pass.

    Args:
        input: Input tensor [*, hidden_size]
        weight: Weight tensor [hidden_size]
        eps: Epsilon for numerical stability

    Returns:
        output: Normalized and scaled tensor [*, hidden_size]
    """
    return RMSNormFunction.apply(input, weight, eps)
