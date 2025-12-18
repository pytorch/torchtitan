# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
    from batch_invariant_backward import enable_batch_invariant_backward_mode

    # Initialize vLLM's deterministic mode first
    init_batch_invariance()

    # Then enable gradient support
    enable_batch_invariant_backward_mode()

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


class SiluAndMulFunction(Function):
    """
    Autograd function for vLLM's SiluAndMul activation.

    Forward: splits input into [gate, up], returns silu(gate) * up
    where silu(x) = x * sigmoid(x)
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass using vLLM's SiluAndMul.

        Args:
            x: Input tensor [..., hidden_dim * 2] where first half is gate, second half is up

        Returns:
            output: silu(gate) * up, shape [..., hidden_dim]
        """
        from vllm.model_executor.layers.activation import SiluAndMul as VLLMSiluAndMul

        # Use vLLM's implementation for forward
        vllm_silu_and_mul = VLLMSiluAndMul()
        output = vllm_silu_and_mul(x)

        # Save for backward
        ctx.save_for_backward(x)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for SiluAndMul.

        Let gate = x[:d], up = x[d:] where d = hidden_dim
        Forward: out = silu(gate) * up = (gate * sigmoid(gate)) * up

        Gradients:
        - grad_gate = grad_out * up * d_silu(gate)
        - grad_up = grad_out * silu(gate)

        where d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        """
        (x,) = ctx.saved_tensors

        # Split input into gate and up
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]

        # Compute sigmoid and silu for backward
        sigmoid_gate = torch.sigmoid(gate)
        silu_gate = gate * sigmoid_gate

        # Gradient of silu: d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        d_silu_gate = sigmoid_gate * (1 + gate * (1 - sigmoid_gate))

        # Compute gradients
        grad_gate = grad_output * up * d_silu_gate
        grad_up = grad_output * silu_gate

        # Concatenate gradients
        grad_x = torch.cat([grad_gate, grad_up], dim=-1)

        return grad_x


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


# ============================================================================
# Registration
# ============================================================================

_batch_invariant_backward_mode = False
_batch_invariant_backward_lib = None


def enable_batch_invariant_backward_mode():
    """Enable batch invariant backward mode to support gradients.

    This function adds backward pass support to vLLM's existing batch_invariant
    implementations by registering the backward operations. vLLM handles all the
    forward passes, we just add gradient support.
    """
    global _batch_invariant_backward_mode, _batch_invariant_backward_lib

    if _batch_invariant_backward_mode:
        return

    # Get vLLM's batch_invariant library (already created by init_batch_invariance)
    from vllm.model_executor.layers import batch_invariant as vllm_bi

    if (
        not hasattr(vllm_bi, "_batch_invariant_LIB")
        or vllm_bi._batch_invariant_LIB is None
    ):
        raise RuntimeError(
            "vLLM's batch_invariant mode is not initialized. "
            "Call init_batch_invariance() first."
        )

    # Use vLLM's existing library - don't destroy it!
    _batch_invariant_backward_lib = vllm_bi._batch_invariant_LIB

    # Just add the backward operations - everything else is already handled by vLLM
    _batch_invariant_backward_lib.impl(
        "aten::matmul_backward", matmul_backward_impl, "CUDA"
    )
    _batch_invariant_backward_lib.impl(
        "aten::linear_backward", linear_backward_impl, "CUDA"
    )

    _batch_invariant_backward_mode = True


def disable_batch_invariant_backward_mode():
    """Disable batch invariant backward mode."""
    global _batch_invariant_backward_mode, _batch_invariant_backward_lib

    if _batch_invariant_backward_lib is not None:
        _batch_invariant_backward_lib._destroy()

    _batch_invariant_backward_mode = False
    _batch_invariant_backward_lib = None


def is_batch_invariant_backward_mode_enabled():
    """Check if batch invariant backward mode is enabled."""
    return _batch_invariant_backward_mode


# ============================================================================
# Public API for gradient-enabled vLLM operations
# ============================================================================


def rms_norm_with_gradients(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
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


def silu_and_mul_with_gradients(x: torch.Tensor) -> torch.Tensor:
    """
    SiluAndMul activation with gradient support.

    Uses vLLM's implementation for forward pass (deterministic) and
    implements proper backward pass for training.

    Args:
        x: Input tensor [..., hidden_dim * 2] where first half is gate, second half is up

    Returns:
        output: silu(gate) * up, shape [..., hidden_dim]
    """
    return SiluAndMulFunction.apply(x)
