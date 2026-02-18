# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch-invariant operations with backward pass support.

Adds gradient support to vLLM's deterministic batch_invariant mode by:
- Registering aten::matmul_backward and aten::linear_backward dispatch overrides
  that use vLLM's deterministic matmul kernels
- Providing custom ops (torch.library.custom_op) for rms_norm and silu_and_mul
  that are opaque to Dynamo/AOT autograd with proper backward implementations

This achieves bitwise-deterministic RL training where both rollouts (forward)
and training (forward + backward) produce identical results.
"""

import torch
import torch.library


# ============================================================================
# Custom ops via torch.library.custom_op
#
# These are opaque to both Dynamo and AOT autograd, so the compiled graph
# preserves them as single nodes that call vLLM kernels at runtime.
# ============================================================================


_cached_silu_and_mul = None

# Avoid reinstantiating SiluAndMul each time it's called
def _get_silu_and_mul():
    global _cached_silu_and_mul
    if _cached_silu_and_mul is None:
        from vllm.config import set_current_vllm_config, VllmConfig
        from vllm.model_executor.layers.activation import SiluAndMul as VLLMSiluAndMul

        with set_current_vllm_config(VllmConfig()):
            _cached_silu_and_mul = VLLMSiluAndMul(compile_native=False)
    return _cached_silu_and_mul


@torch.library.custom_op("vllm_compat::silu_and_mul", mutates_args=())
def silu_and_mul_op(x: torch.Tensor) -> torch.Tensor:
    """SiluAndMul activation using vLLM's implementation.

    Splits input into [gate, up], returns silu(gate) * up.
    """
    return _get_silu_and_mul()(x)


@silu_and_mul_op.register_fake
def silu_and_mul_fake(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return x[..., :d].contiguous()


def _silu_and_mul_setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)


def _silu_and_mul_backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]

    sigmoid_gate = torch.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    d_silu_gate = sigmoid_gate * (1 + gate * (1 - sigmoid_gate))

    grad_gate = grad_output * up * d_silu_gate
    grad_up = grad_output * silu_gate
    grad_x = torch.cat([grad_gate, grad_up], dim=-1)
    return (grad_x,)


torch.library.register_autograd(
    "vllm_compat::silu_and_mul",
    _silu_and_mul_backward,
    setup_context=_silu_and_mul_setup_context,
)


@torch.library.custom_op("vllm_compat::rms_norm", mutates_args=())
def rms_norm_op(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS normalization using vLLM's Triton kernel."""
    from vllm.model_executor.layers.batch_invariant import rms_norm as vllm_rms_norm

    return vllm_rms_norm(input, weight, eps)


@rms_norm_op.register_fake
def rms_norm_fake(
    input: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return torch.empty_like(input)


def _rms_norm_setup_context(ctx, inputs, output):
    input, weight, eps = inputs
    ctx.save_for_backward(input, weight)
    ctx.eps = eps


def _rms_norm_backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    eps = ctx.eps

    # Recompute forward values for backward
    # Use sum/count instead of mean to avoid vLLM's mean_batch_invariant
    # which converts to float32.
    squared = input * input
    variance = squared.sum(dim=-1, keepdim=True) / input.shape[-1]
    rms = torch.sqrt(variance + eps)
    x_norm = input / rms

    # grad_weight = sum(grad_output * x_norm) over all dims except last
    grad_weight = (grad_output * x_norm).sum(dim=tuple(range(grad_output.ndim - 1)))

    # grad_input
    grad_x_norm = grad_output * weight
    product = grad_x_norm * x_norm
    mean_term = product.sum(dim=-1, keepdim=True) / input.shape[-1]
    grad_input = (grad_x_norm - mean_term * x_norm) / rms

    return grad_input, grad_weight, None


torch.library.register_autograd(
    "vllm_compat::rms_norm",
    _rms_norm_backward,
    setup_context=_rms_norm_setup_context,
)


# ============================================================================
# Backward operation implementations for aten matmul/linear dispatch overrides
# ============================================================================


def matmul_backward_impl(grad_output, self, other, output_mask):
    """
    Backward pass for matmul: y = matmul(a, b)

    grad_a = grad_output @ b.T
    grad_b = a.T @ grad_output

    Uses torch.matmul which is overridden by vLLM's batch_invariant mode.
    """
    grad_self = grad_other = None

    compute_grad_self = output_mask[0] if len(output_mask) > 0 else True
    compute_grad_other = output_mask[1] if len(output_mask) > 1 else True

    if compute_grad_self:
        grad_self = torch.matmul(grad_output, other.transpose(-2, -1))

    if compute_grad_other:
        grad_other = torch.matmul(self.transpose(-2, -1), grad_output)

    return grad_self, grad_other


def linear_backward_impl(grad_output, input, weight, output_mask):
    """
    Backward pass for linear: y = input @ weight.T + bias

    PyTorch passes args in swapped order: (saved_input, grad_output, weight, output_mask)
    so we swap the first two args.
    """
    input, grad_output = grad_output, input

    grad_input = grad_weight = grad_bias = None

    compute_grad_input = output_mask[0] if len(output_mask) > 0 else True
    compute_grad_weight = output_mask[1] if len(output_mask) > 1 else True
    compute_grad_bias = output_mask[2] if len(output_mask) > 2 else True

    if compute_grad_input:
        grad_input = torch.matmul(grad_output, weight)

    if compute_grad_weight:
        # Flatten to 2D for multi-dimensional inputs
        input_2d = input.reshape(-1, input.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_weight = torch.matmul(grad_output_2d.transpose(0, 1), input_2d)

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
    """Register backward pass support for vLLM's batch_invariant dispatch overrides."""
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
            "Call init_batch_invariance(AttentionBackendEnum.FLASH_ATTN) first."
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
    """RMS normalization with gradient support."""
    return torch.ops.vllm_compat.rms_norm(input, weight, eps)


def silu_and_mul_with_gradients(x: torch.Tensor) -> torch.Tensor:
    """SiluAndMul activation with gradient support."""
    return torch.ops.vllm_compat.silu_and_mul(x)
