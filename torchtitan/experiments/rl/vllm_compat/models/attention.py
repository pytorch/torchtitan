# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


# ============================================================================
# Flash attention custom op via torch.library.custom_op
#
# Opaque to both Dynamo and AOT autograd, so the compiled graph preserves
# it as a single node that calls vLLM's flash attention at runtime.
# ============================================================================


@torch.library.custom_op("vllm_compat::flash_attn", mutates_args=())
def flash_attn_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_len: int,
    scale: float,
    num_splits: int,
) -> torch.Tensor:
    """Flash attention varlen using vLLM's implementation."""
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

    total_tokens = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]

    out = torch.empty(
        (total_tokens, num_heads, head_dim), dtype=q.dtype, device=q.device
    )

    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        softmax_scale=scale,
        causal=True,
        num_splits=num_splits,
        fa_version=get_flash_attn_version(),
        out=out,
    )
    return output


@flash_attn_op.register_fake
def flash_attn_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_len: int,
    scale: float,
    num_splits: int,
) -> torch.Tensor:
    total_tokens = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    return torch.empty(
        (total_tokens, num_heads, head_dim), dtype=q.dtype, device=q.device
    )


def _flash_attn_setup_context(ctx, inputs, output):
    q, k, v, cu_seqlens, seq_len, scale, num_splits = inputs
    ctx.save_for_backward(q, k, v)
    ctx.scale = scale
    ctx.seq_len = seq_len


def _flash_attn_backward(ctx, grad_output):
    q, k, v = ctx.saved_tensors
    scale = ctx.scale
    seq_len = ctx.seq_len

    total_tokens = q.shape[0]
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    batch_size = total_tokens // seq_len

    q_batch = q.reshape(batch_size, seq_len, num_heads, head_dim)
    k_batch = k.reshape(batch_size, seq_len, num_kv_heads, head_dim)
    v_batch = v.reshape(batch_size, seq_len, num_kv_heads, head_dim)
    grad_out_batch = grad_output.reshape(batch_size, seq_len, num_heads, head_dim)

    # Transpose to (batch, num_heads, seq_len, head_dim)
    q_t = q_batch.transpose(1, 2)
    k_t = k_batch.transpose(1, 2)
    v_t = v_batch.transpose(1, 2)
    grad_out_t = grad_out_batch.transpose(1, 2)

    # For GQA, expand K/V to match Q's num_heads
    if num_kv_heads < num_heads:
        n_rep = num_heads // num_kv_heads
        k_t = k_t.repeat_interleave(n_rep, dim=1)
        v_t = v_t.repeat_interleave(n_rep, dim=1)

    # Compute attention scores: QK^T * scale
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

    # Apply causal mask
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))

    # Softmax
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)

    # grad_v = attn_weights^T @ grad_out
    grad_v_t = torch.matmul(attn_weights.transpose(-2, -1), grad_out_t)

    # grad_attn_weights = grad_out @ v^T
    grad_attn_weights = torch.matmul(grad_out_t, v_t.transpose(-2, -1))

    # Backward through softmax
    sum_term = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
    grad_scores = attn_weights * (grad_attn_weights - sum_term)
    grad_scores = grad_scores.masked_fill(causal_mask, 0.0)
    grad_scores = grad_scores * scale

    # grad_q = grad_scores @ K
    grad_q_t = torch.matmul(grad_scores, k_t)

    # grad_k = grad_scores^T @ Q
    grad_k_t = torch.matmul(grad_scores.transpose(-2, -1), q_t)

    # Transpose back and reshape to varlen format
    grad_q = grad_q_t.transpose(1, 2).reshape(total_tokens, num_heads, head_dim)

    # For GQA, reduce grad_k and grad_v back to num_kv_heads
    if num_kv_heads < num_heads:
        n_rep = num_heads // num_kv_heads
        grad_k_t = grad_k_t.reshape(
            batch_size, num_kv_heads, n_rep, seq_len, head_dim
        ).sum(dim=2)
        grad_v_t = grad_v_t.reshape(
            batch_size, num_kv_heads, n_rep, seq_len, head_dim
        ).sum(dim=2)

    grad_k = grad_k_t.transpose(1, 2).reshape(total_tokens, num_kv_heads, head_dim)
    grad_v = grad_v_t.transpose(1, 2).reshape(total_tokens, num_kv_heads, head_dim)

    # Return gradients for all inputs (None for non-differentiable ones)
    return grad_q, grad_k, grad_v, None, None, None, None


torch.library.register_autograd(
    "vllm_compat::flash_attn",
    _flash_attn_backward,
    setup_context=_flash_attn_setup_context,
)


class VLLMCompatibleFlashAttention(torch.nn.Module):
    """Wrapper around FlashAttention as used by VLLM"""

    def __init__(self) -> None:
        super().__init__()
        from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant

        self._num_splits = 1 if vllm_is_batch_invariant() else 0

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,  # unused; GQA handled automatically in backward
    ) -> torch.Tensor:
        # Input is (batch, num_heads, seq_len, head_dim) - need to transpose
        q = q.transpose(1, 2)  # -> (batch, seq_len, num_heads, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Get dimensions
        batch_size, seq_len, num_heads, head_dim = q.shape
        num_kv_heads = k.shape[2]

        # Convert to varlen format: flatten batch and sequence dimensions
        q_varlen = q.reshape(-1, num_heads, head_dim)
        k_varlen = k.reshape(-1, num_kv_heads, head_dim)
        v_varlen = v.reshape(-1, num_kv_heads, head_dim)

        # Create cumulative sequence lengths
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=q.device
        )

        if scale is None:
            scale = 1.0 / math.sqrt(q.size(-1))

        # Call flash attention custom op (opaque to compiler)
        output_varlen = torch.ops.vllm_compat.flash_attn(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens,
            seq_len,
            scale,
            self._num_splits,
        )

        # Convert back to batch format
        output = output_varlen.reshape(batch_size, seq_len, num_heads, head_dim)

        # Transpose back to TorchTitan format: (batch, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2)

        return output
