# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
vLLM-compatible Flash Attention implementation for deterministic RL training.
"""

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func


class VLLMCompatibleFlashAttention(torch.nn.Module):
    """Wrapper around FlashAttention as used by VLLM"""

    def __init__(self) -> None:
        super().__init__()
        self.flash_attn_varlen_func = flash_attn_varlen_func
        from vllm.attention.utils.fa_utils import get_flash_attn_version
        from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant

        self.vllm_is_batch_invariant = vllm_is_batch_invariant
        self.fa_version = get_flash_attn_version()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        # Flash Attention varlen expects: (batch, seqlen, nheads, headdim)
        # The input from TorchTitan is always (batch, num_heads, seq_len, head_dim)
        # We need to transpose to (batch, seq_len, num_heads, head_dim)

        # Input is (batch, num_heads, seq_len, head_dim) - need to transpose
        q = q.transpose(1, 2)  # -> (batch, seq_len, num_heads, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Get dimensions
        batch_size, seq_len, num_heads, head_dim = q.shape

        # Convert to varlen format: flatten batch and sequence dimensions
        # (batch, seqlen, nheads, headdim) -> (total_tokens, nheads, headdim)
        q_varlen = q.reshape(-1, num_heads, head_dim)
        k_varlen = k.reshape(-1, k.shape[2], head_dim)
        v_varlen = v.reshape(-1, v.shape[2], head_dim)

        # Create cumulative sequence lengths
        # cu_seqlens: [0, seq_len, 2*seq_len, ..., batch_size*seq_len]
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=q.device
        )

        # Wrap Flash Attention with manual backward pass
        class FlashAttnWithBackward(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx,
                q,
                k,
                v,
                cu_seqlens,
                seq_len,
                scale,
                num_splits,
                flash_fn,
                fa_version,
            ):
                # Call flash attention for forward (fast)
                output = flash_fn(
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
                    fa_version=fa_version,
                )
                # Save for backward
                ctx.save_for_backward(q, k, v, output)
                ctx.scale = scale
                ctx.seq_len = seq_len
                return output

            @staticmethod
            def backward(ctx, grad_output):
                q, k, v, output = ctx.saved_tensors
                scale = ctx.scale
                seq_len = ctx.seq_len

                # Reshape from varlen back to batch format for attention computation
                # Assume uniform sequence lengths (batch_size = total_tokens / seq_len)
                total_tokens = q.shape[0]
                num_heads = q.shape[1]
                head_dim = q.shape[2]
                batch_size = total_tokens // seq_len

                q_batch = q.reshape(batch_size, seq_len, num_heads, head_dim)
                k_batch = k.reshape(batch_size, seq_len, num_heads, head_dim)
                v_batch = v.reshape(batch_size, seq_len, num_heads, head_dim)
                out_batch = output.reshape(batch_size, seq_len, num_heads, head_dim)
                grad_out_batch = grad_output.reshape(
                    batch_size, seq_len, num_heads, head_dim
                )

                # Transpose to (batch, num_heads, seq_len, head_dim)
                q_t = q_batch.transpose(1, 2)
                k_t = k_batch.transpose(1, 2)
                v_t = v_batch.transpose(1, 2)
                out_t = out_batch.transpose(1, 2)
                grad_out_t = grad_out_batch.transpose(1, 2)

                # Compute attention scores: QK^T
                # q_t: (B, H, N, D), k_t: (B, H, N, D) -> scores: (B, H, N, N)
                scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

                # Apply causal mask
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores = scores.masked_fill(causal_mask, float("-inf"))

                # Softmax
                attn_weights = torch.nn.functional.softmax(
                    scores, dim=-1
                )  # (B, H, N, N)

                # Backward through attention
                # out = attn_weights @ v
                # grad_v = attn_weights^T @ grad_out
                grad_v_t = torch.matmul(attn_weights.transpose(-2, -1), grad_out_t)

                # grad_attn_weights = grad_out @ v^T
                grad_attn_weights = torch.matmul(grad_out_t, v_t.transpose(-2, -1))

                # Backward through softmax
                # d_softmax = attn_weights * (grad_attn_weights - sum(grad_attn_weights * attn_weights))
                sum_term = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
                grad_scores = attn_weights * (grad_attn_weights - sum_term)

                # Apply causal mask to gradients
                grad_scores = grad_scores.masked_fill(causal_mask, 0.0)

                # Backward through QK^T and scale
                grad_scores = grad_scores * scale

                # grad_q = grad_scores @ K
                grad_q_t = torch.matmul(grad_scores, k_t)

                # grad_k = grad_scores^T @ Q
                grad_k_t = torch.matmul(grad_scores.transpose(-2, -1), q_t)

                # Transpose back and reshape to varlen format
                grad_q = grad_q_t.transpose(1, 2).reshape(
                    total_tokens, num_heads, head_dim
                )
                grad_k = grad_k_t.transpose(1, 2).reshape(
                    total_tokens, num_heads, head_dim
                )
                grad_v = grad_v_t.transpose(1, 2).reshape(
                    total_tokens, num_heads, head_dim
                )

                return grad_q, grad_k, grad_v, None, None, None, None, None, None

        # Call Flash Attention varlen with custom backward
        output_varlen = FlashAttnWithBackward.apply(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens,
            seq_len,
            scale,
            1 if self.vllm_is_batch_invariant() else 0,
            self.flash_attn_varlen_func,
            self.fa_version,
        )

        # Convert back to batch format
        # (total_tokens, nheads, headdim) -> (batch, seqlen, nheads, headdim)
        output = output_varlen.reshape(batch_size, seq_len, num_heads, head_dim)

        # Transpose back to TorchTitan format: (batch, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2)

        return output
