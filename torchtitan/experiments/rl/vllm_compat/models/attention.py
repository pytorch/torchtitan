# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


class VLLMCompatibleFlashAttention(torch.nn.Module):
    """Wrapper around FlashAttention as used by VLLM"""

    def __init__(self) -> None:
        super().__init__()
        self.flash_attn_varlen_func = flash_attn_varlen_func
        from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

        self.vllm_is_batch_invariant = vllm_is_batch_invariant
        self.fa_version = get_flash_attn_version()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        # Flash Attention varlen expects: (batch, seqlen, nheads, headdim)
        # The input from TorchTitan is always (batch, num_heads, seq_len, head_dim)
        # We need to transpose to (batch, seq_len, num_heads, head_dim)

        # Input is (batch, num_heads, seq_len, head_dim) - need to transpose
        # TODO(jianiw): Explicitly convert to bfloat16 because vllm's Flash attention kernel only supports bfloat16.
        # Need to handle precision change properly.
        original_dtype = q.dtype
        q = q.transpose(1, 2).to(
            torch.bfloat16
        )  # -> (batch, seq_len, num_heads, head_dim)
        k = k.transpose(1, 2).to(torch.bfloat16)
        v = v.transpose(1, 2).to(torch.bfloat16)

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

        # Scaling factor applied prior to softmax. If none, the default value is set to :math:`\frac{1}{\sqrt{E}}`.
        if scale is None:
            scale = 1.0 / math.sqrt(q.size(-1))

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
                num_kv_heads = k.shape[1]
                head_dim = q.shape[2]
                batch_size = total_tokens // seq_len
                num_groups = num_heads // num_kv_heads

                q_batch = q.reshape(batch_size, seq_len, num_heads, head_dim)
                k_batch = k.reshape(batch_size, seq_len, num_kv_heads, head_dim)
                v_batch = v.reshape(batch_size, seq_len, num_kv_heads, head_dim)
                out_batch = output.reshape(batch_size, seq_len, num_heads, head_dim)
                grad_out_batch = grad_output.reshape(
                    batch_size, seq_len, num_heads, head_dim
                )

                # Transpose to (batch, heads, seq_len, head_dim)
                q_t = q_batch.transpose(1, 2)  # (B, num_heads, N, D)
                k_t = k_batch.transpose(1, 2)  # (B, num_kv_heads, N, D)
                v_t = v_batch.transpose(1, 2)  # (B, num_kv_heads, N, D)
                grad_out_t = grad_out_batch.transpose(1, 2)

                # Expand k/v to match q's head count for GQA
                if num_groups > 1:
                    k_t = k_t.repeat_interleave(
                        num_groups, dim=1
                    )  # (B, num_heads, N, D)
                    v_t = v_t.repeat_interleave(
                        num_groups, dim=1
                    )  # (B, num_heads, N, D)

                # Compute attention scores: QK^T
                # q_t: (B, num_heads, N, D), k_t: (B, num_heads, N, D) -> scores: (B, num_heads, N, N)
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
                )  # (B, num_heads, N, N)

                # Backward through attention
                # grad_v = attn_weights^T @ grad_out
                grad_v_t = torch.matmul(
                    attn_weights.transpose(-2, -1), grad_out_t
                )  # (B, num_heads, N, D)

                # grad_attn_weights = grad_out @ v^T
                grad_attn_weights = torch.matmul(grad_out_t, v_t.transpose(-2, -1))

                # Backward through softmax
                sum_term = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
                grad_scores = attn_weights * (grad_attn_weights - sum_term)

                # Apply causal mask to gradients
                grad_scores = grad_scores.masked_fill(causal_mask, 0.0)

                # Backward through QK^T and scale
                grad_scores = grad_scores * scale

                # grad_q = grad_scores @ K
                grad_q_t = torch.matmul(grad_scores, k_t)  # (B, num_heads, N, D)

                # grad_k = grad_scores^T @ Q
                grad_k_t = torch.matmul(
                    grad_scores.transpose(-2, -1), q_t
                )  # (B, num_heads, N, D)

                # Reduce grad_k and grad_v back to num_kv_heads by summing over groups
                if num_groups > 1:
                    grad_k_t = grad_k_t.reshape(
                        batch_size, num_kv_heads, num_groups, seq_len, head_dim
                    ).sum(dim=2)
                    grad_v_t = grad_v_t.reshape(
                        batch_size, num_kv_heads, num_groups, seq_len, head_dim
                    ).sum(dim=2)

                # Transpose back and reshape to varlen format
                grad_q = grad_q_t.transpose(1, 2).reshape(
                    total_tokens, num_heads, head_dim
                )
                grad_k = grad_k_t.transpose(1, 2).reshape(
                    total_tokens, num_kv_heads, head_dim
                )
                grad_v = grad_v_t.transpose(1, 2).reshape(
                    total_tokens, num_kv_heads, head_dim
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

        return output.to(original_dtype)
