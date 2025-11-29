# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
vLLM-compatible Flash Attention implementation for deterministic RL training.
"""

import itertools

import torch

from vllm.attention import Attention
from vllm.attention.utils.fa_utils import is_flash_attn_varlen_func_available


class VLLMCompatibleFlashAttention(torch.nn.Module):
    """
    Wrapper around vLLM's Attention layer for deterministic RL training.

    This uses vLLM's high-level Attention module which handles:
    - KV cache management
    - Multiple attention backend selection (FlashAttention, xFormers, SDPA, etc.)
    - Quantization support
    - Optimized inference
    """

    # Class variable for auto-generating unique layer names (thread-safe)
    _layer_counter = itertools.count()

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        scale: float | None = None,
        causal: bool = True,
        use_fused_qkv: bool = True,
        use_qk_norm: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        if not is_flash_attn_varlen_func_available():
            raise RuntimeError(
                "Flash attention is not available. "
                "Please install flash-attn or use XPU platform with IPEX."
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.dropout = dropout
        self.causal = causal

        if scale is None:
            self.scale = self.head_dim**-0.5
        else:
            self.scale = scale

        # Create vLLM Attention layer to handle KV cache and backend selection
        try:
            from vllm.config import get_current_vllm_config

            config = get_current_vllm_config()
            cache_config = (
                config.cache_config if hasattr(config, "cache_config") else None
            )

            # Generate unique prefix for this attention layer
            # vLLM expects format "layers.X" for layer index extraction
            layer_idx = next(VLLMCompatibleFlashAttention._layer_counter)
            prefix = f"layers.{layer_idx}"

            self.vllm_attn = Attention(
                num_heads=num_heads,
                head_size=self.head_dim,
                scale=self.scale,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=None,
                prefix=prefix,
            )
        except (ImportError, RuntimeError, AttributeError):
            # Not in vLLM context - will fall back to direct flash attention
            self.vllm_attn = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Forward pass using vLLM's Attention layer.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
            scale: Optional scale override (uses self.scale if None)

        Returns:
            output: Attention output [batch, num_heads, seq_len, head_dim]
        """
        # Input is (batch, num_heads, seq_len, head_dim) from TorchTitan
        # Need to transpose to (batch, seq_len, num_heads, head_dim)
        q = q.transpose(1, 2)  # -> (batch, seq_len, num_heads, head_dim)
        k = k.transpose(1, 2)  # -> (batch, seq_len, num_kv_heads, head_dim)
        v = v.transpose(1, 2)  # -> (batch, seq_len, num_kv_heads, head_dim)

        # Get dimensions
        batch_size, seq_len, num_heads, head_dim = q.shape

        # Flatten to vLLM format: (total_tokens, num_heads, head_dim)
        total_tokens = batch_size * seq_len
        q = q.reshape(total_tokens, num_heads, head_dim)
        k = k.reshape(total_tokens, self.num_kv_heads, head_dim)
        v = v.reshape(total_tokens, self.num_kv_heads, head_dim)

        # Use vLLM's Attention layer if available (handles KV cache, backend selection)
        if self.vllm_attn is not None and not self.training:
            # vLLM Attention expects and returns [total_tokens, num_heads * head_dim]
            # But it can also accept [total_tokens, num_heads, head_dim]
            attn_output = self.vllm_attn(q, k, v)
            # Output is [total_tokens, num_heads * head_dim] or [total_tokens, num_heads, head_dim]
            if attn_output.dim() == 2:
                # Reshape to [total_tokens, num_heads, head_dim]
                attn_output = attn_output.reshape(total_tokens, num_heads, head_dim)
        else:
            # Training mode or fallback: use PyTorch SDPA
            # Reshape for SDPA: [batch, num_heads, seq_len, head_dim]
            q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.reshape(batch_size, seq_len, self.num_kv_heads, head_dim).transpose(
                1, 2
            )
            v = v.reshape(batch_size, seq_len, self.num_kv_heads, head_dim).transpose(
                1, 2
            )

            # Handle GQA by repeating k, v if needed
            if self.num_kv_heads != self.num_heads:
                num_repeats = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(num_repeats, dim=1)
                v = v.repeat_interleave(num_repeats, dim=1)

            # Use PyTorch SDPA
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
                scale=scale or self.scale,
            )

            # Transpose back and flatten: [batch, num_heads, seq_len, head_dim] -> [total_tokens, num_heads, head_dim]
            attn_output = attn_output.transpose(1, 2).reshape(
                total_tokens, num_heads, head_dim
            )

        # Reshape back to batch format and transpose to TorchTitan format
        # [total_tokens, num_heads, head_dim] -> [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        output = attn_output.reshape(
            batch_size, seq_len, num_heads, head_dim
        ).transpose(1, 2)

        return output
