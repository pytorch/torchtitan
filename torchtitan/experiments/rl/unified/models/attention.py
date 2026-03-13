# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from vllm.model_executor.layers.attention import Attention


class VLLMAttention(torch.nn.Module):
    """
    Wrapper around vLLM's Attention. Compatible with TorchTitan input shape.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_name: str,
        scale: float | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.layer_name = layer_name

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        if scale is None:
            self.scale = head_dim**-0.5
        else:
            self.scale = scale

        cache_config = (
            vllm_config.cache_config if hasattr(vllm_config, "cache_config") else None
        )

        self.vllm_attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=self.scale,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            prefix=f"model.layers.{layer_name}.attention.inner_attention",
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass using vLLM's Attention layer for inference.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
            scale: Optional attention scale override (unused, vLLM uses internal scale)
            enable_gqa: Whether GQA is enabled (unused, vLLM handles GQA internally)

        Returns:
            output: [batch, num_heads, seq_len, head_dim]
        """
        # Input is (batch, num_heads, seq_len, head_dim)
        # TODO: may be good to use einops in future as we can explicitly reshape
        # with dimension names - see https://github.com/arogozhnikov/einops
        batch_size, num_heads, seq_len, head_dim = q.shape
        _, num_kv_heads, _, _ = k.shape

        # vLLM expects (num_tokens, num_heads, head_dim) where num_tokens = batch * seq_len
        # First transpose to (batch, seq_len, num_heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # TODO: reimplement as a 4d tensor once vLLM fix has landed
        # Then flatten batch and seq_len: (batch * seq_len, num_heads, head_dim)
        q = q.reshape(batch_size * seq_len, num_heads, head_dim)
        k = k.reshape(batch_size * seq_len, num_kv_heads, head_dim)
        v = v.reshape(batch_size * seq_len, num_kv_heads, head_dim)

        # vLLM attention returns (num_tokens, hidden_size) where hidden_size = num_heads * head_dim
        output_flat = self.vllm_attn(q, k, v)

        # Output is (batch * seq_len, num_heads * head_dim), reshape to (batch, seq_len, num_heads, head_dim)
        output = output_flat.view(batch_size, seq_len, num_heads, head_dim)

        # Transpose back to TorchTitan format: (batch, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2)

        return output
