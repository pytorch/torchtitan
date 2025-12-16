# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from vllm.attention.layer import Attention


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
    ) -> torch.Tensor:
        """
        Forward pass using vLLM's Attention layer for inference.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
            scale: Optional attention scale override (unused, vLLM uses internal scale)

        Returns:
            output: [batch, num_heads, seq_len, head_dim]
        """
        # Input is (batch, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Transpose to (batch, seq_len, num_heads, head_dim) for vLLM
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output_varlen = self.vllm_attn(q, k, v)

        # Reshape back to batch format
        output = output_varlen.view(batch_size, seq_len, num_heads, head_dim)

        # Transpose back to TorchTitan format: (batch, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2)

        return output
