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
        tp_enabled: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.layer_name = layer_name
        self._tp_enabled = tp_enabled

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
            q: Query tensor [batch, num_heads, seq_len, head_dim] (DTensor when TP enabled)
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim] (DTensor when TP enabled)
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim] (DTensor when TP enabled)
            scale: Optional attention scale override (unused, vLLM uses internal scale)
            enable_gqa: Whether GQA is enabled (unused, vLLM handles GQA internally)

        Returns:
            output: [batch, num_heads, seq_len, head_dim] (plain tensor, not DTensor)
        """
        batch_size = q.shape[0]
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        if self._tp_enabled:
            # Capture correct seq_len from DTensor global shape BEFORE
            # to_local() corrupts.  Under torch.compile, prim_to_local
            # uses ceiling division for ALL dimensions, producing buggy
            # symbolic shapes like 2*(((s72+1)//2)) instead of s72 for
            # non-sharded dims.
            seq_len = q.shape[2]

            # Convert DTensors to local tensors
            q = q.to_local()
            k = k.to_local()
            v = v.to_local()

            # Fix the corrupted symbolic seq_len dimension with narrow().
            # At runtime this is a no-op view (actual shapes are correct).
            # During tracing it constrains the symbolic dim back to seq_len.
            q = q.narrow(2, 0, seq_len)
            k = k.narrow(2, 0, seq_len)
            v = v.narrow(2, 0, seq_len)

        # vLLM expects (num_tokens, num_heads, head_dim) where num_tokens = batch * seq_len
        # First transpose to (batch, seq_len, num_heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flatten batch and seq_len: (batch * seq_len, num_heads, head_dim)
        # Use -1 to avoid relying on symbolic shape from prim_to_local
        q = q.reshape(-1, num_heads, head_dim)
        k = k.reshape(-1, num_kv_heads, head_dim)
        v = v.reshape(-1, num_kv_heads, head_dim)

        # vLLM attention returns (num_tokens, hidden_size) where hidden_size = num_heads * head_dim
        output_flat = self.vllm_attn(q, k, v)

        # Reshape back: (batch, seq_len, num_heads, head_dim)
        # Use view(1, -1, ...) since batch is always 1 in vLLM varlen format
        output = output_flat.view(batch_size, -1, num_heads, head_dim)

        # Transpose back to TorchTitan format: (batch, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2)

        return output
