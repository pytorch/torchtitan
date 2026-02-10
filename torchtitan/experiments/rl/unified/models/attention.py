# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.attention import activate_flash_attention_impl

activate_flash_attention_impl("FA3")

from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend


@register_backend(AttentionBackendEnum.CUSTOM)
class UnifiedFlashAttentionBackend(FlashAttentionBackend):
    """Unified FlashAttention backend for RL."""

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return UnifiedFlashAttentionImpl


class UnifiedFlashAttentionImpl(FlashAttentionImpl):
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        assert (
            self.vllm_flash_attn_version is not None
        ), "FlashAttention version not detected."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        assert attn_type not in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER)

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)

        assert not self.kv_cache_dtype.startswith("fp8")

        assert not attn_metadata.use_cascade

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        assert self.dcp_world_size == 1

        if attn_metadata.causal:
            sliding_window_size = (-1, 0)
        else:
            raise RuntimeError("Non-causal attention not supported yet.")

        assert self.alibi_slopes is None

        return torch.nn.attention.varlen.varlen_attn_out(
            output,
            query,
            key_cache,
            value_cache,
            cu_seqlens_q,
            None,
            max_seqlen_q,
            max_seqlen_k,
            scale=self.scale,
            window_size=sliding_window_size,
            cache_batch_idx=None,
            page_table=block_table,
            seqused_k=seqused_k,
        )


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

        attention = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=self.scale,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            prefix=f"model.layers.{layer_name}.attention.inner_attention",
        )
        assert attention.attn_backend is UnifiedFlashAttentionBackend
        self.vllm_attn = attention

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
