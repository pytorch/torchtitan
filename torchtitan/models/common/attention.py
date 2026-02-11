# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config.configurable import Module
from torchtitan.models.attention import (
    FlexAttentionWrapper,
    ScaledDotProductAttentionWrapper,
    VarlenAttentionWrapper,
    VarlenMetadata,
)
from torchtitan.models.common.rope import (
    apply_rotary_emb_complex,
    apply_rotary_emb_cos_sin,
)
from torchtitan.models.utils import trunc_normal_
from torchtitan.protocols.model import AttentionMasksType


class GQAttention(Module):
    """Grouped-Query Attention module shared across Llama3, Llama4, Qwen3.

    Supports GQA (grouped-query attention) with optional QK normalization,
    optional RoPE (for iRoPE layers), and multiple attention backends
    (flex, varlen, sdpa).

    Config parameters define the attention head structure. Runtime ``dim``
    is passed via ``build(dim=...)``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_heads: int
        n_kv_heads: int | None = None
        head_dim: int | None = None
        qk_norm: bool = False
        norm_eps: float = 1e-5
        bias: bool = False
        use_rope: bool = True
        attn_type: str = "sdpa"
        rope_format: str = "complex"  # "complex" or "cos_sin"

    def __init__(self, config: Config, *, dim: int):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = (
            config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        )
        self.head_dim = (
            config.head_dim if config.head_dim is not None else dim // config.n_heads
        )
        self.enable_gqa = self.n_heads > self.n_kv_heads
        self.use_rope = config.use_rope
        self.rope_format = config.rope_format

        # Optional QK normalization (Qwen3-style)
        self.q_norm: nn.RMSNorm | None = None
        self.k_norm: nn.RMSNorm | None = None
        if config.qk_norm:
            self.q_norm = nn.RMSNorm(
                self.head_dim, eps=config.norm_eps, elementwise_affine=True
            )
            self.k_norm = nn.RMSNorm(
                self.head_dim, eps=config.norm_eps, elementwise_affine=True
            )

        # Scaling factor (needed when head_dim differs from dim // n_heads)
        self.scaling = self.head_dim**-0.5 if config.head_dim is not None else None

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=config.bias)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=config.bias)

        self.attn_type = config.attn_type
        self.inner_attention: nn.Module
        match self.attn_type:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                self.inner_attention = VarlenAttentionWrapper()
            case "sdpa":
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Optional QK normalization (before RoPE, per Qwen3)
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        # Apply rotary embeddings
        if self.use_rope:
            if self.rope_format == "cos_sin":
                xq, xk = apply_rotary_emb_cos_sin(xq, xk, rope_cache, positions)
            else:
                xq, xk = apply_rotary_emb_complex(
                    xq, xk, freqs_cis=rope_cache, positions=positions
                )

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)

        scale_kwargs = {"scale": self.scaling} if self.scaling is not None else {}

        match self.attn_type:
            case "flex":
                # For iRoPE (Llama4), attention_masks may be a dict
                if isinstance(attention_masks, dict):
                    mask_key = "rope" if self.use_rope else "nope"
                    block_mask = attention_masks[mask_key]
                else:
                    assert isinstance(attention_masks, BlockMask), attention_masks
                    block_mask = attention_masks
                output = (
                    self.inner_attention(
                        xq,
                        xk,
                        xv,
                        block_mask=block_mask,
                        enable_gqa=self.enable_gqa,
                        **scale_kwargs,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
            case "varlen":
                assert isinstance(attention_masks, VarlenMetadata), attention_masks
                output = self.inner_attention(
                    xq, xk, xv, attention_masks, **scale_kwargs
                )
            case "sdpa":
                assert attention_masks is None
                output = (
                    self.inner_attention(
                        xq,
                        xk,
                        xv,
                        enable_gqa=self.enable_gqa,
                        **scale_kwargs,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

        output = output.view(bs, seqlen, -1)
        return self.wo(output)

    def init_weights(self, init_std: float = 0.02, **kwargs) -> None:
        for linear in (self.wq, self.wk, self.wv):
            trunc_normal_(linear.weight, mean=0.0, std=0.02)
            if linear.bias is not None:
                trunc_normal_(linear.bias, mean=0.0, std=0.02)
        trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.wo.bias is not None:
            trunc_normal_(self.wo.bias, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
