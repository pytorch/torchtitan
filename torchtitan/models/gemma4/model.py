# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseQKVLinear,
    FlexAttention,
    get_causal_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
    get_sliding_window_mask_mod,
    GQAttention,
    local_head_split,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module


class Gemma4QKVLinear(BaseQKVLinear):
    """Separate Q, K, and optional V projection supporting attention_k_eq_v."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseQKVLinear.Config):
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build() if config.wv is not None else None

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x) if self.wv is not None else xk
        return (
            local_head_split(xq, self.head_dim),
            local_head_split(xk, self.head_dim),
            local_head_split(xv, self.head_dim),
        )


class Gemma4RoPE(CosSinRoPE):
    """RoPE implementation for Gemma-4 supporting proportional partial rotation."""

    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        partial_rotary_factor: float = 1.0

    def _precompute_cache(self) -> torch.Tensor:
        cfg = self.config
        dim = cfg.dim
        max_context_length = cfg.max_context_length
        base = cfg.theta
        partial_rotary_factor = getattr(cfg, "partial_rotary_factor", 1.0)

        if partial_rotary_factor < 1.0:
            # Proportional RoPE (HF Gemma-4):
            # Only rotate the first partial_rotary_factor proportion of head_dim,
            # with frequencies normalized across the full head_dim.
            rope_angles = int(partial_rotary_factor * dim // 2)
            inv_freq_rotated = 1.0 / (
                base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / dim)
            )
            nope_angles = dim // 2 - rope_angles
            if nope_angles > 0:
                inv_freq = torch.cat(
                    [inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)],
                    dim=0,
                )
            else:
                inv_freq = inv_freq_rotated
        else:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )

        t = torch.arange(
            max_context_length, dtype=inv_freq.dtype, device=inv_freq.device
        )
        freqs = torch.outer(t, inv_freq).float()
        theta = torch.cat([freqs, freqs], dim=-1)

        cos = theta.cos()
        sin = theta.sin()
        return torch.cat([cos, sin], dim=-1)


class Gemma4GlobalSDPA(Module):
    """Causal ScaledDotProductAttention for Gemma-4 global layers (head_dim=512).

    Adapts packed [T, H, K] tokens to [1, H, T, K] for F.scaled_dot_product_attention,
    matching the exact polymorphic tensor interface of FlexAttention.

    Architectural Rationale & Upstream Paper Trail:
        1. SRAM/LDS Dimension Ceiling (Cross-Vendor):
           FlashAttention and FlexAttention Triton kernel templates (e.g. flex_backwards.py.jinja)
           tile Q/K/V into on-chip SRAM/LDS scratchpad buffers. Tile accumulators sized
           [BLOCK_M2, QK_HEAD_DIM_ROUNDED] at head_dim=512 require 128 KiB of scratchpad memory
           per accumulator tile alone. When combined with K/V/DO/DK/DV staging buffers, this
           exceeds per-SM/per-CU shared memory limits across all current hardware:
             - NVIDIA Hopper/Blackwell and Ada (SM89, SM90)
             - AMD CDNA2/3/4 (gfx90a, gfx942, gfx950)
           Inductor autotuning consequently fails with NoValidChoicesError.

        2. Upstream Ecosystem Status (Cross-Framework Consensus):
           This constraint and the per-layer dispatch solution are actively tracked and validated:
             - huggingface/transformers#45201: "[Gemma 4] Support per-layer FlashAttention: FA2
               for sliding layers, SDPA for global layers" (Details how FA2 fails on global
               layers with RuntimeError: FlashAttention only supports head dimensions up to 256,
               proposing per-layer dispatch as the canonical solution).
             - Dao-AILab/flash-attention#2427: "Support head_dim=512 for Gemma 4 global attention layers"
               (Identifies Gemma-4 as the first production model requiring multi-head GQA at
               head_dim=512, recommending FA2/FlexAttention for sliding layers and SDPA for global layers).
             - Dao-AILab/flash-attention#2581: "Support head_dim=512 on SM89 (Ada) for Gemma 4 global attention layers"
               (Notes flash_attn_varlen_func rejects head_dim=512 outright on Ada GPUs).
             - Dao-AILab/flash-attention#2318: (FA4 Hopper head_dim=256 supported, 512 still unsupported).
             - Dao-AILab/flash-attention#801: "May support headdim>256? such as 512" (Closed unresolved).

        3. Dispatch Solution:
           Global layers have no sliding-window constraint and execute standard causal attention.
           Static layer construction dispatch routes global layers to PyTorch's native C++ SDPA,
           preserving FlexAttention for sliding-window layers without any runtime branching in forward().
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_masks: Any = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # Adapt 3D packed tokens [T, H, K] to 4D kernel layout [1, H, T, K]
        q_1HTK = q_THK.transpose(0, 1).unsqueeze(0)
        k_1HTK = k_THK.transpose(0, 1).unsqueeze(0)
        v_1HTV = v_THV.transpose(0, 1).unsqueeze(0)

        # Handle Grouped-Query Attention if key/value heads are fewer than query heads
        if q_1HTK.shape[1] != k_1HTK.shape[1]:
            n_rep = q_1HTK.shape[1] // k_1HTK.shape[1]
            k_1HTK = k_1HTK.repeat_interleave(n_rep, dim=1)
            v_1HTV = v_1HTV.repeat_interleave(n_rep, dim=1)

        # Gemma-4 specifies unit attention scaling (scale=1.0) because of QK RMSNorm.
        # If scale is explicitly passed (self.scaling), F.sdpa respects it;
        # otherwise it defaults to standard 1 / sqrt(head_dim).
        out_1HTV = F.scaled_dot_product_attention(
            q_1HTK,
            k_1HTK,
            v_1HTV,
            is_causal=True,
            scale=scale,
        )
        return out_1HTV.squeeze(0).transpose(0, 1).contiguous()


class Gemma4Attention(GQAttention):
    """Gemma-4 GQA with unit attention scale, QK RMSNorm, and V RMSNorm."""

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        attn_scale: float = 1.0
        v_norm: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.scaling = config.attn_scale
        self.v_norm = config.v_norm.build() if config.v_norm is not None else None

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xq_THK, xk_THK, xv_THV = self.qkv_linear(x_TD)
        if self.q_norm is not None and self.k_norm is not None:
            xq_THK = self.q_norm(xq_THK)
            xk_THK = self.k_norm(xk_THK)
        if positions is not None and positions.shape[0] != xq_THK.shape[0]:
            # Under TP Sequence Parallelism, query tokens are sharded across TP ranks.
            # Slice positions locally to match this rank's token count.
            positions = positions[: xq_THK.shape[0]]
        xq_THK, xk_THK = self.rope(xq_THK, xk_THK, positions)
        if self.v_norm is not None:
            xv_THV = self.v_norm(xv_THV)

        out_THV = self.inner_attention(
            xq_THK,
            xk_THK,
            xv_THV,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()
        return self.wo(out_THV.view(out_THV.shape[0], -1))


class Gemma4FeedForward(FeedForward):
    """Gemma-4 GeGLU feed-forward module."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x), approximate="tanh") * self.w3(x))


class Gemma4TransformerBlock(TransformerBlock):
    """Gemma-4 Transformer block with 4 layer norms and hybrid mask routing."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        use_global_attention: bool = False
        post_attention_norm: RMSNorm.Config
        post_ffn_norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.use_global_attention = config.use_global_attention
        self.attention = config.attention.build()
        if config.feed_forward is None:
            raise ValueError("feed_forward configuration must be provided")
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.post_attention_norm = config.post_attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.post_ffn_norm = config.post_ffn_norm.build()
        self.register_buffer("layer_scalar", torch.ones(1))

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        super()._init_self_buffers(buffer_device=buffer_device)
        self.layer_scalar.fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        layer_mask = attention_masks
        if isinstance(attention_masks, dict):
            layer_mask = attention_masks[
                "global" if self.use_global_attention else "sliding_window"
            ]

        h = x + self.post_attention_norm(
            self.attention(self.attention_norm(x), layer_mask, positions)
        )
        h = h + self.post_ffn_norm(self.feed_forward(self.ffn_norm(h)))
        h = h * self.layer_scalar
        return h


class Gemma4Model(Decoder):
    """Gemma-4 hybrid attention model."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 3584
        vocab_size: int = 262144
        sliding_window_size: int = 4096
        global_attn_interval: int = 6
        enable_sliding_window: bool = True

        def update_from_config(self, *, config, **kwargs) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            tp = config.parallelism.tensor_parallel_degree
            if tp > 1:
                for idx, layer in enumerate(self.layers):
                    attn = layer.attention
                    n_heads = attn.n_heads
                    n_kv_heads = getattr(attn, "n_kv_heads", None) or n_heads
                    if n_heads % tp != 0:
                        raise ValueError(
                            f"tensor_parallel_degree ({tp}) must divide "
                            f"n_heads ({n_heads}) on layer {idx}."
                        )
                    if n_kv_heads % tp != 0:
                        is_global = getattr(layer, "use_global_attention", False)
                        layer_type = "global" if is_global else "sliding"
                        raise ValueError(
                            f"tensor_parallel_degree ({tp}) must divide "
                            f"{layer_type} attention n_kv_heads ({n_kv_heads}) "
                            f"on layer {idx}."
                        )
            from torchtitan.models.gemma4.sharding import set_gemma4_sharding_config

            set_gemma4_sharding_config(
                self, enable_sp=config.parallelism.enable_sequence_parallel
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            attention_op_flops = sum(
                quadratic_attention_flops_per_token(
                    num_heads=layer.attention.n_heads,
                    qk_head_dim=layer.attention.head_dim
                    or (layer.attention.dim // layer.attention.n_heads),
                    v_head_dim=layer.attention.head_dim
                    or (layer.attention.dim // layer.attention.n_heads),
                    seq_len=seq_len
                    if layer.use_global_attention
                    else min(seq_len, self.sliding_window_size),
                )
                for layer in self.layers
            )
            return nparams, 6 * active_nparams + attention_op_flops

    def __init__(self, config: Config):
        super().__init__(config)
        self.embed_scale = config.dim**0.5
        self.sliding_window_size = config.sliding_window_size
        self.enable_sliding_window = config.enable_sliding_window

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ):
        h = (
            self.tok_embeddings(tokens) * self.embed_scale
            if self.tok_embeddings is not None
            else tokens
        )
        for layer in self.layers.values():
            h = layer(h, attention_masks, positions)
        h = self.norm(h) if self.norm is not None else h
        return h if self._skip_lm_head or self.lm_head is None else self.lm_head(h)

    def get_attention_masks(self, positions: torch.Tensor) -> AttentionMasksType | None:
        attn_config = self.config.first_attention
        if attn_config is None or not isinstance(
            attn_config.inner_attention, FlexAttention.Config
        ):
            return super().get_attention_masks(positions)

        global_mask = self._create_flex_attention_mask_for_document(
            positions, attn_config
        )
        if not self.enable_sliding_window:
            return global_mask

        sliding_mask = self._create_flex_attention_mask(
            positions,
            attn_config,
            [
                get_causal_mask_mod(),
                get_efficient_causal_mask_mod_for_packed_document(positions),
                get_sliding_window_mask_mod(self.sliding_window_size),
            ],
        )
        return {"global": global_mask, "sliding_window": sliding_mask}
