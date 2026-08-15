# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffix legend (see attention.py):
#   B = batch, L = sequence length, D = model dimension, C = packed QKV channels,
#   N = num heads, K = key/value head dimension (KDA uses K == V), W = conv width

from dataclasses import dataclass
from typing import Any, get_args, Literal

import torch
from torch import nn

from torchtitan.models.common.attention.attention import (
    AttentionMasksType,
    VarlenMetadata,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv1d, RMSNorm
from torchtitan.protocols.module import Module

try:
    from attn_gym.linear.kda.fwd.cute import chunk_kda
    from attn_gym.linear.kda.fwd.recurrent import recurrent_kda
    from attn_gym.linear.kda.fwd.triton.gate_fwd import bounded_gate_cumsum
    from attn_gym.linear.kda.fwd.triton.l2norm_fwd import l2norm
    from attn_gym.linear.kda.short_conv import cute_causal_conv1d_silu
except ImportError:
    chunk_kda: Any = None
    recurrent_kda: Any = None
    bounded_gate_cumsum: Any = None
    l2norm: Any = None
    cute_causal_conv1d_silu: Any = None


__all__ = ["KDA", "KDAAttention", "KDABackend", "KDAInnerAttention"]

KDABackend = Literal["chunked", "recurrent"]


class KDAInnerAttention(Module):
    """Stateless dispatch to the delta-rule core."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # "chunked": parallel within chunks, for training (default)
        # "recurrent": token at a time, inference only, no backward
        backend: KDABackend = "chunked"

        def __post_init__(self):
            valid = get_args(KDABackend)
            if self.backend not in valid:
                raise ValueError(
                    f"unknown KDA backend {self.backend!r}. "
                    f"Valid: {', '.join(repr(name) for name in valid)}."
                )

    def __init__(self, config: Config):
        super().__init__()
        if chunk_kda is None:
            raise ImportError(
                "KDA requires attention-gym, an optional dependency: "
                "pip install attention-gym"
            )
        self.backend = config.backend

    def forward(
        self,
        q_BLNK: torch.Tensor,
        k_BLNK: torch.Tensor,
        v_BLNK: torch.Tensor,
        g_BLNK: torch.Tensor,
        beta_BLN: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the gated delta rule.

        ``g_BLNK`` is the per-channel log2 decay: chunk-local cumulative for
        ``"chunked"``, per token for ``"recurrent"``.
        """
        core = chunk_kda if self.backend == "chunked" else recurrent_kda
        output, _ = core(
            q_BLNK,
            k_BLNK,
            v_BLNK,
            g_BLNK,
            beta_BLN,
            cu_seqlens=cu_seqlens,
        )
        return output


class KDAAttention(Module):
    """Dense inner KDA implementation shared with the paged vLLM boundary.

    Owns the depthwise convolution, the q/k L2 norm, the gate map, and the
    recurrence. Holds no parameters: the outer :class:`KDA` passes the conv weight
    and the gate parameters in, so an inference wrapper can replace this module
    without moving state. It is also the single DTensor-to-local boundary for the
    head-parallel region, so it computes on rank-local head shards.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int = 128
        gate_lower_bound: float = -5.0
        inner_attention: KDAInnerAttention.Config

        def __post_init__(self):
            if self.gate_lower_bound >= 0:
                raise ValueError(
                    f"gate_lower_bound must be negative, got {self.gate_lower_bound}"
                )

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.gate_lower_bound = config.gate_lower_bound
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        mixed_qkv_BLC: torch.Tensor,
        raw_gate_BLNK: torch.Tensor,
        raw_beta_BLN: torch.Tensor,
        conv_weight_C1W: torch.Tensor,
        A_log_N: torch.Tensor,
        dt_bias_NK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run convolution, gate map, and recurrence on rank-local head shards.

        ``mixed_qkv_BLC`` is the packed QKV projection before the convolution;
        ``raw_gate_BLNK`` and ``raw_beta_BLN`` are the projections before the gate
        map and the sigmoid.
        """
        B, L, _ = mixed_qkv_BLC.shape
        conv_output_BLC = cute_causal_conv1d_silu(
            mixed_qkv_BLC,
            conv_weight_C1W[:, 0],
            cu_seqlens=cu_seqlens,
        )
        q_BLNK, k_BLNK, v_BLNK = (
            tensor.contiguous()
            for tensor in conv_output_BLC.view(B, L, -1, 3, self.head_dim).unbind(dim=3)
        )

        return self.inner_attention(
            l2norm(q_BLNK),
            l2norm(k_BLNK),
            v_BLNK,
            self._gate(raw_gate_BLNK, A_log_N, dt_bias_NK, cu_seqlens),
            raw_beta_BLN.float().sigmoid(),
            cu_seqlens=cu_seqlens,
        )

    def _gate(
        self,
        raw_gate_BLNK: torch.Tensor,
        A_log_N: torch.Tensor,
        dt_bias_NK: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Map the raw gate projection to a log2 decay.

        ``chunk_size=1`` makes the cumulative sum an identity, which is what the
        recurrent backend wants; the chunked core wants it accumulated within each
        64-token chunk. The prefix sums are sequence-local, so ``cu_seqlens`` has
        to reach here too: a document starting mid-chunk would otherwise inherit
        its predecessor's gate prefix.
        """
        return bounded_gate_cumsum(
            raw_gate_BLNK.to(torch.bfloat16).contiguous(),
            A_log_N.float(),
            dt_bias_NK.float(),
            chunk_size=64 if self.inner_attention.backend == "chunked" else 1,
            lower_bound=self.gate_lower_bound,
            cu_seqlens=cu_seqlens,
        )


class KDA(Module):
    """KDA (Kimi Delta Attention) linear-attention layer.

    Recurrent state and per channel gated delta rule::

        x -> fused QKV + low-rank gate/beta projections
          -> attention: conv + SiLU -> L2-norm q/k -> gate map -> delta rule
          -> sigmoid-gated RMSNorm -> output projection
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_heads: int
        head_dim: int = 128
        in_proj_qkv: Linear.Config
        conv_qkv: Conv1d.Config
        gate_proj_a: Linear.Config
        gate_proj_b: Linear.Config
        beta_proj: Linear.Config
        out_gate_proj_a: Linear.Config
        out_gate_proj_b: Linear.Config
        out_norm: RMSNorm.Config
        out_proj: Linear.Config
        # Module.Config, not KDAAttention.Config: this is the slot an inference
        # wrapper substitutes its own module into.
        attention: Module.Config

        def __post_init__(self):
            if self.num_heads < 1:
                raise ValueError(f"num_heads must be positive, got {self.num_heads}")

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.in_proj_qkv = config.in_proj_qkv.build()
        self.conv_qkv = config.conv_qkv.build()
        self.gate_proj_a = config.gate_proj_a.build()
        self.gate_proj_b = config.gate_proj_b.build()
        self.beta_proj = config.beta_proj.build()
        self.out_gate_proj_a = config.out_gate_proj_a.build()
        self.out_gate_proj_b = config.out_gate_proj_b.build()
        self.out_norm = config.out_norm.build()
        self.out_proj = config.out_proj.build()
        self.attention = config.attention.build()

        self.A_log = nn.Parameter(torch.empty(config.num_heads))
        self.dt_bias = nn.Parameter(torch.empty(config.num_heads, config.head_dim))

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        B, L, _ = x_BLD.shape
        cu_seqlens = None
        if isinstance(attention_masks, VarlenMetadata):
            cu_seqlens = attention_masks.cu_seq_q

        # add back B = 1 for packed documents
        kernel_B, kernel_L = (B, L) if cu_seqlens is None else (1, B * L)

        def flatten(tensor: torch.Tensor) -> torch.Tensor:
            if cu_seqlens is None:
                return tensor
            return tensor.reshape(1, B * L, *tensor.shape[2:])

        raw_gate_BLNK = self.gate_proj_b(self.gate_proj_a(x_BLD)).reshape(
            kernel_B, kernel_L, -1, self.head_dim
        )
        out_BLNK = self.attention(
            flatten(self.in_proj_qkv(x_BLD)),
            raw_gate_BLNK,
            flatten(self.beta_proj(x_BLD)).reshape(kernel_B, kernel_L, -1),
            self.conv_qkv.weight,
            self.A_log,
            self.dt_bias,
            cu_seqlens=cu_seqlens,
        )
        out_BLNK = self._gated_norm(out_BLNK, flatten(x_BLD))
        # Merge heads and, under varlen, unpack the flattened batch.
        return self.out_proj(out_BLNK.reshape(B, L, -1))

    def _gated_norm(self, out_BLNK: torch.Tensor, x_BLD: torch.Tensor) -> torch.Tensor:
        """Per-head RMSNorm scaled by a sigmoid gate (Kimi gates with ``sigmoid``)"""
        normed = self.out_norm(out_BLNK)
        gate = self.out_gate_proj_b(self.out_gate_proj_a(x_BLD)).view_as(normed)
        return normed * gate.sigmoid()
