# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 multi-head latent attention backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    FlexInnerAttention,
    InnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.module import Module

__all__ = [
    "KimiMLAAttention",
    "MLAFlexInnerAttention",
    "MLAInnerAttention",
    "MLAVarlenInnerAttention",
]

# Shape suffixes:
# T = packed token count (num_tokens)
# D = model dimension (dim)
# H = attention head count (n_heads)
# L = low-rank KV latent dimension (kv_lora_rank)
# N = per-head non-positional key dimension (qk_nope_head_dim)
# R = head-shared key dimension (qk_rope_head_dim)
# V = per-head value dimension (v_head_dim)
# C = compressed KV channels (L + R)
# K = full per-head query/key dimension (q_head_dim = N + R)
# P = packed per-head KV channels (N + V)


class MLAInnerAttention(InnerAttention, ABC):
    """Inner attention accepting Kimi K3's packed KV and head-shared key."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        pass

    @abstractmethod
    def forward(
        self,
        q_THK: torch.Tensor,
        kv_THP: torch.Tensor,
        k_shared_TR: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run attention from Kimi K3's compact Q/KV representation."""

    @staticmethod
    def _materialize_kv(
        q_THK: torch.Tensor,
        kv_THP: torch.Tensor,
        k_shared_TR: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize per-head K/V from MLA's packed KV and shared key."""
        nope_head_dim = q_THK.shape[-1] - k_shared_TR.shape[-1]
        value_head_dim = kv_THP.shape[-1] - nope_head_dim
        k_nope_THN, v_THV = torch.split(kv_THP, [nope_head_dim, value_head_dim], dim=-1)
        k_shared_THR = k_shared_TR.unsqueeze(1).expand(-1, k_nope_THN.shape[1], -1)
        k_THK = torch.cat((k_nope_THN, k_shared_THR), dim=-1)
        return k_THK, v_THV


class MLAFlexInnerAttention(MLAInnerAttention, FlexInnerAttention):
    """Flex attention accepting Kimi K3's compact MLA inputs."""

    @dataclass(kw_only=True, slots=True)
    class Config(MLAInnerAttention.Config, FlexInnerAttention.Config):
        pass

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        q_THK: torch.Tensor,
        kv_THP: torch.Tensor,
        k_shared_TR: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        k_THK, v_THV = self._materialize_kv(q_THK, kv_THP, k_shared_TR)
        return FlexInnerAttention.forward(self, q_THK, k_THK, v_THV, **kwargs)


class MLAVarlenInnerAttention(MLAInnerAttention, VarlenInnerAttention):
    """Variable-length attention accepting Kimi K3's compact MLA inputs."""

    @dataclass(kw_only=True, slots=True)
    class Config(MLAInnerAttention.Config, VarlenInnerAttention.Config):
        pass

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        q_THK: torch.Tensor,
        kv_THP: torch.Tensor,
        k_shared_TR: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        k_THK, v_THV = self._materialize_kv(q_THK, kv_THP, k_shared_TR)
        return VarlenInnerAttention.forward(self, q_THK, k_THK, v_THV, **kwargs)


class KimiMLAAttention(BaseAttention):
    """Kimi K3 multi-head latent attention.

    Unlike DeepSeek-V3 MLA, the released K3 configuration sets
    ``mla_use_nope=True``: the RoPE-sized query/key slices remain part of the
    projected head, but no rotary transform is applied, so this has no rope
    config at all. Attention delegates to the configured inner backend.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        kv_lora_rank: int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        v_head_dim: int
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv_a: Linear.Config
        kv_norm: RMSNorm.Config
        wkv_b: Linear.Config
        gate: Linear.Config
        wo: Linear.Config
        inner_attention: Module.Config = field(
            default_factory=MLAFlexInnerAttention.Config
        )

        def __post_init__(self) -> None:
            BaseAttention.Config.__post_init__(self)
            if not isinstance(self.inner_attention, MLAInnerAttention.Config):
                raise ValueError(
                    "KimiMLAAttention requires an MLAInnerAttention.Config, "
                    f"but got {type(self.inner_attention).__qualname__}."
                )

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.scale = self.q_head_dim**-0.5

        self.wq_a = config.wq_a.build()
        self.q_norm = config.q_norm.build()
        self.wq_b = config.wq_b.build()
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.gate = config.gate.build()
        self.wo = config.wo.build()
        inner_attention = config.inner_attention.build()
        assert isinstance(inner_attention, MLAInnerAttention)
        self.inner_attention = inner_attention

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions

        num_tokens = x_TD.shape[0]
        q_THK = self.wq_b(self.q_norm(self.wq_a(x_TD))).view(
            num_tokens, self.n_heads, self.q_head_dim
        )

        compressed_kv_TC = self.wkv_a(x_TD)
        kv_latent_TL, k_shared_TR = torch.split(
            compressed_kv_TC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_THP = self.wkv_b(self.kv_norm(kv_latent_TL)).view(
            num_tokens,
            self.n_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        out_THV = self.inner_attention(
            q_THK,
            kv_THP,
            k_shared_TR,
            attention_masks=attention_masks,
            scale=self.scale,
        )
        out_TD = out_THV.reshape(num_tokens, self.n_heads * self.v_head_dim)
        out_TD = out_TD * torch.sigmoid(self.gate(x_TD))
        return self.wo(out_TD)
