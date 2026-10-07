# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 multi-head latent attention backends."""

from dataclasses import dataclass, field

import torch
import torch_remat as remat

from torchtitan.models.common.attention import (
    BaseAttention,
    FlexAttentionMetadata,
    local_head_split,
    MLAFlexInnerAttention,
    MLAInnerAttention,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.module import Module

__all__ = [
    "KimiMLAAttention",
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
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_metadata: FlexAttentionMetadata
        | VarlenAttentionMetadata
        | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions

        # The MLA and gate projections all consume x. Gather once at their
        # common attention boundary.
        x_TD = maybe_gather_tp_input(self, x_TD)

        q_TC = self.wq_a(x_TD)
        remat.recompute_needs_tensor(q_TC)
        q_THK = local_head_split(
            self.wq_b(self.q_norm(q_TC)), self.q_head_dim, cp_shard_dim=0
        )

        compressed_kv_TC = self.wkv_a(x_TD)
        remat.recompute_needs_tensor(compressed_kv_TC)
        kv_latent_TC, k_shared_TR = torch.split(
            compressed_kv_TC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_THP = local_head_split(
            self.wkv_b(self.kv_norm(kv_latent_TC)),
            self.qk_nope_head_dim + self.v_head_dim,
            cp_shard_dim=0,
        )
        out_THV = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            q_THK,
            kv_THP,
            k_shared_TR,
            attention_metadata=attention_metadata,
            scale=self.scale,
        )
        gate_TD = self.gate(x_TD)
        remat.recompute_needs_tensor(out_THV, gate_TD)
        out_TD = out_THV.flatten(-2)
        out_TD = out_TD * torch.sigmoid(gate_TD)
        return self.wo(out_TD)
