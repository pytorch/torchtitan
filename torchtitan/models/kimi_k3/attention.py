# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.models.common import Linear
from torchtitan.models.common.attention import (
    Attention,
    FlexAttentionMetadata,
    InnerAttention,
    local_head_split,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from torchtitan.models.common.linear import maybe_gather_tp_input
from torchtitan.models.common.nn_modules import RMSNorm

__all__ = ["KimiMLAAttention"]

# Shape suffixes:
# T = packed tokens, D = model dimension, C = projection channels, H = heads,
# K = query/key head dimension, V = value head dimension.


class KimiMLAAttention(Attention):
    """Kimi K3 multi-head latent attention.

    Unlike DeepSeek-V3 MLA, the released K3 configuration sets
    ``mla_use_nope=True``: the RoPE-sized query/key slices remain part of the
    projected head, but no rotary transform is applied, so this has no rope
    config at all. Attention delegates to the configured inner backend.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
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
        inner_attention: InnerAttention.Config = field(
            default_factory=VarlenInnerAttention.Config
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
        # q_norm reads the wq_a projection output with bare ops.
        remat.recompute_needs_tensor(q_TC)
        q_THK = local_head_split(
            self.wq_b(self.q_norm(q_TC)),
            self.q_head_dim,
            cp_shard_dim=0,
        )

        compressed_kv_TC = self.wkv_a(x_TD)
        # kv_norm and the rope expand read the wkv_a projection output with bare ops.
        remat.recompute_needs_tensor(compressed_kv_TC)
        kv_latent_TC, k_rope_TK = torch.split(
            compressed_kv_TC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_THC = local_head_split(
            self.wkv_b(self.kv_norm(kv_latent_TC)),
            self.qk_nope_head_dim + self.v_head_dim,
            cp_shard_dim=0,
        )
        k_nope_THK, v_THV = torch.split(
            kv_THC,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        # Headless rope slice broadcast onto the local heads, as in DeepSeek-V3's MLA.
        # The key concat reads the wkv_b projection output with bare ops.
        remat.recompute_needs_tensor(k_nope_THK)
        with spmd.local():
            k_rope_THK = k_rope_TK.unsqueeze(1).expand(-1, k_nope_THK.shape[-2], -1)
            k_THK = torch.cat((k_nope_THK, k_rope_THK), dim=-1)
            if spmd.is_type_checking():
                spmd.assert_type(
                    k_THK,
                    {"dp": spmd.S(0), "cp": spmd.S(0), "tp": spmd.S(1)},
                )

        out_THV = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            q_THK,
            k_THK,
            v_THV,
            attention_metadata=attention_metadata,
            scale=self.scale,
        )
        gate_TD = self.gate(x_TD)
        # The output gating reads the inner_attention and gate projection outputs
        # with bare ops.
        remat.recompute_needs_tensor(out_THV, gate_TD)
        out_TD = out_THV.flatten(-2)
        out_TD = out_TD * torch.sigmoid(gate_TD)
        return self.wo(out_TD)
