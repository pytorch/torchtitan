# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel Gated DeltaNet backed by Attention Gym."""

from dataclasses import dataclass

import spmd_types as spmd
import torch

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attn_gym import (
    context_parallel_conv_history,
    context_parallel_gdn,
    ContextParallelRouting,
    l2norm,
    require_attn_gym,
)

from .cp_attention import CPInnerAttention, Ulysses
from .cp_linear_attention import (
    ContextParallelLinearAttentionMetadata,
    prepare_cp_linear_attention_metadata,
    prepare_ulysses_linear_attention_metadata,
    UlyssesLinearAttentionMetadata,
)
from .gdn import InnerGDN
from .linear_attention import LinearAttentionMetadata

# Shape suffixes: T = tokens, C = channels, H = heads, K/V = head dimensions,
# F = packed features per head, W = convolution kernel width.
# Q/K and V may have different head counts.


class UlyssesInnerGDN(Ulysses[LinearAttentionMetadata], InnerGDN):
    """Run convolution and GDN on full sequences with CP-sharded heads."""

    @dataclass(kw_only=True, slots=True)
    class Config(Ulysses.Config, InnerGDN.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: LinearAttentionMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> LinearAttentionMetadata:
        """Prepare Ulysses GDN metadata in original token order."""
        return prepare_ulysses_linear_attention_metadata(
            attention_metadata, permutation=permutation
        )

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
        attention_metadata: LinearAttentionMetadata | None,
    ) -> torch.Tensor:
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "Ulysses GDN requires an active multi-rank CP mesh axis."
            )
        packed_qk_THF = torch.cat(
            (
                query_TC.unflatten(-1, (-1, key_head_dim)),
                key_TC.unflatten(-1, (-1, key_head_dim)),
            ),
            dim=-1,
        )
        packed_qk_THF = spmd.redistribute(
            packed_qk_THF, cp_group, src=spmd.S(0), dst=spmd.S(1)
        )
        if isinstance(attention_metadata, UlyssesLinearAttentionMetadata):
            packed_qk_THF = packed_qk_THF.index_select(
                0, attention_metadata.inverse_permutation
            )
        q_THK, k_THK = packed_qk_THF.split([key_head_dim] * 2, dim=-1)
        packed_vab_THF = torch.cat(
            (
                value_TC.unflatten(-1, (-1, value_head_dim)),
                a_TH.unsqueeze(-1),
                b_TH.unsqueeze(-1),
            ),
            dim=-1,
        )
        packed_vab_THF = spmd.redistribute(
            packed_vab_THF, cp_group, src=spmd.S(0), dst=spmd.S(1)
        )
        if isinstance(attention_metadata, UlyssesLinearAttentionMetadata):
            packed_vab_THF = packed_vab_THF.index_select(
                0, attention_metadata.inverse_permutation
            )
        v_THV, a_TH1, b_TH1 = packed_vab_THF.split([value_head_dim, 1, 1], dim=-1)
        parameters = tuple(
            spmd.shard(parameter, cp_group, src=spmd.R, dst=spmd.S(0))
            for parameter in (
                conv_q_weight_C1W,
                conv_k_weight_C1W,
                conv_v_weight_C1W,
                A_log_H,
                dt_bias_H,
            )
        )
        out_THV = super().forward(
            q_THK.flatten(-2),
            k_THK.flatten(-2),
            v_THV.flatten(-2),
            a_TH1.squeeze(-1),
            b_TH1.squeeze(-1),
            *parameters,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            attention_metadata=attention_metadata,
        )
        if isinstance(attention_metadata, UlyssesLinearAttentionMetadata):
            out_THV = out_THV.index_select(0, attention_metadata.permutation)
        return spmd.redistribute(out_THV, cp_group, src=spmd.S(1), dst=spmd.S(0))


class ContextParallelInnerGDN(
    CPInnerAttention[
        LinearAttentionMetadata,
        ContextParallelLinearAttentionMetadata,
    ],
    InnerGDN,
):
    """Gated DeltaNet with distributed convolution and recurrent-state routing."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, InnerGDN.Config):
        pass

    def __init__(self, config: Config):
        # Every attn_gym export this module calls; keep the list in sync.
        require_attn_gym(
            context_parallel_gdn,
            context_parallel_conv_history,
            ContextParallelRouting,
            l2norm,
        )
        super().__init__(config)

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: LinearAttentionMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> ContextParallelLinearAttentionMetadata:
        """Build rank-local routing from global Gated DeltaNet sequence metadata."""
        return prepare_cp_linear_attention_metadata(
            attention_metadata, permutation=permutation, family="Gated DeltaNet"
        )

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
        attention_metadata: LinearAttentionMetadata | None,
    ) -> torch.Tensor:
        if not isinstance(attention_metadata, ContextParallelLinearAttentionMetadata):
            raise ValueError(
                "Gated DeltaNet context parallelism requires rank-local "
                "attention metadata."
            )
        routing = attention_metadata.cp_routing
        if routing.tail_sources.shape[1] != conv_q_weight_C1W.shape[-1] - 1:
            raise ValueError(
                "Gated DeltaNet context-parallel routing convolution history "
                "must match the model's convolution width minus one."
            )
        return self.run_stages(
            query_TC,
            key_TC,
            value_TC,
            a_TH,
            b_TH,
            conv_q_weight_C1W,
            conv_k_weight_C1W,
            conv_v_weight_C1W,
            A_log_H,
            dt_bias_H,
            cu_seqlens=routing.cu_seqlens,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            routing=routing,
        )

    def short_convolution(
        self,
        x_TC: torch.Tensor,
        weight_C1W: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        assert routing is not None, "CP forward must validate routing."
        assert (
            initial_state is None
        ), "Gated DeltaNet context parallelism constructs convolution history."
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "Gated DeltaNet context parallelism requires an active "
                "multi-rank CP mesh axis."
            )
        initial_state = context_parallel_conv_history(
            x_TC.unsqueeze(0), routing, cp_group
        )
        return super().short_convolution(
            x_TC,
            weight_C1W,
            initial_state,
            cu_seqlens=cu_seqlens,
            routing=routing,
        )

    def gdn_core(
        self,
        xq_THK: torch.Tensor,
        xk_THK: torch.Tensor,
        xv_THV: torch.Tensor,
        g_TH: torch.Tensor,
        beta_TH: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        del cu_seqlens
        assert routing is not None, "CP forward must validate routing."
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "Gated DeltaNet context parallelism requires an active "
                "multi-rank CP mesh axis."
            )
        normalized_q_1THK = l2norm(xq_THK.unsqueeze(0), cu_seqlens=routing.cu_seqlens)
        normalized_k_1THK = l2norm(xk_THK.unsqueeze(0), cu_seqlens=routing.cu_seqlens)
        output_1THV, _ = context_parallel_gdn(
            normalized_q_1THK,
            normalized_k_1THK,
            xv_THV.unsqueeze(0),
            g_TH.unsqueeze(0),
            beta_TH.unsqueeze(0),
            routing=routing,
            group=cp_group,
            scale=xq_THK.shape[-1] ** -0.5,
        )
        return output_1THV.squeeze(0)
