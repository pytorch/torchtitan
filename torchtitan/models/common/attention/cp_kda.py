# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel KDA stages backed by Attention Gym."""

from dataclasses import dataclass

import spmd_types as spmd
import torch

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attn_gym import (
    context_parallel_conv_history,
    context_parallel_kda,
    ContextParallelRouting,
    require_attn_gym,
)

from .cp_attention import CPInnerAttention, Ulysses
from .cp_linear_attention import (
    ContextParallelLinearAttentionMetadata,
    prepare_cp_linear_attention_metadata,
    prepare_ulysses_linear_attention_metadata,
    UlyssesLinearAttentionMetadata,
)
from .kda import InnerKDA
from .linear_attention import LinearAttentionMetadata

# Shape suffixes:
# T = tokens, C = channels, H = heads, K/V = head dimensions,
# F = packed features per head, W = convolution kernel width.


class UlyssesInnerKDA(Ulysses[LinearAttentionMetadata], InnerKDA):
    """Run convolution and KDA on full sequences with CP-sharded heads."""

    @dataclass(kw_only=True, slots=True)
    class Config(Ulysses.Config, InnerKDA.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: LinearAttentionMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> LinearAttentionMetadata:
        """Prepare Ulysses KDA metadata in original token order."""
        return prepare_ulysses_linear_attention_metadata(
            attention_metadata, permutation=permutation
        )

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        raw_gate_THK: torch.Tensor,
        raw_beta_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        attention_metadata: LinearAttentionMetadata | None,
    ) -> torch.Tensor:
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "Ulysses KDA requires an active multi-rank CP mesh axis."
            )
        packed_THF = torch.cat(
            (
                query_TC.unflatten(-1, (-1, self.head_dim)),
                key_TC.unflatten(-1, (-1, self.head_dim)),
                value_TC.unflatten(-1, (-1, self.head_dim)),
                raw_gate_THK,
                raw_beta_TH.unsqueeze(-1),
            ),
            dim=-1,
        )
        packed_THF = spmd.redistribute(
            packed_THF, cp_group, src=spmd.S(0), dst=spmd.S(1)
        )
        if isinstance(attention_metadata, UlyssesLinearAttentionMetadata):
            # All-to-all gathers the shared CP order, not chronological order.
            packed_THF = packed_THF.index_select(
                0, attention_metadata.inverse_permutation
            )
        q_THK, k_THK, v_THV, raw_gate_THK, beta_TH1 = packed_THF.split(
            [self.head_dim] * 4 + [1], dim=-1
        )
        # Parameters remain CP-replicated outside this region. Shard their heads
        # locally; backward contributes zeros for heads owned by other CP ranks.
        parameters = tuple(
            spmd.shard(parameter, cp_group, src=spmd.R, dst=spmd.S(0))
            for parameter in (
                conv_q_weight_C1W,
                conv_k_weight_C1W,
                conv_v_weight_C1W,
                A_log_H,
                dt_bias_HK,
            )
        )
        out_THV = super().forward(
            q_THK.flatten(-2),
            k_THK.flatten(-2),
            v_THV.flatten(-2),
            raw_gate_THK,
            beta_TH1.squeeze(-1),
            *parameters,
            attention_metadata=attention_metadata,
        )
        if isinstance(attention_metadata, UlyssesLinearAttentionMetadata):
            out_THV = out_THV.index_select(0, attention_metadata.permutation)
        return spmd.redistribute(out_THV, cp_group, src=spmd.S(1), dst=spmd.S(0))


class ContextParallelInnerKDA(
    CPInnerAttention[
        LinearAttentionMetadata,
        ContextParallelLinearAttentionMetadata,
    ],
    InnerKDA,
):
    """Inner KDA with distributed convolution and recurrent-state plumbing."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, InnerKDA.Config):
        pass

    def __init__(self, config: Config):
        # Every attn_gym export this module calls; keep the list in sync.
        require_attn_gym(
            context_parallel_kda, context_parallel_conv_history, ContextParallelRouting
        )
        super().__init__(config)

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: LinearAttentionMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> ContextParallelLinearAttentionMetadata:
        """Build rank-local routing from global KDA sequence metadata."""
        return prepare_cp_linear_attention_metadata(
            attention_metadata, permutation=permutation, family="KDA"
        )

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        raw_gate_THK: torch.Tensor,
        raw_beta_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        attention_metadata: LinearAttentionMetadata | None,
    ) -> torch.Tensor:
        if not isinstance(attention_metadata, ContextParallelLinearAttentionMetadata):
            raise ValueError(
                "KDA context parallelism requires rank-local attention metadata."
            )
        routing = attention_metadata.cp_routing
        if routing.tail_sources.shape[1] != conv_q_weight_C1W.shape[-1] - 1:
            raise ValueError(
                "KDA context-parallel routing convolution history must match "
                "the model's convolution width minus one."
            )
        return self.run_stages(
            query_TC,
            key_TC,
            value_TC,
            raw_gate_THK,
            raw_beta_TH,
            conv_q_weight_C1W,
            conv_k_weight_C1W,
            conv_v_weight_C1W,
            A_log_H,
            dt_bias_HK,
            cu_seqlens=routing.cu_seqlens,
            routing=routing,
        )

    def short_convolution(
        self,
        qkv_1TC: torch.Tensor,
        conv_weight_C1W: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        assert routing is not None, "CP forward must validate routing."
        assert (
            initial_state is None
        ), "KDA context parallelism constructs the convolution history."
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "KDA context parallelism requires an active multi-rank CP mesh axis."
            )
        initial_state = context_parallel_conv_history(
            qkv_1TC,
            routing,
            cp_group,
        )
        return super().short_convolution(
            qkv_1TC,
            conv_weight_C1W,
            initial_state,
            cu_seqlens=cu_seqlens,
            routing=routing,
        )

    def kda_core(
        self,
        q_1THK: torch.Tensor,
        k_1THK: torch.Tensor,
        v_1THV: torch.Tensor,
        raw_gate_1THK: torch.Tensor,
        raw_beta_1TH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        # CP routing contains the rank-local sequence boundaries. Ignore the
        # global packed-sequence offsets passed through the regular KDA interface.
        del cu_seqlens
        assert routing is not None, "CP forward must validate routing."
        q_1THK, k_1THK, gate_1THK, beta_1TH = self.kernel.prepare_inputs(
            q_1THK,
            k_1THK,
            raw_gate_1THK,
            raw_beta_1TH,
            A_log_H,
            dt_bias_HK,
        )
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "KDA context parallelism requires an active multi-rank CP mesh axis."
            )
        output_1THV, _ = context_parallel_kda(
            q_1THK,
            k_1THK,
            v_1THV,
            gate_1THK,
            beta_1TH,
            routing=routing,
            group=cp_group,
        )
        return output_1THV
