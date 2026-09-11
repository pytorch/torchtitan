# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel KDA stages backed by Attention Gym."""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist
from attn_gym.linear.context_parallel import (
    _ContextParallelChunk,
    context_parallel_conv_history,
    ContextParallelRouting,
)
from attn_gym.linear.kda.context_parallel import context_parallel_kda

from torchtitan.distributed.context_parallel import get_token_fragments
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group

from torchtitan.models.common.attention import KDAAttentionMetadata
from torchtitan.models.common.attention.kda import InnerKDA
from torchtitan.models.common.cp_attention import CPInnerAttention

spmd.register_local_autograd_function(_ContextParallelChunk)


@dataclass(frozen=True, slots=True)
class ContextParallelKDAAttentionMetadata(KDAAttentionMetadata):
    """KDA metadata prepared for rank-local context-parallel execution."""

    cp_routing: ContextParallelRouting

    _ROUTING_SPMD_TYPE = spmd.SpmdType(
        {
            MeshAxisName.DP: spmd.V,
            MeshAxisName.CP: spmd.V,
            MeshAxisName.TP: spmd.R,
        }
    )

    def annotate_spmd_types(self) -> None:
        """Annotate sequence offsets and rank-local CP routing tensors."""
        KDAAttentionMetadata.annotate_spmd_types(self)
        for value in vars(self.cp_routing).values():
            if isinstance(value, torch.Tensor):
                spmd.assert_type(value, self._ROUTING_SPMD_TYPE)


class ContextParallelInnerKDA(
    CPInnerAttention[KDAAttentionMetadata, ContextParallelKDAAttentionMetadata],
    InnerKDA,
):
    """Inner KDA with distributed convolution and recurrent-state plumbing."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, InnerKDA.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: KDAAttentionMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> ContextParallelKDAAttentionMetadata:
        """Build rank-local routing from global KDA sequence metadata."""
        if attention_metadata.varlen is None:
            if permutation is None:
                raise ValueError(
                    "KDA context parallelism requires global "
                    "VarlenAttentionMetadata when using contiguous CP sharding."
                )
            cu_seqlens_global = [0, permutation.shape[1]]
            device = permutation.device
        else:
            cu_seqlens_global = attention_metadata.varlen.cu_seq_q.tolist()
            device = attention_metadata.varlen.cu_seq_q.device

        group = spmd_mesh_group(MeshAxisName.CP)
        if group is None:
            raise RuntimeError(
                "KDA metadata preparation requires an active multi-rank CP mesh axis."
            )
        routing = ContextParallelRouting.from_fragments(
            cu_seqlens_global=cu_seqlens_global,
            fragments=get_token_fragments(
                cu_seqlens_global[-1],
                cp_size=group.size(),
                permutation=permutation,
            ),
            cp_rank=dist.get_rank(group),
            device=device,
            conv_history=attention_metadata.num_conv_history_tokens,
        )
        return ContextParallelKDAAttentionMetadata(
            varlen=attention_metadata.varlen,
            num_conv_history_tokens=attention_metadata.num_conv_history_tokens,
            cp_routing=routing,
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
        attention_metadata: KDAAttentionMetadata | None,
    ) -> torch.Tensor:
        if not isinstance(attention_metadata, ContextParallelKDAAttentionMetadata):
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
