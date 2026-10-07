# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel Gated DeltaNet backed by Attention Gym."""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist
from attn_gym.linear import l2norm
from attn_gym.linear.context_parallel import (
    _ContextParallelChunk,
    context_parallel_conv_history,
    ContextParallelRouting,
)
from attn_gym.linear.gdn.context_parallel import context_parallel_gdn

from torchtitan.distributed.context_parallel import get_token_fragments
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group

from .attention import LinearAttentionMetadata
from .cp_attention import CPInnerAttention
from .cp_linear_attention import ContextParallelLinearAttentionMetadata
from .gdn import InnerGDN

spmd.register_local_autograd_function(_ContextParallelChunk)


class ContextParallelInnerGatedDeltaNet(
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

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: LinearAttentionMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> ContextParallelLinearAttentionMetadata:
        """Build rank-local routing from global Gated DeltaNet metadata."""
        if attention_metadata.varlen is None:
            if permutation is None:
                raise ValueError(
                    "Gated DeltaNet context parallelism requires global "
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
                "Gated DeltaNet metadata preparation requires an active "
                "multi-rank CP mesh axis."
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
        return ContextParallelLinearAttentionMetadata(
            varlen=attention_metadata.varlen,
            num_conv_history_tokens=attention_metadata.num_conv_history_tokens,
            cp_routing=routing,
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
        normalized_q_1THK = l2norm(  # pyrefly: ignore[not-callable]
            xq_THK.unsqueeze(0), cu_seqlens=routing.cu_seqlens
        )
        normalized_k_1THK = l2norm(  # pyrefly: ignore[not-callable]
            xk_THK.unsqueeze(0), cu_seqlens=routing.cu_seqlens
        )
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
