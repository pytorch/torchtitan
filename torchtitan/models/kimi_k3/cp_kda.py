# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel KDA stages backed by Attention Gym."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
from attn_gym.linear.context_parallel import (
    context_parallel_conv_history,
    ContextParallelRouting,
)
from attn_gym.linear.kda.context_parallel import context_parallel_kda

from torchtitan.distributed.context_parallel import get_token_fragments
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.cp_attention import CPInnerAttention

from .kda import InnerKDA, KDAAttentionMetadata


class ContextParallelInnerKDA(
    CPInnerAttention[KDAAttentionMetadata, KDAAttentionMetadata], InnerKDA
):
    """Inner KDA with distributed convolution and recurrent-state plumbing."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, InnerKDA.Config):
        pass

    @classmethod
    def prepare_cp_batch_metadata(
        cls,
        input_dict: dict[str, Any],
        *,
        permutation: torch.Tensor | None,
        config: CPInnerAttention.Config,
    ) -> dict[str, Any]:
        """Prepare the KDA metadata stored in the model inputs."""
        context_metadata = input_dict.get("attention_masks")
        if context_metadata is None:
            return input_dict
        if not isinstance(context_metadata, Mapping):
            raise ValueError(
                "KDA context parallelism requires mapping context metadata."
            )
        kda_metadata = context_metadata.get("kda")
        if not isinstance(kda_metadata, KDAAttentionMetadata):
            raise ValueError(
                "KDA context parallelism requires KDAAttentionMetadata in "
                "context_metadata['kda']."
            )
        input_dict["attention_masks"] = {
            **context_metadata,
            "kda": cls.prepare_cp_metadata(
                kda_metadata,
                permutation=permutation,
                config=config,
            ),
        }
        return input_dict

    @staticmethod
    def prepare_cp_metadata(
        context_metadata: KDAAttentionMetadata,
        *,
        permutation: torch.Tensor | None,
        config: CPInnerAttention.Config,
    ) -> KDAAttentionMetadata:
        """Build rank-local routing from global KDA sequence metadata."""
        assert isinstance(config, ContextParallelInnerKDA.Config)
        kda_metadata = context_metadata
        if kda_metadata.varlen is None:
            if permutation is None:
                raise ValueError(
                    "KDA context parallelism requires global VarlenMetadata when "
                    "using contiguous CP sharding."
                )
            cu_seqlens_global = [0, permutation.shape[1]]
            device = permutation.device
        else:
            cu_seqlens_global = kda_metadata.varlen.cu_seq_q.tolist()
            device = kda_metadata.varlen.cu_seq_q.device

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
            conv_history=config.conv_kernel_size - 1,
        )
        return replace(kda_metadata, cp_routing=routing)

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
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        # The shared KDA interface passes global packed-sequence offsets. CP
        # stages instead use the rank-local boundaries stored in routing.
        del cu_seqlens
        if routing is None:
            raise ValueError("KDA context parallelism requires per-batch routing.")
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
