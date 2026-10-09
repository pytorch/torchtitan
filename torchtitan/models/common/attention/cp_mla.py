# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel execution for compact multi-head latent attention."""

from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd
import torch

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group

from .attention import FlexInnerAttention
from .cp_attention import (
    all_gather_cp_tensors,
    KVAllGatherFlexAttention,
    UlyssesAttention,
)
from .mla import materialize_mla_kv, MLAAttention

__all__ = [
    "KVAllGatherMLAFlexAttention",
    "UlyssesMLAAttention",
]

# Shape suffixes:
# T = packed token count (num_tokens)
# H = attention head count (n_heads)
# R = head-shared key dimension (qk_rope_head_dim)
# V = per-head value dimension (v_head_dim)
# K = full per-head query/key dimension
# P = packed per-head KV channels
# F = fused communication channels (K + P)

_TOKEN_DIM = 0
_HEAD_DIM = 1


class KVAllGatherMLAFlexAttention(
    KVAllGatherFlexAttention,
    MLAAttention,
):
    """All-gather compact MLA latents before projecting and materializing K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(KVAllGatherFlexAttention.Config, MLAAttention.Config):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

        def __post_init__(self) -> None:
            if not isinstance(self.inner_attention, FlexInnerAttention.Config):
                raise ValueError(
                    "KVAllGatherMLAFlexAttention requires a FlexInnerAttention.Config."
                )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.attention_metadata_key = type(self)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]

    def forward(
        self,
        q_THK: torch.Tensor,
        kv_c_normed_TL: torch.Tensor,
        k_shared_TR: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        kv_c_normed_TL, k_shared_TR = all_gather_cp_tensors(
            (kv_c_normed_TL, k_shared_TR), reduce_dtype=self.reduce_dtype
        )
        # Gathering compressed KV reduces communication volume. Each rank then
        # repeats the KV projection and shared-key expansion locally, trading
        # extra computation for the typically more expensive communication.
        return MLAAttention.forward(self, q_THK, kv_c_normed_TL, k_shared_TR, **kwargs)


class UlyssesMLAAttention(
    UlyssesAttention,
    MLAAttention,
):
    """MLA attention with headless shared-key communication."""

    @dataclass(kw_only=True, slots=True)
    class Config(
        UlyssesAttention.Config,
        MLAAttention.Config,
    ):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.attention_metadata_key = type(self)

    def forward(
        self,
        q_THK: torch.Tensor,
        kv_c_normed_TL: torch.Tensor,
        k_shared_TR: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "CP attention requires an active multi-rank CP mesh axis."
            )

        kv_THP = self.project_mla_kv(kv_c_normed_TL)

        # Q and packed KV have heads, so move their CP shard from T to H with
        # one all-to-all. The shared key has no H dimension and is all-gathered.
        qk_head_dim = q_THK.shape[-1]
        packed_q_kv_THF = torch.cat((q_THK, kv_THP), dim=-1)
        packed_q_kv_THF = spmd.redistribute(
            packed_q_kv_THF,
            cp_group,
            src=spmd.S(_TOKEN_DIM),
            dst=spmd.S(_HEAD_DIM),
        )
        q_THK, kv_THP = torch.split(
            packed_q_kv_THF,
            [qk_head_dim, kv_THP.shape[-1]],
            dim=-1,
        )
        k_shared_TR = spmd.redistribute(
            k_shared_TR,
            cp_group,
            src=spmd.S(_TOKEN_DIM),
            dst=spmd.R,
            backward_options={"op_dtype": k_shared_TR.dtype},
        )

        k_THK, v_THV = materialize_mla_kv(q_THK, kv_THP, k_shared_TR)
        out_THV = self.inner_attention(q_THK, k_THK, v_THV, **kwargs)
        return spmd.redistribute(
            out_THV,
            cp_group,
            src=spmd.S(_HEAD_DIM),
            dst=spmd.S(_TOKEN_DIM),
        )
