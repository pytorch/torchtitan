# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel backends for Kimi K3 multi-head latent attention.

Tensor suffixes: ``T`` tokens, ``H`` heads, ``K`` qk head dim, ``V`` v head dim,
``P`` packed MLA K/V channels, ``R`` head-shared key channels, and ``F`` fused
communication features.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attention import (
    FlexAttentionMetadata,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.cp_attention import (
    all_gather_cp_tensors,
    CPInnerAttention,
    KVAllGatherCPFlexInnerAttention,
    UlyssesCPInnerAttention,
)

from .attention import MLAFlexInnerAttention, MLAInnerAttention, MLAVarlenInnerAttention

__all__ = [
    "KVAllGatherCPMLAFlexInnerAttention",
    "UlyssesCPMLAFlexInnerAttention",
    "UlyssesCPMLAVarlenInnerAttention",
]

_TOKEN_DIM = 0
_HEAD_DIM = 1


def _ulysses_cp_mla_forward(
    inner_forward: Callable[..., torch.Tensor],
    inner_attention: MLAInnerAttention,
    q_THK: torch.Tensor,
    kv_THP: torch.Tensor,
    k_shared_TR: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    cp_group = spmd_mesh_group(MeshAxisName.CP)
    if cp_group is None:
        raise RuntimeError("CP attention requires an active multi-rank CP mesh axis.")

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
        packed_q_kv_THF, [qk_head_dim, kv_THP.shape[-1]], dim=-1
    )
    k_shared_TR = spmd.redistribute(
        k_shared_TR,
        cp_group,
        src=spmd.S(_TOKEN_DIM),
        dst=spmd.R,
        backward_options={"op_dtype": k_shared_TR.dtype},
    )

    out_THV = inner_forward(inner_attention, q_THK, kv_THP, k_shared_TR, **kwargs)
    return spmd.redistribute(
        out_THV,
        cp_group,
        src=spmd.S(_HEAD_DIM),
        dst=spmd.S(_TOKEN_DIM),
    )


class KVAllGatherCPMLAFlexInnerAttention(
    CPInnerAttention[BlockMask, BlockMask],
    MLAFlexInnerAttention,
):
    """All-gather compact MLA K/V before materializing the shared key."""

    @dataclass(kw_only=True, slots=True)
    class Config(KVAllGatherCPFlexInnerAttention.Config, MLAFlexInnerAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]

    @classmethod
    def prepare_cp_batch_metadata(
        cls,
        input_dict: dict[str, Any],
        *,
        permutation: torch.Tensor | None,
        config: CPInnerAttention.Config,
    ) -> dict[str, Any]:
        return KVAllGatherCPFlexInnerAttention.prepare_cp_batch_metadata(
            input_dict,
            permutation=permutation,
            config=config,
        )

    @staticmethod
    def prepare_cp_metadata(
        context_metadata: BlockMask,
        *,
        permutation: torch.Tensor | None,
        config: CPInnerAttention.Config,
    ) -> BlockMask:
        return KVAllGatherCPFlexInnerAttention.prepare_cp_metadata(
            context_metadata,
            permutation=permutation,
            config=config,
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        kv_THP: torch.Tensor,
        k_shared_TR: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        kv_THP, k_shared_TR = all_gather_cp_tensors(
            (kv_THP, k_shared_TR), reduce_dtype=self.reduce_dtype
        )
        return super().forward(q_THK, kv_THP, k_shared_TR, **kwargs)


class UlyssesCPMLAFlexInnerAttention(
    UlyssesCPInnerAttention[FlexAttentionMetadata], MLAFlexInnerAttention
):
    """MLA Flex Attention with headless shared-key communication."""

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesCPInnerAttention.Config, MLAFlexInnerAttention.Config):
        pass

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        q_THK: torch.Tensor,
        kv_THP: torch.Tensor,
        k_shared_TR: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return _ulysses_cp_mla_forward(
            MLAFlexInnerAttention.forward,
            self,
            q_THK,
            kv_THP,
            k_shared_TR,
            **kwargs,
        )


class UlyssesCPMLAVarlenInnerAttention(
    UlyssesCPInnerAttention[VarlenAttentionMetadata], MLAVarlenInnerAttention
):
    """MLA variable-length attention with headless shared-key communication."""

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesCPInnerAttention.Config, MLAVarlenInnerAttention.Config):
        pass

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        q_THK: torch.Tensor,
        kv_THP: torch.Tensor,
        k_shared_TR: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return _ulysses_cp_mla_forward(
            MLAVarlenInnerAttention.forward,
            self,
            q_THK,
            kv_THP,
            k_shared_TR,
            **kwargs,
        )
