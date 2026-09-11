# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel attention kernels.

Tensor suffixes: ``T`` tokens, ``H`` heads, ``K`` qk head dim, ``V`` v head dim.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

import spmd_types as spmd

import torch

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group

from torchtitan.models.common.attention import FlexInnerAttention

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

__all__ = [
    "CPInnerAttention",
    "KVAllGatherCPFlexInnerAttention",
    "UlyssesCPFlexInnerAttention",
]

_TOKEN_DIM = 0
_HEAD_DIM = 1


class CPInnerAttention(ABC):
    """Inner attention that owns its context-parallel behavior."""

    @classmethod
    @abstractmethod
    def cp_shard(
        cls,
        input_dict: dict[str, Any],
        input_shardings: dict[str, Any] | None,
        cp_mesh: "DeviceMesh",
        load_balancer_type: str | None,
        ptrr_mask_key: str | None,
    ) -> dict[str, Any]:
        """Shard model inputs for this attention implementation."""


class KVAllGatherCPFlexInnerAttention(CPInnerAttention, FlexInnerAttention):
    """FlexInnerAttention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]

    @classmethod
    def cp_shard(
        cls,
        input_dict: dict[str, Any],
        input_shardings: dict[str, Any] | None,
        cp_mesh: "DeviceMesh",
        load_balancer_type: str | None,
        ptrr_mask_key: str | None,
    ) -> dict[str, Any]:
        from torchtitan.distributed.context_parallel.api import (
            prepare_context_parallel_input,
        )

        return prepare_context_parallel_input(
            input_dict,
            input_shardings,
            cp_mesh,
            load_balancer_type,
            ptrr_mask_key,
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "CP attention requires an active multi-rank CP mesh axis."
            )
        k_THK, v_THV = (
            spmd.redistribute(
                x,
                cp_group,
                src=spmd.S(_TOKEN_DIM),
                dst=spmd.R,
                backward_options={"op_dtype": self.reduce_dtype},
            )
            for x in (k_THK, v_THV)
        )
        return super().forward(q_THK, k_THK, v_THV, **kwargs)


class UlyssesCPFlexInnerAttention(CPInnerAttention, FlexInnerAttention):
    """Run FlexInnerAttention with sequence-to-head all-to-all redistribution."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        pass

    @classmethod
    def cp_shard(
        cls,
        input_dict: dict[str, Any],
        input_shardings: dict[str, Any] | None,
        cp_mesh: "DeviceMesh",
        load_balancer_type: str | None,
        ptrr_mask_key: str | None,
    ) -> dict[str, Any]:
        from torchtitan.distributed.context_parallel.api import (
            prepare_context_parallel_input,
        )

        attention_masks = input_dict.pop("attention_masks", None)
        prepare_context_parallel_input(
            input_dict,
            input_shardings,
            cp_mesh,
            None,
            None,
        )
        if attention_masks is not None:
            input_dict["attention_masks"] = attention_masks
        return input_dict

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "CP attention requires an active multi-rank CP mesh axis."
            )
        # Shard heads instead of tokens: (T/cp, H, *) -> (T, H/cp, *).
        q_THK, k_THK, v_THV = (
            spmd.redistribute(
                x,
                cp_group,
                src=spmd.S(_TOKEN_DIM),
                dst=spmd.S(_HEAD_DIM),
            )
            for x in (q_THK, k_THK, v_THV)
        )
        out_THV = super().forward(q_THK, k_THK, v_THV, **kwargs)
        # Back to sharded tokens: (T, H/cp, V) -> (T/cp, H, V).
        return spmd.redistribute(
            out_THV,
            cp_group,
            src=spmd.S(_HEAD_DIM),
            dst=spmd.S(_TOKEN_DIM),
        )
