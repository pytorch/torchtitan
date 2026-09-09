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
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.tensor.experimental._context_parallel import flex_cp_allgather

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import require_spmd_mesh_axis_group

from torchtitan.models.common.attention import FlexAttention

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

__all__ = [
    "CPInnerAttention",
    "KVAllGatherCPFlexInnerAttention",
]

_SEQ_DIM = 0


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


class KVAllGatherCPFlexInnerAttention(CPInnerAttention, FlexAttention):
    """FlexAttention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
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
        # TODO(fegin): replace flex_cp_allgather with spmd_types.redistribute.
        cp_group = require_spmd_mesh_axis_group(MeshAxisName.CP)
        pg_name = dist._get_process_group_name(cp_group)
        k_THK, v_THV = flex_cp_allgather(
            k_THK.contiguous(), v_THV.contiguous(), _SEQ_DIM, pg_name
        )
        return super().forward(q_THK, k_THK, v_THV, **kwargs)
