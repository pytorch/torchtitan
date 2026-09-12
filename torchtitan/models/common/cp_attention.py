# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel attention kernels.

Tensor suffixes: ``T`` tokens, ``H`` heads, ``K`` qk head dim, ``V`` v head dim.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast, Literal, TYPE_CHECKING

import spmd_types as spmd

import torch
from torch.distributed.tensor.experimental._attention import _context_parallel_shard
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.context_parallel.api import ContextParallelLoadBalancer
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
_BLOCK_MASK_QUERY_DIM = 2


class CPInnerAttention(ABC):
    """Inner attention that owns its context-parallel metadata sharding."""

    @classmethod
    @abstractmethod
    def cp_shard_metadata(
        cls,
        input_dict: dict[str, Any],
        cp_mesh: "DeviceMesh",
        load_balancer: ContextParallelLoadBalancer,
    ) -> dict[str, Any]:
        """Shard metadata owned by this attention implementation."""


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
    def cp_shard_metadata(
        cls,
        input_dict: dict[str, Any],
        cp_mesh: "DeviceMesh",
        load_balancer: ContextParallelLoadBalancer,
    ) -> dict[str, Any]:
        attention_masks = input_dict.get("attention_masks")
        if attention_masks is None:
            return input_dict

        if isinstance(attention_masks, BlockMask):
            block_masks = [attention_masks]
            block_mask_keys = None
        elif isinstance(attention_masks, Mapping):
            block_mask_dict = {
                key: mask
                for key, mask in attention_masks.items()
                if isinstance(mask, BlockMask)
            }
            if not block_mask_dict:
                return input_dict
            block_masks = list(block_mask_dict.values())
            block_mask_keys = block_mask_dict.keys()
        else:
            raise ValueError(
                "K/V all-gather context parallelism requires BlockMask metadata, "
                f"but got {type(attention_masks).__name__}."
            )

        sharded_masks = cast(
            "tuple[BlockMask, ...]",
            _context_parallel_shard(
                mesh=cp_mesh,
                buffers=block_masks,
                seq_dims=(_BLOCK_MASK_QUERY_DIM,) * len(block_masks),
                load_balancer=load_balancer,
            ),
        )
        if block_mask_keys is None:
            input_dict["attention_masks"] = sharded_masks[0]
        else:
            input_dict["attention_masks"] = {
                **attention_masks,
                **dict(zip(block_mask_keys, sharded_masks)),
            }
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
    def cp_shard_metadata(
        cls,
        input_dict: dict[str, Any],
        cp_mesh: "DeviceMesh",
        load_balancer: ContextParallelLoadBalancer,
    ) -> dict[str, Any]:
        """Keep attention metadata global for the Ulysses head-sharded layout.

        Ulysses redistributes token-sharded Q/K/V into full-sequence,
        head-sharded tensors in ``forward``, so its attention metadata must not
        be sharded along the token dimension here.
        """
        del cp_mesh, load_balancer
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
