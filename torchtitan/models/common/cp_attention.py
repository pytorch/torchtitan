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
from typing import Any, cast, Literal

import spmd_types as spmd

import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.utils import _pytree as pytree

from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.distributed.context_parallel.api import ContextParallelLoadBalancer
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group

from torchtitan.models.common.attention import FlexInnerAttention, VarlenInnerAttention

__all__ = [
    "CPInnerAttention",
    "KVAllGatherCPFlexInnerAttention",
    "UlyssesCPInnerAttention",
    "UlyssesCPFlexInnerAttention",
    "UlyssesCPVarlenInnerAttention",
]

_TOKEN_DIM = 0
_HEAD_DIM = 1
_BLOCK_MASK_QUERY_DIM = 2


class CPInnerAttention:
    """Inner attention whose config owns CP metadata sharding.

    Model inputs are sharded before the attention backend is selected. Each CP
    backend config then handles only the metadata required by its attention
    layout, using the same per-batch load balancer that sharded the model inputs.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config, ABC):
        @abstractmethod
        def cp_shard_metadata(
            self,
            input_dict: dict[str, Any],
            load_balancer: ContextParallelLoadBalancer,
        ) -> dict[str, Any]:
            """Shard metadata owned by this attention configuration.

            Args:
                input_dict: Model-forward inputs containing attention metadata.
                load_balancer: The current batch's CP partition, shared with
                    model input sharding.

            Returns:
                ``input_dict`` with backend-owned metadata updated as required
                by the attention layout.
            """


class KVAllGatherCPFlexInnerAttention(CPInnerAttention, FlexInnerAttention):
    """FlexInnerAttention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, FlexInnerAttention.Config):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

        def cp_shard_metadata(
            self,
            input_dict: dict[str, Any],
            load_balancer: ContextParallelLoadBalancer,
        ) -> dict[str, Any]:
            """Shard BlockMask metadata for K/V all-gather CP.

            ``attention_masks`` may be a single ``BlockMask`` or a mapping
            containing ``BlockMask`` and other metadata. Every ``BlockMask`` is
            sharded along its query dimension with the same partition as the
            model inputs; the surrounding structure and other values are kept.
            """
            attention_masks = input_dict.get("attention_masks")
            if attention_masks is None:
                return input_dict

            if not isinstance(attention_masks, (BlockMask, Mapping)):
                raise ValueError(
                    "K/V all-gather context parallelism requires BlockMask "
                    f"metadata, but got {type(attention_masks).__name__}."
                )

            # Collect only BlockMask leaves for one sharding collective.
            block_masks = [
                mask
                for mask in pytree.tree_leaves(
                    attention_masks,
                    is_leaf=lambda value: isinstance(value, BlockMask),
                )
                if isinstance(mask, BlockMask)
            ]
            if not block_masks:
                return input_dict

            sharded_masks = cast(
                "tuple[BlockMask, ...]",
                load_balancer.shard(
                    block_masks,
                    (_BLOCK_MASK_QUERY_DIM,) * len(block_masks),
                ),
            )
            sharded_mask_iter = iter(sharded_masks)
            # Replace BlockMask leaves in order and preserve other metadata.
            input_dict["attention_masks"] = pytree.tree_map(
                lambda value: (
                    next(sharded_mask_iter) if isinstance(value, BlockMask) else value
                ),
                attention_masks,
                is_leaf=lambda value: isinstance(value, BlockMask),
            )
            return input_dict

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]

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


class UlyssesCPInnerAttention(CPInnerAttention):
    """Move CP sharding between the token and head dimensions."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config):
        def cp_shard_metadata(
            self,
            input_dict: dict[str, Any],
            load_balancer: ContextParallelLoadBalancer,
        ) -> dict[str, Any]:
            """Keep metadata global for the Ulysses head-sharded layout.

            Ulysses redistributes token-sharded Q/K/V into full-sequence,
            head-sharded tensors in ``forward``, so attention metadata must
            remain global rather than being sharded along the token dimension.
            """
            del load_balancer
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
        # super() follows the concrete class MRO to its inner attention.
        # pyrefly: ignore [missing-attribute]
        out_THV = super().forward(q_THK, k_THK, v_THV, **kwargs)
        # Back to sharded tokens: (T, H/cp, V) -> (T/cp, H, V).
        return spmd.redistribute(
            out_THV,
            cp_group,
            src=spmd.S(_HEAD_DIM),
            dst=spmd.S(_TOKEN_DIM),
        )


class UlyssesCPFlexInnerAttention(UlyssesCPInnerAttention, FlexInnerAttention):
    """FlexInnerAttention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesCPInnerAttention.Config, FlexInnerAttention.Config):
        pass


class UlyssesCPVarlenInnerAttention(UlyssesCPInnerAttention, VarlenInnerAttention):
    """VarlenInnerAttention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesCPInnerAttention.Config, VarlenInnerAttention.Config):
        pass
