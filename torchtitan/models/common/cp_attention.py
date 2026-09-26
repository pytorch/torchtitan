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
from typing import Any, Generic, Literal, TypeVar

import spmd_types as spmd

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask
from torch.utils import _pytree as pytree

from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group

from torchtitan.models.common.attention import (
    create_attention_mask,
    FlexAttentionMetadata,
    FlexInnerAttention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)

__all__ = [
    "CPInnerAttention",
    "KVAllGatherCPFlexInnerAttention",
    "UlyssesCPInnerAttention",
    "UlyssesCPFlexInnerAttention",
    "UlyssesCPVarlenInnerAttention",
]

_TOKEN_DIM = 0
_HEAD_DIM = 1

_GlobalContextMetadataT = TypeVar("_GlobalContextMetadataT")
_LocalContextMetadataT = TypeVar("_LocalContextMetadataT")


class CPInnerAttention(ABC, Generic[_GlobalContextMetadataT, _LocalContextMetadataT]):
    """Inner attention that owns CP execution and metadata preparation.

    Subclasses implement the CP attention algorithm and prepare its context
    metadata for rank-local execution.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @classmethod
    @abstractmethod
    def prepare_cp_batch_metadata(
        cls,
        input_dict: dict[str, Any],
        *,
        permutation: torch.Tensor | None,
    ) -> dict[str, Any]:
        """Find and prepare the metadata owned by this backend."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def prepare_cp_metadata(
        context_metadata: _GlobalContextMetadataT,
        *,
        permutation: torch.Tensor | None,
    ) -> _LocalContextMetadataT:
        """Prepare local metadata; ``None`` means contiguous CP sharding."""
        raise NotImplementedError


class KVAllGatherCPFlexInnerAttention(
    CPInnerAttention[BlockMask, BlockMask],
    FlexInnerAttention,
):
    """FlexInnerAttention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, FlexInnerAttention.Config):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]

    @classmethod
    def prepare_cp_batch_metadata(
        cls,
        input_dict: dict[str, Any],
        *,
        permutation: torch.Tensor | None,
    ) -> dict[str, Any]:
        """Prepare every FlexAttention BlockMask in the model inputs."""
        if "attention_masks" not in input_dict:
            return input_dict

        context_metadata = input_dict["attention_masks"]
        flat_metadata, spec = pytree.tree_flatten(
            context_metadata,
            is_leaf=lambda value: isinstance(value, BlockMask),
        )
        if not any(isinstance(value, BlockMask) for value in flat_metadata):
            raise ValueError(
                "K/V all-gather context parallelism requires BlockMask metadata."
            )

        flat_local_metadata = [
            (
                cls.prepare_cp_metadata(value, permutation=permutation)
                if isinstance(value, BlockMask)
                else value
            )
            for value in flat_metadata
        ]
        input_dict["attention_masks"] = pytree.tree_unflatten(flat_local_metadata, spec)
        return input_dict

    @staticmethod
    def prepare_cp_metadata(
        context_metadata: BlockMask,
        *,
        permutation: torch.Tensor | None,
    ) -> BlockMask:
        """Shard one global BlockMask to match the model-input partition."""
        if not isinstance(context_metadata, BlockMask):
            raise ValueError(
                "K/V all-gather context parallelism requires BlockMask metadata, "
                f"but got {type(context_metadata).__name__}."
            )
        return KVAllGatherCPFlexInnerAttention._shard_block_mask(
            context_metadata,
            permutation=permutation,
        )

    @staticmethod
    def _shard_block_mask(
        block_mask: BlockMask,
        *,
        permutation: torch.Tensor | None,
    ) -> BlockMask:
        """Build a rank-local BlockMask for K/V all-gather CP.

        The returned mask covers the current rank's Q shard and the global K/V
        sequence. Its mask function maps local Q and global K/V positions from
        the permuted sequence back to their original global indices before
        applying the input mask function. The input block size and full-block
        representation are preserved.

        Args:
            block_mask: BlockMask for the unsharded global sequence.
            permutation: Global token permutation applied before CP sharding,
                with shape ``[1, seq_len]`` or ``[batch, seq_len]``. ``None``
                selects contiguous sharding without token reordering.

        Returns:
            A local-Q/global-KV BlockMask for the current CP rank.

        Raises:
            NotImplementedError: If the global Q length cannot be evenly
                divided into complete Q blocks across CP ranks.
            ValueError: If ``permutation`` has an invalid shape or sequence
                length.
        """
        global_q_len, global_kv_len = block_mask.seq_lengths
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "CP metadata preparation requires an active multi-rank CP mesh axis."
            )
        cp_size = cp_group.size()
        cp_rank = dist.get_rank(cp_group)
        q_block_size, _ = block_mask.BLOCK_SIZE
        if global_q_len % (cp_size * q_block_size) != 0:
            raise NotImplementedError(
                f"Global Q length ({global_q_len}) must be divisible by CP size "
                f"({cp_size}) * Q block size ({q_block_size})."
            )

        if permutation is not None:
            if permutation.ndim != 2:
                raise ValueError(
                    "CP permutation must have shape [1, seq_len] or "
                    f"[batch, seq_len], but got {tuple(permutation.shape)}."
                )
            if permutation.shape[1] != global_q_len:
                raise ValueError(
                    f"CP permutation length ({permutation.shape[1]}) must match "
                    f"global Q length ({global_q_len})."
                )

        local_q_len = global_q_len // cp_size
        mask_mod = block_mask.mask_mod

        def _unpermute_idx(
            batch_idx: torch.Tensor, post_permutation_idx: torch.Tensor
        ) -> torch.Tensor:
            if permutation is None:
                return post_permutation_idx
            if permutation.shape[0] == 1:
                return permutation[0][post_permutation_idx]
            return permutation[batch_idx][post_permutation_idx]

        def _local_q_idx_to_global_q_idx(local_q_idx: torch.Tensor) -> torch.Tensor:
            local_block_idx, local_block_offset = (
                local_q_idx // q_block_size,
                local_q_idx % q_block_size,
            )
            num_local_blocks = local_q_len // q_block_size
            global_block_idx = num_local_blocks * cp_rank + local_block_idx
            return global_block_idx * q_block_size + local_block_offset

        def _local_mask_mod(
            batch_idx: torch.Tensor,
            head_idx: torch.Tensor,
            local_q_idx: torch.Tensor,
            global_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            # The original mask_mod expects indices in the pre-permutation order.
            return mask_mod(
                batch_idx,
                head_idx,
                _unpermute_idx(
                    batch_idx,
                    _local_q_idx_to_global_q_idx(local_q_idx),
                ),
                _unpermute_idx(batch_idx, global_kv_idx),
            )

        return create_attention_mask(
            _local_mask_mod,
            block_mask.kv_num_blocks.shape[0],
            block_mask.kv_num_blocks.shape[1],
            local_q_len,
            global_kv_len,
            device=block_mask.kv_num_blocks.device,
            BLOCK_SIZE=block_mask.BLOCK_SIZE,
            # Preserve whether the global mask stores full blocks separately.
            separate_full_blocks=block_mask.full_kv_num_blocks is not None,
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


class UlyssesCPInnerAttention(
    CPInnerAttention[_GlobalContextMetadataT, _GlobalContextMetadataT]
):
    """Move CP sharding from tokens to heads while keeping metadata global."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config):
        pass

    @classmethod
    def prepare_cp_batch_metadata(
        cls,
        input_dict: dict[str, Any],
        *,
        permutation: torch.Tensor | None,
    ) -> dict[str, Any]:
        del cls, input_dict, permutation
        raise RuntimeError(
            "Ulysses CP keeps context metadata global; "
            "prepare_cp_batch_metadata must not be called."
        )

    @staticmethod
    def prepare_cp_metadata(
        context_metadata: _GlobalContextMetadataT,
        *,
        permutation: torch.Tensor | None,
    ) -> _GlobalContextMetadataT:
        del context_metadata, permutation
        raise RuntimeError(
            "Ulysses CP keeps context metadata global; "
            "prepare_cp_metadata must not be called."
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


class UlyssesCPFlexInnerAttention(
    UlyssesCPInnerAttention[FlexAttentionMetadata], FlexInnerAttention
):
    """FlexInnerAttention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesCPInnerAttention.Config, FlexInnerAttention.Config):
        pass


class UlyssesCPVarlenInnerAttention(
    UlyssesCPInnerAttention[VarlenAttentionMetadata], VarlenInnerAttention
):
    """VarlenInnerAttention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesCPInnerAttention.Config, VarlenInnerAttention.Config):
        pass
