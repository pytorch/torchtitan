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
from typing import cast, Generic, Literal, TypeVar

import spmd_types as spmd

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask
from torch.utils import _pytree as pytree

from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group

from torchtitan.models.common.attention import (
    create_attention_mask,
    FlexAttentionMetadata,
    FlexInnerAttention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
    VarlenMetadata,
)

__all__ = [
    "CPInnerAttention",
    "KVAllGatherCPInnerAttention",
    "KVAllGatherCPFlexInnerAttention",
    "KVAllGatherCPVarlenInnerAttention",
    "UlyssesCPInnerAttention",
    "UlyssesCPFlexInnerAttention",
    "UlyssesCPVarlenInnerAttention",
]

_TOKEN_DIM = 0
_HEAD_DIM = 1

_GlobalAttentionMetadataT = TypeVar("_GlobalAttentionMetadataT")
_LocalAttentionMetadataT = TypeVar("_LocalAttentionMetadataT")


class CPInnerAttention(
    ABC, Generic[_GlobalAttentionMetadataT, _LocalAttentionMetadataT]
):
    """Inner attention that owns CP execution and metadata preparation.

    Subclasses implement the CP attention algorithm and prepare its context
    metadata for rank-local execution.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    # TODO(acisseJZhong): Let each attention backend own and prepare only its
    # corresponding metadata instead of receiving the model-level container.
    @staticmethod
    @abstractmethod
    def prepare_cp_metadata(
        attention_metadata: _GlobalAttentionMetadataT,
        *,
        permutation: torch.Tensor | None,
    ) -> _LocalAttentionMetadataT:
        """Prepare local metadata; ``None`` means contiguous CP sharding."""
        raise NotImplementedError


class KVAllGatherCPInnerAttention(
    CPInnerAttention[_GlobalAttentionMetadataT, _LocalAttentionMetadataT]
):
    """Inner attention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config):
        pass

    reduce_dtype: torch.dtype

    def _all_gather_kv(
        self,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return k_THK, v_THV


class KVAllGatherCPFlexInnerAttention(
    KVAllGatherCPInnerAttention[FlexAttentionMetadata, FlexAttentionMetadata],
    FlexInnerAttention,
):
    """FlexInnerAttention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(KVAllGatherCPInnerAttention.Config, FlexInnerAttention.Config):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: FlexAttentionMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> FlexAttentionMetadata:
        """Shard global BlockMask metadata to match the model-input partition."""
        if not isinstance(attention_metadata, (BlockMask, Mapping)):
            raise ValueError(
                "K/V all-gather context parallelism requires BlockMask metadata, "
                f"but got {type(attention_metadata).__name__}."
            )

        flat_metadata, spec = pytree.tree_flatten(
            attention_metadata,
            is_leaf=lambda value: isinstance(value, BlockMask),
        )
        if not any(isinstance(value, BlockMask) for value in flat_metadata):
            raise ValueError(
                "K/V all-gather context parallelism requires BlockMask metadata."
            )

        flat_local_metadata = [
            (
                KVAllGatherCPFlexInnerAttention._shard_block_mask(
                    value,
                    permutation=permutation,
                )
                if isinstance(value, BlockMask)
                else value
            )
            for value in flat_metadata
        ]
        return cast(
            FlexAttentionMetadata,
            pytree.tree_unflatten(flat_local_metadata, spec),
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
        k_THK, v_THV = self._all_gather_kv(k_THK, v_THV)
        return super().forward(q_THK, k_THK, v_THV, **kwargs)


@dataclass(frozen=True, eq=False, kw_only=True)
class HeadTailCPVarlenMetadata:
    """Global varlen metadata and the inverse head-tail permutation."""

    varlen_metadata: VarlenMetadata
    kv_restore_indices: torch.Tensor

    @classmethod
    def from_global(
        cls,
        global_metadata: VarlenMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> "HeadTailCPVarlenMetadata":
        """Prepare metadata for global head-tail load balancing."""
        if global_metadata.cu_seq_k is not global_metadata.cu_seq_q:
            raise ValueError(
                "CP varlen attention currently supports only self-attention."
            )
        if permutation is None or permutation.ndim != 2 or permutation.shape[0] != 1:
            raise ValueError(
                "Head-tail varlen CP requires a permutation with shape (1, seq_len)."
            )

        seq_len = permutation.shape[1]
        restore_indices = torch.empty_like(permutation[0])
        restore_indices[permutation[0]] = torch.arange(
            seq_len,
            device=permutation.device,
            dtype=permutation.dtype,
        )
        return cls(
            varlen_metadata=global_metadata,
            kv_restore_indices=restore_indices,
        )

    def chunk_metadata(
        self,
        *,
        start: int,
        end: int,
        kv_start: int,
    ) -> VarlenMetadata:
        """Describe one fixed query chunk and its contiguous K/V range."""
        global_cu_seq = self.varlen_metadata.cu_seq_q
        cu_seq_q = global_cu_seq.clamp(min=start, max=end) - start
        cu_seq_k = global_cu_seq.clamp(min=kv_start, max=end) - kv_start
        chunk_len = end - start
        return VarlenMetadata(
            cu_seq_q=cu_seq_q,
            cu_seq_k=cu_seq_k,
            max_q=min(self.varlen_metadata.max_q, chunk_len),
            max_k=min(self.varlen_metadata.max_k, end - kv_start),
        )


pytree.register_dataclass(HeadTailCPVarlenMetadata)


class KVAllGatherCPVarlenInnerAttention(
    KVAllGatherCPInnerAttention[VarlenMetadata, HeadTailCPVarlenMetadata],
    VarlenInnerAttention,
):
    """Varlen attention over two global head-tail query chunks."""

    @dataclass(kw_only=True, slots=True)
    class Config(KVAllGatherCPInnerAttention.Config, VarlenInnerAttention.Config):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: VarlenMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> HeadTailCPVarlenMetadata:
        if not isinstance(attention_metadata, VarlenMetadata):
            raise ValueError(
                "K/V all-gather varlen CP requires VarlenMetadata, "
                f"but got {type(attention_metadata).__name__}."
            )
        return HeadTailCPVarlenMetadata.from_global(
            attention_metadata,
            permutation=permutation,
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_masks: VarlenMetadata | HeadTailCPVarlenMetadata,
        **kwargs,
    ) -> torch.Tensor:
        if not isinstance(attention_masks, HeadTailCPVarlenMetadata):
            raise ValueError(
                f"{type(self).__name__} requires HeadTailCPVarlenMetadata, but got "
                f"{type(attention_masks).__name__}."
            )
        cp_metadata = attention_masks
        left = self.window_size[0]

        k_THK, v_THV = self._all_gather_kv(k_THK, v_THV)
        with spmd.no_typecheck():
            restore_indices = cp_metadata.kv_restore_indices
            k_THK = k_THK.index_select(0, restore_indices)
            v_THV = v_THV.index_select(0, restore_indices)

        cp_group = spmd_mesh_group(MeshAxisName.CP)
        assert cp_group is not None
        cp_rank = dist.get_rank(cp_group)
        chunk_len = q_THK.shape[0] // 2
        global_len = q_THK.shape[0] * cp_group.size()
        head_start = cp_rank * chunk_len
        tail_start = global_len - (cp_rank + 1) * chunk_len

        def run_chunk(q_chunk_THK: torch.Tensor, start: int) -> torch.Tensor:
            end = start + chunk_len
            kv_start = 0 if left == -1 else max(0, start - left)
            metadata = cp_metadata.chunk_metadata(
                start=start,
                end=end,
                kv_start=kv_start,
            )
            return super(KVAllGatherCPVarlenInnerAttention, self).forward(
                q_chunk_THK,
                k_THK[kv_start:end],
                v_THV[kv_start:end],
                attention_masks=metadata,
                **kwargs,
            )

        head_out_THV = run_chunk(q_THK[:chunk_len], head_start)
        tail_out_THV = run_chunk(q_THK[chunk_len:], tail_start)
        out_THV = torch.cat((head_out_THV, tail_out_THV))
        if kwargs.get("out_transform") is None and spmd.is_type_checking():
            spmd.assert_type(
                out_THV,
                spmd.get_local_type(q_THK),
                spmd.get_partition_spec(q_THK),
            )
        return out_THV


class UlyssesCPInnerAttention(
    CPInnerAttention[_GlobalAttentionMetadataT, _GlobalAttentionMetadataT]
):
    """Move CP sharding from tokens to heads while keeping metadata global."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: _GlobalAttentionMetadataT,
        *,
        permutation: torch.Tensor | None,
    ) -> _GlobalAttentionMetadataT:
        del permutation
        return attention_metadata

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
