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
from typing import Generic, Literal, NamedTuple, TypeVar

import spmd_types as spmd

import torch
import torch.distributed as dist

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.context_parallel import cp_all_gather_tensors
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group

from .attention import (
    create_attention_mask,
    FlexAttentionMetadata,
    FlexInnerAttention,
    InnerAttention,
    SlidingWindowFlexInnerAttention,
    SlidingWindowVarlenInnerAttention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)

__all__ = [
    "CPInnerAttention",
    "AllGatherKVFlex",
    "AllGatherKVVarlen",
    "AllGatherKVFlexInnerAttention",
    "AllGatherKVSlidingWindowFlexInnerAttention",
    "AllGatherKVSlidingWindowVarlenInnerAttention",
    "AllGatherKVVarlenInnerAttention",
    "Ulysses",
    "UlyssesInnerAttention",
    "UlyssesFlexInnerAttention",
    "UlyssesSlidingWindowFlexInnerAttention",
    "UlyssesSlidingWindowVarlenInnerAttention",
    "UlyssesVarlenInnerAttention",
]

_TOKEN_DIM = 0
_HEAD_DIM = 1

_GlobalAttentionMetadataT = TypeVar("_GlobalAttentionMetadataT")
_LocalAttentionMetadataT = TypeVar("_LocalAttentionMetadataT")


class CPInnerAttention(
    InnerAttention,
    ABC,
    Generic[_GlobalAttentionMetadataT, _LocalAttentionMetadataT],
):
    """Inner attention that owns CP execution and metadata preparation.

    Subclasses implement the CP attention algorithm and prepare its context
    metadata for rank-local execution.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        pass

    @staticmethod
    @abstractmethod
    def prepare_cp_metadata(
        attention_metadata: _GlobalAttentionMetadataT,
        *,
        permutation: torch.Tensor | None,
    ) -> _LocalAttentionMetadataT:
        """Prepare this backend's metadata for rank-local CP execution."""
        raise NotImplementedError


class AllGatherKVFlex(CPInnerAttention[FlexAttentionMetadata, FlexAttentionMetadata]):
    """Base for K/V all-gather CP attention using FlexAttention masks."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: FlexAttentionMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> FlexAttentionMetadata:
        """Prepare a rank-local BlockMask for K/V all-gather CP.

        The returned mask covers the current rank's Q shard and the global K/V
        sequence. Its mask function maps local Q and global K/V positions from
        the permuted sequence back to their original global indices before
        applying the input mask function. The input block size and full-block
        representation are preserved.

        Args:
            attention_metadata: BlockMask for the unsharded global sequence.
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
        block_mask = attention_metadata
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


class AllGatherKVFlexInnerAttention(
    AllGatherKVFlex,
    FlexInnerAttention,
):
    """FlexInnerAttention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(
        AllGatherKVFlex.Config,
        FlexInnerAttention.Config,
    ):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

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
        k_THK, v_THV = cp_all_gather_tensors(
            (k_THK, v_THV), reduce_dtype=self.reduce_dtype
        )
        return super().forward(q_THK, k_THK, v_THV, **kwargs)


class AllGatherKVSlidingWindowFlexInnerAttention(
    AllGatherKVFlex, SlidingWindowFlexInnerAttention
):
    """SlidingWindowFlexInnerAttention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(
        AllGatherKVFlex.Config,
        SlidingWindowFlexInnerAttention.Config,
    ):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

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
        k_THK, v_THV = cp_all_gather_tensors(
            (k_THK, v_THV), reduce_dtype=self.reduce_dtype
        )
        return super().forward(q_THK, k_THK, v_THV, **kwargs)


class AllGatherKVVarlenAttentionMetadata(NamedTuple):
    """Global document offsets and optional K/V reordering indices."""

    varlen_metadata: VarlenAttentionMetadata
    inverse_permutation: torch.Tensor | None


class AllGatherKVVarlen(
    CPInnerAttention[VarlenAttentionMetadata, AllGatherKVVarlenAttentionMetadata],
):
    """Share K/V all-gather CP logic across the varlen variants."""

    reduce_dtype: torch.dtype

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: VarlenAttentionMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> AllGatherKVVarlenAttentionMetadata:
        """Prepare global document offsets and K/V reordering indices."""
        if not isinstance(attention_metadata, VarlenAttentionMetadata):
            raise ValueError(
                "K/V all-gather varlen CP requires VarlenAttentionMetadata, "
                f"but got {type(attention_metadata).__name__}."
            )
        left, right = attention_metadata.window_size
        if right != 0 or left < -1:
            raise ValueError("K/V all-gather varlen CP requires a causal window.")
        if attention_metadata.cu_seq_k is not attention_metadata.cu_seq_q:
            raise ValueError("Varlen K/V all-gather CP supports only self-attention.")
        if permutation is None:
            return AllGatherKVVarlenAttentionMetadata(
                varlen_metadata=attention_metadata, inverse_permutation=None
            )
        # A permutation must come from global head-tail load balancing.
        if permutation.ndim != 2 or permutation.shape[0] != 1:
            raise ValueError(
                "Head-tail varlen CP requires a permutation with shape (1, seq_len)."
            )

        seq_len = permutation.shape[1]
        inverse_permutation = torch.empty_like(permutation[0])
        inverse_permutation[permutation[0]] = torch.arange(
            seq_len,
            device=permutation.device,
            dtype=permutation.dtype,
        )
        return AllGatherKVVarlenAttentionMetadata(
            varlen_metadata=attention_metadata,
            inverse_permutation=inverse_permutation,
        )

    @staticmethod
    def _chunk_metadata(
        varlen_metadata: VarlenAttentionMetadata,
        *,
        q_start: int,
        end: int,
        kv_start: int,
    ) -> VarlenAttentionMetadata:
        """Describe one fixed query chunk and its contiguous K/V range."""
        global_cu_seq = varlen_metadata.cu_seq_q
        if spmd.is_type_checking():
            # Document offsets vary across DP but are shared across CP and TP.
            spmd.assert_type(
                global_cu_seq,
                spmd.R,
                {MeshAxisName.DP: spmd.V},
                partition_spec=spmd.PartitionSpec(MeshAxisName.DP),
            )
        cu_seq_q = global_cu_seq.clamp(min=q_start, max=end) - q_start
        cu_seq_k = global_cu_seq.clamp(min=kv_start, max=end) - kv_start
        if spmd.is_type_checking():
            # cu_seq_q and cu_seq_k have different values across CP ranks.
            for cu_seq in (cu_seq_q, cu_seq_k):
                spmd.mutate_type(cu_seq, MeshAxisName.CP, src=spmd.R, dst=spmd.V)
        chunk_len = end - q_start
        return VarlenAttentionMetadata(
            cu_seq_q=cu_seq_q,
            cu_seq_k=cu_seq_k,
            max_q=min(varlen_metadata.max_q, chunk_len),
            max_k=min(varlen_metadata.max_k, end - kv_start),
            window_size=varlen_metadata.window_size,
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: (
            VarlenAttentionMetadata | AllGatherKVVarlenAttentionMetadata
        ),
        **kwargs,
    ) -> torch.Tensor:
        if not isinstance(attention_metadata, AllGatherKVVarlenAttentionMetadata):
            raise ValueError(
                f"{type(self).__name__} requires AllGatherKVVarlenAttentionMetadata, "
                f"but got {type(attention_metadata).__name__}."
            )
        varlen_metadata = attention_metadata.varlen_metadata
        k_THK, v_THV = cp_all_gather_tensors(
            (k_THK, v_THV),
            reduce_dtype=self.reduce_dtype,
        )

        left = varlen_metadata.window_size[0]
        # A chunk needs at most one document length of earlier keys.
        max_document_length = varlen_metadata.max_k
        max_lookback = (
            max_document_length if left == -1 else min(left, max_document_length)
        )
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        assert cp_group is not None
        cp_rank = dist.get_rank(cp_group)

        varlen_forward = super().forward

        def run_chunk(q_chunk_THK: torch.Tensor, q_start: int) -> torch.Tensor:
            end = q_start + q_chunk_THK.shape[0]
            kv_start = max(0, q_start - max_lookback)
            metadata = self._chunk_metadata(
                varlen_metadata,
                q_start=q_start,
                end=end,
                kv_start=kv_start,
            )
            return varlen_forward(
                q_chunk_THK,
                k_THK[kv_start:end],
                v_THV[kv_start:end],
                attention_metadata=metadata,
                **kwargs,
            )

        inverse_permutation = attention_metadata.inverse_permutation
        if inverse_permutation is None:
            # No load balancer: Q is one contiguous shard.
            out_THV = run_chunk(q_THK, cp_rank * q_THK.shape[0])
        else:
            # Head-tail load balancer.
            gathered_k_THK, gathered_v_THV = k_THK, v_THV
            with spmd.no_typecheck():
                k_THK = gathered_k_THK.index_select(0, inverse_permutation)
                v_THV = gathered_v_THV.index_select(0, inverse_permutation)
            if spmd.is_type_checking():
                spmd.assert_type_like(k_THK, gathered_k_THK)
                spmd.assert_type_like(v_THV, gathered_v_THV)

            chunk_len = q_THK.shape[0] // 2
            head_start = cp_rank * chunk_len
            global_len = q_THK.shape[0] * cp_group.size()
            tail_start = global_len - (cp_rank + 1) * chunk_len
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


class AllGatherKVVarlenInnerAttention(
    AllGatherKVVarlen,
    VarlenInnerAttention,
):
    """Varlen attention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(
        AllGatherKVVarlen.Config,
        VarlenInnerAttention.Config,
    ):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]


class AllGatherKVSlidingWindowVarlenInnerAttention(
    AllGatherKVVarlen,
    SlidingWindowVarlenInnerAttention,
):
    """Sliding-window varlen attention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(
        AllGatherKVVarlen.Config,
        SlidingWindowVarlenInnerAttention.Config,
    ):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward reduce-scatter."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]


class Ulysses(CPInnerAttention[_GlobalAttentionMetadataT, _GlobalAttentionMetadataT]):
    """Base for Ulysses CP attention with global attention metadata."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: _GlobalAttentionMetadataT,
        *,
        permutation: torch.Tensor | None,
    ) -> _GlobalAttentionMetadataT:
        if permutation is not None:
            raise ValueError(
                "Ulysses CP requires contiguous token shards without a permutation."
            )
        return attention_metadata


class UlyssesInnerAttention(Ulysses[_GlobalAttentionMetadataT]):
    """Move CP sharding from tokens to heads while keeping metadata global."""

    @dataclass(kw_only=True, slots=True)
    class Config(Ulysses.Config):
        pass

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
        out_THV = super().forward(q_THK, k_THK, v_THV, **kwargs)
        # Back to sharded tokens: (T, H/cp, V) -> (T/cp, H, V).
        return spmd.redistribute(
            out_THV,
            cp_group,
            src=spmd.S(_HEAD_DIM),
            dst=spmd.S(_TOKEN_DIM),
        )


class UlyssesFlexInnerAttention(
    UlyssesInnerAttention[FlexAttentionMetadata], FlexInnerAttention
):
    """FlexInnerAttention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesInnerAttention.Config, FlexInnerAttention.Config):
        pass


class UlyssesSlidingWindowFlexInnerAttention(
    UlyssesInnerAttention[FlexAttentionMetadata], SlidingWindowFlexInnerAttention
):
    """SlidingWindowFlexInnerAttention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(
        UlyssesInnerAttention.Config,
        SlidingWindowFlexInnerAttention.Config,
    ):
        pass


class UlyssesVarlenInnerAttention(
    UlyssesInnerAttention[VarlenAttentionMetadata], VarlenInnerAttention
):
    """VarlenInnerAttention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesInnerAttention.Config, VarlenInnerAttention.Config):
        pass


class UlyssesSlidingWindowVarlenInnerAttention(
    UlyssesInnerAttention[VarlenAttentionMetadata],
    SlidingWindowVarlenInnerAttention,
):
    """Sliding-window varlen attention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(
        UlyssesInnerAttention.Config,
        SlidingWindowVarlenInnerAttention.Config,
    ):
        pass
