# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel attention kernels.

Tensor suffixes: ``T`` tokens, ``H`` heads, ``K`` qk head dim, ``V`` v head dim.
"""

from abc import ABC, abstractmethod
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Generic, Literal, TYPE_CHECKING, TypeVar

import spmd_types as spmd

import torch
import torch.distributed as dist
import torch_remat as remat
from torch.export import register_dataclass

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.context_parallel import (
    cp_all_gather_tensors,
    cp_query_ranges,
    get_copy_engine_block_exchange,
)
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

if TYPE_CHECKING:
    from attn_gym.distributed.selective_gather import CopyEngineBlockExchange

__all__ = [
    "CPInnerAttention",
    "AllGatherKVFlex",
    "AllGatherKVVarlen",
    "AllGatherKVFlexInnerAttention",
    "AllGatherKVSlidingWindowFlexInnerAttention",
    "AllGatherKVSlidingWindowVarlenInnerAttention",
    "AllGatherKVVarlenInnerAttention",
    "SelectiveGatherKVSlidingWindowVarlenInnerAttention",
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
_VarlenMetadataT = TypeVar("_VarlenMetadataT", bound=VarlenAttentionMetadata)


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
            # Preserve the dQ write-order metadata of the deterministic FLASH backward.
            compute_dq_write_order=block_mask.dq_write_order is not None,
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


@dataclass(frozen=True, slots=True, kw_only=True)
class AllGatherKVVarlenAttentionMetadata(VarlenAttentionMetadata):
    """Global document offsets and optional K/V reordering indices."""

    inverse_permutation: torch.Tensor | None


register_dataclass(AllGatherKVVarlenAttentionMetadata)


def _validate_varlen_cp_metadata(
    metadata: VarlenAttentionMetadata,
    *,
    permutation: torch.Tensor | None,
) -> None:
    if not isinstance(metadata, VarlenAttentionMetadata):
        raise ValueError(
            "K/V varlen CP requires VarlenAttentionMetadata, "
            f"but got {type(metadata).__name__}."
        )
    left, right = metadata.window_size
    if right != 0 or left < -1:
        raise ValueError("K/V varlen CP requires a causal window.")
    if metadata.cu_seq_k is not metadata.cu_seq_q:
        raise ValueError("K/V varlen CP supports only self-attention.")
    if permutation is not None and (permutation.ndim != 2 or permutation.shape[0] != 1):
        raise ValueError(
            "Head-tail varlen CP requires a permutation with shape (1, seq_len)."
        )


def _chunk_varlen_metadata(
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


def _run_varlen_cp_attention(
    attention: VarlenInnerAttention[_VarlenMetadataT],
    q_THK: torch.Tensor,
    *,
    attention_metadata: VarlenAttentionMetadata,
    query_runs: tuple[tuple[int, int], ...],
    select_kv: Callable[[int, int], tuple[torch.Tensor, torch.Tensor]],
    **kwargs,
) -> torch.Tensor:
    """Run one varlen call per local Q range.

    Each Q range uses a K/V interval that includes its causal lookback.

    Args:
        attention: Varlen attention module used for each Q chunk.
        q_THK: Rank-local Q with shape ``[T, H, K]``.
        attention_metadata: Global document offsets, length bounds, and window.
        query_runs: Global ``[q_start, end)`` ranges for equal-sized chunks of
            ``q_THK``, in local Q order.
        select_kv: Callback ``(kv_start, end) -> (k, v)`` for each Q range.
            It reads ``[kv_start, end)`` from full gathered K/V or a compact
            selected buffer, in global token order. The start includes the
            needed lookback; the end matches the Q range.
        **kwargs: Arguments forwarded to the varlen kernel wrapper.

    Returns:
        Attention outputs joined in local Q order.
    """
    left = attention_metadata.window_size[0]
    # A chunk needs at most one document length of earlier keys.
    max_document_length = attention_metadata.max_k
    max_lookback = max_document_length if left == -1 else min(left, max_document_length)
    outputs = []
    for q_chunk_THK, (q_start, end) in zip(
        q_THK.chunk(len(query_runs)), query_runs, strict=True
    ):
        kv_start = max(0, q_start - max_lookback)
        metadata = _chunk_varlen_metadata(
            attention_metadata,
            q_start=q_start,
            end=end,
            kv_start=kv_start,
        )
        k_chunk_THK, v_chunk_THV = select_kv(kv_start, end)
        outputs.append(
            VarlenInnerAttention._forward_varlen(
                attention,
                q_chunk_THK,
                k_chunk_THK,
                v_chunk_THV,
                attention_metadata=metadata,
                **kwargs,
            )
        )
    out_THV = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    if kwargs.get("out_transform") is None and spmd.is_type_checking():
        spmd.assert_type(
            out_THV,
            spmd.get_local_type(q_THK),
            spmd.get_partition_spec(q_THK),
        )
    return out_THV


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
        _validate_varlen_cp_metadata(attention_metadata, permutation=permutation)
        inverse_permutation = None
        if permutation is not None:
            seq_len = permutation.shape[1]
            inverse_permutation = torch.empty_like(permutation[0])
            inverse_permutation[permutation[0]] = torch.arange(
                seq_len,
                device=permutation.device,
                dtype=permutation.dtype,
            )
        return AllGatherKVVarlenAttentionMetadata(
            **{
                field.name: getattr(attention_metadata, field.name)
                for field in fields(VarlenAttentionMetadata)
            },
            inverse_permutation=inverse_permutation,
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: AllGatherKVVarlenAttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        if not isinstance(attention_metadata, AllGatherKVVarlenAttentionMetadata):
            raise ValueError(
                f"{type(self).__name__} requires AllGatherKVVarlenAttentionMetadata, "
                f"but got {type(attention_metadata).__name__}."
            )
        assert isinstance(self, VarlenInnerAttention)
        k_THK, v_THV = cp_all_gather_tensors(
            (k_THK, v_THV),
            reduce_dtype=self.reduce_dtype,
        )
        inverse_permutation = attention_metadata.inverse_permutation
        if inverse_permutation is not None:
            gathered_k_THK, gathered_v_THV = k_THK, v_THV
            with spmd.no_typecheck():
                k_THK = gathered_k_THK.index_select(0, inverse_permutation)
                v_THV = gathered_v_THV.index_select(0, inverse_permutation)
            if spmd.is_type_checking():
                spmd.assert_type_like(k_THK, gathered_k_THK)
                spmd.assert_type_like(v_THV, gathered_v_THV)
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        assert cp_group is not None
        rank = dist.get_rank(cp_group)
        query_runs = cp_query_ranges(
            q_THK.shape[0] * cp_group.size(),
            cp_group.size(),
            rank,
            head_tail=inverse_permutation is not None,
        )

        def select_kv(start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
            return k_THK[start:end], v_THV[start:end]

        return _run_varlen_cp_attention(
            self,
            q_THK,
            attention_metadata=attention_metadata,
            query_runs=query_runs,
            select_kv=select_kv,
            **kwargs,
        )


class AllGatherKVVarlenInnerAttention(
    AllGatherKVVarlen,
    VarlenInnerAttention[AllGatherKVVarlenAttentionMetadata],
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
    SlidingWindowVarlenInnerAttention[AllGatherKVVarlenAttentionMetadata],
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


@dataclass(frozen=True, slots=True, kw_only=True)
class _SelectiveGatherVarlenMetadata(VarlenAttentionMetadata):
    head_tail: bool


register_dataclass(_SelectiveGatherVarlenMetadata)


class SelectiveGatherKVSlidingWindowVarlenInnerAttention(
    CPInnerAttention[VarlenAttentionMetadata, _SelectiveGatherVarlenMetadata],
    SlidingWindowVarlenInnerAttention[_SelectiveGatherVarlenMetadata],
):
    """Sliding-window varlen attention with a fixed selective-gather halo."""

    @dataclass(kw_only=True, slots=True)
    class Config(
        CPInnerAttention.Config,
        SlidingWindowVarlenInnerAttention.Config,
    ):
        gather_block_size: int = 256
        """Tokens per transport block. The tested default rounds a 4096-token
        window's 4095-token lookback to a 4096-token halo.
        """

        def __post_init__(self) -> None:
            SlidingWindowVarlenInnerAttention.Config.__post_init__(self)
            if self.gather_block_size <= 0:
                raise ValueError("gather_block_size must be positive.")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.sliding_window_size = config.sliding_window_size
        self.gather_block_size = config.gather_block_size
        self._layout: tuple[tuple[int, ...], ...] | None = None
        self._exchange: "CopyEngineBlockExchange | None" = None
        self._exchange_key = None

    def _get_exchange(
        self,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        cp_group: dist.ProcessGroup,
        head_tail: bool,
    ) -> "CopyEngineBlockExchange":
        key = (
            cp_group,
            head_tail,
            k_THK.shape,
            v_THV.shape,
            k_THK.dtype,
            v_THV.dtype,
            k_THK.device,
            v_THV.device,
        )
        exchange = self._exchange
        if exchange is None:
            if k_THK.is_cuda and torch.cuda.is_current_stream_capturing():
                raise ValueError("Warm up selective gather before CUDA Graph capture.")
            from attn_gym.distributed.selective_gather import build_sliding_window_plan

            cp_size = cp_group.size()
            global_tokens = k_THK.shape[0] * cp_size
            local_ranges = [
                cp_query_ranges(global_tokens, cp_size, rank, head_tail=head_tail)
                for rank in range(cp_size)
            ]
            plan, layout = build_sliding_window_plan(
                local_ranges,
                rank=cp_group.rank(),
                block_tokens=self.gather_block_size,
                sliding_window_size=self.sliding_window_size,
            )
            exchange = get_copy_engine_block_exchange(
                cp_group,
                plan,
                k_token_shape=k_THK.shape[1:],
                v_token_shape=v_THV.shape[1:],
                dtype=k_THK.dtype,
                device=k_THK.device,
            )
            self._exchange = exchange
            self._layout = layout
            self._exchange_key = key
        elif self._exchange_key != key:
            raise ValueError("Selective gather requires fixed K/V layout and CP group.")
        return exchange

    def _exchange_kv(
        self,
        exchange: "CopyEngineBlockExchange",
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Unconfigured attention uses fused exchange without remat.
        recompute = bool(self._remat_module_fqn) and self.remat_should_recompute(
            "assemble_kv"
        )
        with spmd.no_typecheck():
            if recompute:
                # Keep the halo; backward repeats only local assembly.
                remote_k_RHK, remote_v_RHV = remat.region(
                    exchange.exchange,
                    self.remat_region_name("remote_kv"),
                    recompute=False,
                )(k_THK, v_THV)
                compact_k_THK, compact_v_THV = remat.region(
                    exchange.assemble,
                    self.remat_region_name("assemble_kv"),
                    recompute=True,
                )(k_THK, v_THV, remote_k_RHK, remote_v_RHV)
            else:
                compact_k_THK, compact_v_THV = remat.region(
                    exchange.exchange_and_assemble,
                    self.remat_region_name("assemble_kv"),
                    recompute=False,
                )(k_THK, v_THV)
        spmd.assert_local_type_like(compact_k_THK, k_THK)
        spmd.assert_local_type_like(compact_v_THV, v_THV)
        return compact_k_THK, compact_v_THV

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: VarlenAttentionMetadata,
        *,
        permutation: torch.Tensor | None,
    ) -> _SelectiveGatherVarlenMetadata:
        _validate_varlen_cp_metadata(attention_metadata, permutation=permutation)
        if attention_metadata.window_size[0] < 0:
            raise ValueError("Selective gather requires a sliding window.")
        return _SelectiveGatherVarlenMetadata(
            cu_seq_q=attention_metadata.cu_seq_q,
            cu_seq_k=attention_metadata.cu_seq_k,
            max_q=attention_metadata.max_q,
            max_k=attention_metadata.max_k,
            window_size=attention_metadata.window_size,
            head_tail=permutation is not None,
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: _SelectiveGatherVarlenMetadata,
        **kwargs,
    ) -> torch.Tensor:
        if not isinstance(attention_metadata, _SelectiveGatherVarlenMetadata):
            raise ValueError("Selective gather requires prepared CP metadata.")
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError("Selective gather requires a multi-rank CP axis.")
        exchange = self._get_exchange(
            k_THK, v_THV, cp_group=cp_group, head_tail=attention_metadata.head_tail
        )
        compact_k_THK, compact_v_THV = self._exchange_kv(exchange, k_THK, v_THV)
        assert self._layout is not None
        rank = cp_group.rank()
        global_blocks = self._layout[rank]
        remat.recompute_needs_tensor(compact_k_THK, compact_v_THV)

        def select_kv(kv_start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
            # Each chunk's K/V interval is contiguous in the compact buffer.
            offset = (
                bisect_left(global_blocks, kv_start // self.gather_block_size)
                * self.gather_block_size
                + kv_start % self.gather_block_size
            )
            stop = offset + end - kv_start
            return compact_k_THK[offset:stop], compact_v_THV[offset:stop]

        return _run_varlen_cp_attention(
            self,
            q_THK,
            attention_metadata=attention_metadata,
            query_runs=cp_query_ranges(
                k_THK.shape[0] * cp_group.size(),
                cp_group.size(),
                rank,
                head_tail=attention_metadata.head_tail,
            ),
            select_kv=select_kv,
            **kwargs,
        )


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
