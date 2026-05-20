# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel support for variable-length attention.

Houses :class:`CPVarlenMetadata`, the CP-specific extension of
:class:`~torchtitan.models.common.attention.VarlenMetadata` that adds
``k_local_indices`` and the per-rank metadata builder.

Lives under ``torchtitan/distributed/`` because it is model-agnostic
CP infrastructure; the base :class:`VarlenMetadata` data type stays in
``models/common/attention.py`` so attention modules don't depend on CP
internals.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _LoadBalancer,
    _PTRRLoadBalancer,
)

from torchtitan.models.common.attention import VarlenMetadata


@dataclass(frozen=True, eq=False)
class CPVarlenMetadata(VarlenMetadata):
    """Per-rank :class:`VarlenMetadata` for context-parallel varlen attention.

    A marker subclass produced by :meth:`from_global` whose
    ``k_local_indices`` field is always populated. The base class
    declares the field as optional so :class:`VarlenAttention` can
    consume both CP-built and plain metadata uniformly; this subclass
    exists to host the CP-specific construction logic and to make the
    "this came from CP sharding" intent visible at call sites.

    Because CP shards the sequence dim (not the batch), under DTensor
    Replicate-on-CP the rank-local packed K can have unused gaps between
    per-segment visible regions when ``B > 1``; ``k_local_indices``
    picks out just those regions. When a load balancer is active the
    indices also compose with the per-batch inverse permutation so the
    gather hits the correct entries in the rearranged K/V.
    """

    @classmethod
    def from_global(
        cls,
        global_metadata: VarlenMetadata,
        device_mesh: DeviceMesh,
        batch_size: int,
        seq_length: int,
        load_balancer: _LoadBalancer | None = None,
    ) -> "CPVarlenMetadata":
        """Build per-rank :class:`CPVarlenMetadata` from a global metadata.

        ``batch_size`` and ``seq_length`` describe the unpacked input
        tensor whose packed form is ``global_metadata``; they are used
        to decompose the rank's shard per-batch and must satisfy
        ``batch_size * seq_length == cu_seq_q[-1]``.

        Each rank holds a shard of the global Q; K/V are assumed already
        all-gathered across the CP dim (e.g. via DTensor ``Replicate``).
        For each (doc, rank) contiguous-Q run the builder emits a varlen
        segment with ``seqlen_k`` = (global doc-relative offset at the
        chunk end) + 1, so FA's right-aligned causal reproduces
        document-causal masking exactly.

        Self-attention only (``cu_seq_q == cu_seq_k``); all segment
        construction is vectorized.

        Example (segment layout, ``B = 1``):
            ``seq_len=20, B=1, 3 docs (cu_seq_q=[0,7,13,20]), CP=4``,
            no load balancer. Rank 2 owns Q rows 10..14 which span the
            tail of doc 1 and the head of doc 2, giving two segments::

              cu_seq_q        = [0, 3, 5]         (segment lengths 3, 2)
              cu_seq_k        = [0, 6, 8]         (visible K per segment 6, 2)
              max_q, max_k    = 3, 6
              k_local_indices = [7, 8, 9, 10, 11, 12,  13, 14]

        Example (non-contiguous K under ``B > 1``):
            ``B=2, S=20, CP=4, rank 2``, same docs per batch as above.
            ``K_packed`` (length ``B*S = 40``) places batch 1 after
            batch 0, so a rank's visible K per segment lives inside one
            batch's range; consecutive segments straddle the batch
            boundary. ``k_local_indices`` (length 16) concatenates the
            visible slices [7..13), [13..15), [27..33), [33..35).
        """
        if isinstance(global_metadata, CPVarlenMetadata):
            raise ValueError(
                "from_global received a CPVarlenMetadata; pass the "
                "unsharded global VarlenMetadata instead."
            )
        # Identity check (not torch.equal) to avoid a D2H sync per forward.
        # create_varlen_metadata_for_document uses the same tensor for both
        # fields; callers must do the same.
        if global_metadata.cu_seq_q is not global_metadata.cu_seq_k:
            raise ValueError(
                "CP varlen sharding currently supports only self-attention; "
                "cu_seq_q and cu_seq_k must be the same tensor object."
            )
        cu_seq_q_global = global_metadata.cu_seq_q
        expected_total = batch_size * seq_length
        if cu_seq_q_global.ndim != 1 or cu_seq_q_global.numel() < 2:
            raise ValueError(
                "VarlenMetadata.cu_seq_q must be a 1-D tensor with at least "
                f"2 entries; got shape {tuple(cu_seq_q_global.shape)}."
            )
        if cu_seq_q_global.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "VarlenMetadata.cu_seq_q must be int32 or int64; "
                f"got {cu_seq_q_global.dtype}."
            )

        if device_mesh.ndim != 1:
            raise ValueError(
                f"CPVarlenMetadata.from_global expects a 1-D CP mesh, "
                f"got ndim={device_mesh.ndim}."
            )
        cp_world_size = device_mesh.size()
        cp_rank = device_mesh.get_local_rank()
        required_divisor = (
            2 * cp_world_size if load_balancer is not None else cp_world_size
        )
        if seq_length % required_divisor != 0:
            reason = (
                "2 * cp world size (extra divisibility required when a load "
                "balancer is used)"
                if load_balancer is not None
                else "cp world size"
            )
            raise ValueError(
                f"seq_length {seq_length} must be divisible by {required_divisor} "
                f"({reason}); got cp world size {cp_world_size}."
            )
        shard_len = seq_length // cp_world_size
        device = cu_seq_q_global.device
        dtype = cu_seq_q_global.dtype

        # Per-batch forward / inverse permutation from the load balancer;
        # ``None`` means no rearrangement.
        rearrange_per_batch: torch.Tensor | None = None
        restore_per_batch: torch.Tensor | None = None
        if load_balancer is not None:
            # TODO: migrate to a public API when upstream stabilizes one.
            rearrange_indices = load_balancer._generate_indices(restore=False)
            if rearrange_indices is not None:
                if rearrange_indices.ndim != 2 or rearrange_indices.shape[0] not in (
                    1,
                    batch_size,
                ):
                    raise ValueError(
                        "load balancer indices must have shape (1, seq_length) "
                        "or (batch_size, seq_length); got "
                        f"{tuple(rearrange_indices.shape)}."
                    )
                rearrange_indices = rearrange_indices.to(dtype)
                # (1, S): same permutation across batches (e.g. headtail);
                # argsort once and expand both permutations.
                # (B, S): per-batch permutation (e.g. _VarlenPTRRLoadBalancer);
                # argsort per row.
                if rearrange_indices.shape[0] == 1:
                    rearrange_per_batch = rearrange_indices.expand(batch_size, -1)
                    restore_per_batch = torch.argsort(rearrange_indices, dim=-1).expand(
                        batch_size, -1
                    )
                else:
                    rearrange_per_batch = rearrange_indices
                    restore_per_batch = torch.argsort(rearrange_indices, dim=-1)

        # Per-batch local-to-global seq mapping, (batch_size, shard_len)
        # in [0, seq_length).
        if rearrange_per_batch is None:
            rank_q_indices = (
                torch.arange(
                    cp_rank * shard_len,
                    (cp_rank + 1) * shard_len,
                    device=device,
                    dtype=dtype,
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        else:
            rank_q_indices = rearrange_per_batch[
                :, cp_rank * shard_len : (cp_rank + 1) * shard_len
            ]

        # Per-batch -> packed global positions, row-major across batch
        # (matches the rank's local packed Q layout).
        batch_offsets = (
            torch.arange(batch_size, device=device, dtype=dtype).unsqueeze(1)
            * seq_length
        )
        packed_local_to_global = (batch_offsets + rank_q_indices).reshape(-1)
        total_local = batch_size * shard_len

        doc_id = (
            torch.searchsorted(
                cu_seq_q_global,
                packed_local_to_global,
                right=True,
                out_int32=(dtype == torch.int32),
            )
            - 1
        )

        # Segment break wherever doc id changes or packed-global positions
        # are non-consecutive. Batch boundaries are covered by diff_doc
        # since adjacent batches' docs always have different ids globally.
        diff_doc = doc_id[1:] != doc_id[:-1]
        diff_global = packed_local_to_global[1:] != packed_local_to_global[:-1] + 1
        is_break = diff_doc | diff_global
        # nonzero() always returns int64; cast back so downstream arithmetic
        # stays in cu_seq_q_global.dtype (typically int32).
        seg_starts_inner = (is_break.nonzero(as_tuple=False).squeeze(-1) + 1).to(dtype)
        seg_starts = torch.cat(
            [torch.zeros(1, dtype=dtype, device=device), seg_starts_inner]
        )
        seg_ends = torch.cat(
            [seg_starts[1:], torch.tensor([total_local], dtype=dtype, device=device)]
        )

        seqlen_q = seg_ends - seg_starts
        # seqlen_k = (last global pos in segment) - (doc global start) + 1.
        last_local_idx = seg_ends - 1
        last_global = packed_local_to_global[last_local_idx]
        seg_doc_id = doc_id[seg_starts]
        doc_global_start = cu_seq_q_global[seg_doc_id]
        seqlen_k = last_global - doc_global_start + 1

        cu_seq_q = torch.cat(
            [
                torch.zeros(1, dtype=dtype, device=device),
                seqlen_q.cumsum(0).to(dtype),
            ]
        )
        cu_seq_k = torch.cat(
            [torch.zeros(1, dtype=dtype, device=device), seqlen_k.cumsum(0).to(dtype)]
        )

        # Fuse max_q / max_k / total_k and cu_seq_q endpoint validation into
        # a single D2H transfer to keep one sync point per forward. The
        # endpoint check below raises before any wrong values escape the
        # function, so it does not need to gate the work above.
        max_q, max_k, total_k, first, last = (
            torch.stack(
                [
                    seqlen_q.max(),
                    seqlen_k.max(),
                    seqlen_k.sum(),
                    cu_seq_q_global[0],
                    cu_seq_q_global[-1],
                ]
            )
            .cpu()
            .tolist()
        )
        if first != 0 or last != expected_total:
            raise ValueError(
                "VarlenMetadata.cu_seq_q must start at 0 and end at "
                f"batch_size * seq_length = {expected_total}; got endpoints "
                f"({first}, {last})."
            )

        # Flat gather index (length total_k) over original K coords; per
        # segment covers [doc_global_start, last_global], built via
        # repeat_interleave + arange offset.
        bases = torch.repeat_interleave(doc_global_start, seqlen_k)
        seg_starts_repeated = torch.repeat_interleave(cu_seq_k[:-1], seqlen_k)
        within_seg = (
            torch.arange(total_k, device=device, dtype=dtype) - seg_starts_repeated
        )
        k_local_indices = bases + within_seg

        # K is all-gathered in the balancer's shuffling order, so compose the
        # gather index with the per-batch inverse permutation.
        if restore_per_batch is not None:
            b_ids = k_local_indices // seq_length
            p_orig = k_local_indices % seq_length
            p_rearr = restore_per_batch[b_ids, p_orig]
            k_local_indices = b_ids * seq_length + p_rearr

        return cls(
            cu_seq_q=cu_seq_q,
            cu_seq_k=cu_seq_k,
            max_q=max_q,
            max_k=max_k,
            k_local_indices=k_local_indices.to(torch.long),
        )


class _VarlenPTRRLoadBalancer(_LoadBalancer):
    """Processing-Time based Round-Robin (PTRR) load balancer for
    **document-causal** varlen attention.

    .. warning::
        This balancer assumes per-document causal masking. The per-Q-block
        work estimate is ``t - doc_start[t] + 1`` summed over the block,
        which is only the correct work estimate under causal masking.
        Using this balancer with a non-causal varlen mask will produce a
        valid permutation but a meaningless load balance -- callers are
        responsible for ensuring the attention mask is causal.

    Estimates per-Q-block work directly from ``cu_seq_q`` (no ``BlockMask``
    needed). Per-block work is the sum of per-token work over the block.
    The same scheduling primitive (``_PTRRLoadBalancer.ptrr_scheduling``)
    is reused.

    Outputs a ``(B, S)`` permutation tensor where each batch element is
    rearranged independently -- important when documents differ across
    the batch.
    """

    def __init__(
        self,
        cu_seq_q: Tensor,
        *,
        batch_size: int,
        seq_length: int,
        world_size: int,
        block_size: int = 128,
    ):
        # cu_seq_q is consumed below by ``torch.searchsorted``, which only
        # requires 1-D and monotonic non-decreasing input. Monotonicity and
        # the ``cu_seq_q[-1] == batch_size * seq_length`` endpoint are
        # already enforced by the same tensor's path through
        # :meth:`CPVarlenMetadata.from_global` (which builds the balancer);
        # we skip re-checking here to avoid two extra D2H syncs per forward.
        if cu_seq_q.dim() != 1:
            raise ValueError(
                f"cu_seq_q must be 1-D, got shape {tuple(cu_seq_q.shape)}."
            )
        if seq_length % block_size != 0:
            raise ValueError(
                f"seq_length {seq_length} must be divisible by block_size {block_size}."
            )
        num_blocks = seq_length // block_size
        if num_blocks % world_size != 0:
            raise ValueError(
                f"num_blocks (seq_length / block_size = {num_blocks}) "
                f"must be divisible by world_size {world_size}."
            )

        self.cu_seq_q = cu_seq_q
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.world_size = world_size
        self.block_size = block_size

    def _generate_indices(self, restore: bool = False) -> Tensor:
        """Compute the per-batch forward or inverse permutation.

        Returns a ``(B, S)`` int64 tensor. When ``restore=False`` the
        result is the forward permutation: ``indices[b, i]`` is the
        original token position to be placed at slot ``i`` of batch
        ``b`` after PTRR rebalancing. When ``restore=True`` the result
        is its inverse (``argsort`` of the forward permutation).

        Note the ``(B, S)`` output shape, vs upstream
        :class:`_LoadBalancer`'s ``(1, S)`` or ``(B, S)`` contract:
        PTRR rebalances every batch independently because document
        structure can vary across the batch.
        """
        B = self.batch_size
        S = self.seq_length
        W = self.world_size
        block_size = self.block_size
        num_blocks = S // block_size

        device = self.cu_seq_q.device
        cu = self.cu_seq_q.to(torch.long)

        # Per-token causal work = (token's offset within its doc) + 1.
        positions = torch.arange(B * S, device=device, dtype=torch.long)
        doc_id = torch.searchsorted(cu, positions, right=True) - 1
        work_per_token = positions - cu[doc_id] + 1  # (B*S,)

        # Per-block work: sum of token work within each block, per batch.
        work_per_block = work_per_token.view(B, num_blocks, block_size).sum(dim=-1)
        # (B, num_blocks)

        batch_ptrr = torch.vmap(
            functools.partial(_PTRRLoadBalancer.ptrr_scheduling, group_size=W)
        )
        ptrr_blocks = batch_ptrr(work_per_block)  # (B, W, num_blocks_per_rank)
        ptrr_blocks = ptrr_blocks.reshape(B, -1)  # (B, num_blocks)

        # Expand block indices to token indices.
        block_indices = torch.arange(num_blocks * block_size, device=device).view(
            num_blocks, block_size
        )  # (num_blocks, block_size)
        indices = block_indices[ptrr_blocks].view(B, -1)  # (B, S)

        if restore:
            # pyrefly: ignore[missing-argument]
            indices = torch.vmap(torch.argsort)(indices)

        return indices
