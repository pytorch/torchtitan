# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Document geometry shared by every layer of a sparse-attention variant."""

from typing import NamedTuple

import spmd_types as spmd
import torch
import torch.nn.functional as F

from torchtitan.models.common.attention import VarlenAttentionMetadata


def compressed_cu_seqlens(cu_seqlens: torch.Tensor, ratio: int) -> torch.Tensor:
    """Pack complete groups per document; discard each trailing partial group."""
    counts = torch.diff(cu_seqlens) // ratio
    return F.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))


class SparseAttentionMetadata(NamedTuple):
    """Immutable geometry, constructed once per variant per microbatch.

    Token-indexed fields are local to the complete DP microbatch. CP must shard
    or permute them along with query tokens before it can use this metadata.
    """

    varlen: VarlenAttentionMetadata
    cu_seq_cmp: torch.Tensor
    cmp_base: torch.Tensor
    causal_cmp: torch.Tensor | None
    group_start: torch.Tensor
    group_first: torch.Tensor
    group_position: torch.Tensor
    aligned: bool = False

    @property
    def cu_seq_q(self) -> torch.Tensor:
        return self.varlen.cu_seq_q

    @property
    def max_k(self) -> int:
        return self.varlen.max_k

    def annotate_spmd_types(self) -> None:
        self.varlen.annotate_spmd_types()
        # Each DP rank owns its document geometry; CP is currently unsupported.
        for tensor in (
            self.cu_seq_cmp,
            self.cmp_base,
            self.group_start,
            self.group_first,
            self.group_position,
            self.causal_cmp,
        ):
            if tensor is not None:
                spmd.assert_type(
                    tensor,
                    {"dp": spmd.V, "cp": spmd.R, "tp": spmd.R},
                    partition_spec=spmd.PartitionSpec(
                        "dp", *([None] * (tensor.ndim - 1))
                    ),
                )


def build_sparse_metadata(
    varlen: VarlenAttentionMetadata,
    positions: torch.Tensor,
    ratio: int,
    *,
    aligned: bool = False,
) -> SparseAttentionMetadata:
    cu = varlen.cu_seq_q
    if aligned:
        torch._assert_async(
            (cu % ratio == 0).all(),
            "Packed document boundaries must match document_alignment",
        )
    cmp_cu = compressed_cu_seqlens(cu, ratio)
    tokens = torch.arange(positions.numel(), device=positions.device)
    doc = torch.searchsorted(cu[1:], tokens, right=True, out_int32=True)
    base = cmp_cu[doc]
    causal = None
    if ratio != 4:
        width = min(positions.numel() // ratio, varlen.max_k // ratio)
        candidates = torch.arange(width, dtype=torch.int32, device=positions.device)
        limit = (tokens - cu[doc] + 1) // ratio
        causal = torch.where(candidates < limit[:, None], candidates, -1)
    slots = torch.arange(positions.numel() // ratio, device=positions.device)
    group_doc = torch.searchsorted(cmp_cu[1:], slots, right=True, out_int32=True)
    group = slots - cmp_cu[group_doc]
    start = torch.where(slots < cmp_cu[-1], cu[group_doc] + group * ratio, 0)
    return SparseAttentionMetadata(
        varlen, cmp_cu, base, causal, start, group == 0, group * ratio, aligned
    )
