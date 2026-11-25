# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel pieces shared by the linear-attention families."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch.export import register_dataclass

from torchtitan.distributed.context_parallel import get_token_fragments
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group

from torchtitan.models.common.attn_gym import (
    _ContextParallelChunk,
    ContextParallelRouting,
    is_missing_kernel,
)

from .linear_attention import LinearAttentionMetadata

if not is_missing_kernel(_ContextParallelChunk):
    spmd.register_local_autograd_function(_ContextParallelChunk)


@dataclass(frozen=True, slots=True)
class UlyssesLinearAttentionMetadata(LinearAttentionMetadata):
    """Token order shared with load-balanced full-attention layers."""

    permutation: torch.Tensor
    inverse_permutation: torch.Tensor

    _PERMUTATION_SPMD_TYPE = spmd.SpmdType(
        {
            MeshAxisName.DP: spmd.V,
            MeshAxisName.CP: spmd.R,
            MeshAxisName.TP: spmd.R,
        },
        partition_spec=spmd.PartitionSpec(MeshAxisName.DP),
    )

    def annotate_spmd_types(self) -> None:
        """Annotate global token orders replicated across CP and TP."""
        LinearAttentionMetadata.annotate_spmd_types(self)
        spmd.assert_type(self.permutation, self._PERMUTATION_SPMD_TYPE)
        spmd.assert_type(self.inverse_permutation, self._PERMUTATION_SPMD_TYPE)


def prepare_ulysses_linear_attention_metadata(
    attention_metadata: LinearAttentionMetadata,
    *,
    permutation: torch.Tensor | None,
) -> LinearAttentionMetadata:
    """Build Ulysses metadata for computation in original token order."""
    if permutation is None:
        return attention_metadata
    if permutation.ndim != 2 or permutation.shape[0] != 1:
        raise ValueError(
            "Ulysses linear attention requires a packed-token permutation "
            "with shape [1, num_tokens]."
        )
    permutation_T = permutation[0]
    return UlyssesLinearAttentionMetadata(
        varlen=attention_metadata.varlen,
        num_conv_history_tokens=attention_metadata.num_conv_history_tokens,
        permutation=permutation_T,
        inverse_permutation=permutation_T.argsort(),
    )


@dataclass(frozen=True, slots=True)
class ContextParallelLinearAttentionMetadata(LinearAttentionMetadata):
    """Linear-attention metadata with rank-local context-parallel routing."""

    cp_routing: ContextParallelRouting

    _ROUTING_SPMD_TYPE = spmd.SpmdType(
        {
            MeshAxisName.DP: spmd.V,
            MeshAxisName.CP: spmd.V,
            MeshAxisName.TP: spmd.R,
        }
    )

    def annotate_spmd_types(self) -> None:
        """Annotate sequence offsets and rank-local CP routing tensors."""
        LinearAttentionMetadata.annotate_spmd_types(self)
        for value in vars(self.cp_routing).values():
            if isinstance(value, torch.Tensor):
                spmd.assert_type(value, self._ROUTING_SPMD_TYPE)


# pytree registration is per class; subclasses are not covered by the base.
register_dataclass(UlyssesLinearAttentionMetadata)
register_dataclass(ContextParallelLinearAttentionMetadata)
# TODO: delete once Attention Gym registers ContextParallelRouting:
# https://github.com/meta-pytorch/attention-gym/issues/850
if not is_missing_kernel(ContextParallelRouting):
    register_dataclass(ContextParallelRouting)


def prepare_cp_linear_attention_metadata(
    attention_metadata: LinearAttentionMetadata,
    *,
    permutation: torch.Tensor | None,
    family: str,
) -> ContextParallelLinearAttentionMetadata:
    """Build rank-local routing from global linear-attention sequence metadata.

    ``family`` names the attention (e.g. "KDA") in error messages.
    """
    if attention_metadata.varlen is None:
        if permutation is None:
            raise ValueError(
                f"{family} context parallelism requires global "
                "VarlenAttentionMetadata when using contiguous CP sharding."
            )
        cu_seqlens_global = [0, permutation.shape[1]]
        device = permutation.device
    else:
        cu_seqlens_global = attention_metadata.varlen.cu_seq_q.tolist()
        device = attention_metadata.varlen.cu_seq_q.device

    group = spmd_mesh_group(MeshAxisName.CP)
    if group is None:
        raise RuntimeError(
            f"{family} metadata preparation requires an active multi-rank CP "
            "mesh axis."
        )
    # Upper bound on token ranges per rank, so routing shapes stay fixed across
    # batches: contiguous sharding gives one range, head-tail gives a head and a
    # tail chunk. PTRR is rejected for linear attention at config time.
    max_num_fragments = 1 if permutation is None else 2
    # Each fragment boundary can split one document into two subsequences.
    max_subsequences = len(cu_seqlens_global) - 1 + max_num_fragments - 1
    routing = ContextParallelRouting.from_fragments(
        cu_seqlens_global=cu_seqlens_global,
        fragments=get_token_fragments(
            cu_seqlens_global[-1],
            cp_size=group.size(),
            permutation=permutation,
        ),
        cp_rank=dist.get_rank(group),
        device=device,
        slots=max_num_fragments,
        max_subsequences=max_subsequences,
        conv_history=attention_metadata.num_conv_history_tokens,
    )
    return ContextParallelLinearAttentionMetadata(
        varlen=attention_metadata.varlen,
        num_conv_history_tokens=attention_metadata.num_conv_history_tokens,
        cp_routing=routing,
    )


def check_num_head_chunks(num_head_chunks: int) -> None:
    """Raise unless ``num_head_chunks`` is positive."""
    if num_head_chunks < 1:
        raise ValueError(f"num_head_chunks must be positive, got {num_head_chunks}")


def chunk_heads(
    kernel: Callable[..., tuple[torch.Tensor, Any]],
    inputs_1TH: Sequence[torch.Tensor],
    num_head_chunks: int,
    **kwargs: Any,
) -> torch.Tensor:
    """Call a CP linear-attention ``kernel`` once per contiguous chunk of heads.

    Heads are independent, so the output is unchanged and each call's kernel
    workspace shrinks by ``num_head_chunks``. ``inputs_1TH`` are
    ``[1, T, H, ...]``, key heads first; contiguous chunks keep each value head
    with its key head.

    TODO(acisseJZhong, meta-pytorch/attention-gym#986): Move into
    attention-gym so the chunks share one state exchange.
    """
    # validate_context_parallel checks this at config time.
    assert inputs_1TH[0].shape[2] % num_head_chunks == 0
    if num_head_chunks == 1:
        return kernel(*inputs_1TH, **kwargs)[0]
    chunks = zip(*(t.chunk(num_head_chunks, dim=2) for t in inputs_1TH))
    outputs_1THV = [kernel(*chunk, **kwargs)[0] for chunk in chunks]
    return torch.cat(outputs_1THV, dim=2)
