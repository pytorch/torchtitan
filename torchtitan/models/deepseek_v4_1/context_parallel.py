# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context parallelism stubs for DeepSeek V4.1.

The full CP implementation moves the query side via all-to-all while keeping
the KV side gathered. On GPU without fused kernels, CP is not supported;
this module provides the identity ``QueryExchange`` that the attention
module imports unconditionally.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torchtitan.models.base.module import Module


class ExchangeMetadata(NamedTuple):
    positions_q: torch.Tensor
    send_counts: torch.Tensor
    recv_counts: torch.Tensor
    send_offsets: torch.Tensor
    recv_offsets: torch.Tensor


def document_bounds(positions: torch.Tensor) -> torch.Tensor:
    """Return packed document boundaries ``[N + 1]`` as int32.

    A new document starts wherever the position resets to 0.
    """
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    batch, seqlen = positions.shape
    resets = positions[:, :-1] > positions[:, 1:]
    starts = torch.cat([
        torch.ones(batch, 1, dtype=torch.bool, device=positions.device),
        resets,
    ], dim=1)
    cu_seqlens_list = []
    for b in range(batch):
        indices = starts[b].nonzero(as_tuple=False).squeeze(-1)
        cu = torch.zeros(indices.numel() + 1, dtype=torch.int32, device=positions.device)
        cu[1:] = indices.to(torch.int32)
        cu[-1] = seqlen
        cu_seqlens_list.append(cu)
    if batch == 1:
        return cu_seqlens_list[0]
    return torch.cat(cu_seqlens_list)


def cp_plan(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    num_tokens: int,
) -> ExchangeMetadata:
    """Identity plan for CP=1: no movement."""
    positions_q = torch.arange(num_tokens, dtype=torch.int32, device=cu_seqlens.device)
    zeros = torch.zeros(1, dtype=torch.int32, device=cu_seqlens.device)
    return ExchangeMetadata(
        positions_q=positions_q,
        send_counts=zeros,
        recv_counts=zeros,
        send_offsets=zeros,
        recv_offsets=zeros,
    )


class QueryExchange(Module):
    """Identity query exchange for CP=1."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def permute(self, x: torch.Tensor, exchange: ExchangeMetadata | None) -> torch.Tensor:
        return x

    def unpermute(self, x: torch.Tensor, exchange: ExchangeMetadata | None) -> torch.Tensor:
        return x
