# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.protocols.module import Module

__all__ = ["Embedding"]


class Embedding(nn.Embedding, Module):
    """Configurable nn.Embedding with optional vocab-parallel TP support.

    Without TP, forward is standard ``F.embedding``.
    With TP, forward uses masked-partial: each TP rank looks up its local
    slice of the vocab table, masks out-of-range tokens, and reduces
    across TP ranks.

    TP state is set by ``_setup_tp`` at parallelize time. The forward
    should be wrapped with ``LocalSpmdConfig`` so that SPMD type
    annotations are handled at the boundary.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int
        embedding_dim: int

    def __init__(self, config: Config):
        super().__init__(config.num_embeddings, config.embedding_dim)
        self._tp_out_type = None

    def _setup_tp(self, tp_out_type) -> None:
        """Configure TP vocab-parallel output type. Called at parallelize time."""
        self._tp_out_type = tp_out_type

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        from torchtitan.distributed.spmd_state import spmd_state

        state = spmd_state()
        tp_pg = state.tp_pg
        weight = self.weight
        chunk_size = weight.shape[0]
        offset = dist.get_rank(tp_pg) * chunk_size if tp_pg is not None else 0
        mask = (input >= offset) & (input < offset + chunk_size)
        local_input = (input - offset).clamp(0, chunk_size - 1)
        out = F.embedding(local_input, weight)
        out = out * mask.unsqueeze(-1).to(out.dtype)
        if tp_pg is not None:
            out = spmd.redistribute(
                out, tp_pg, src=spmd.P, dst=self._tp_out_type
            )
        return out
