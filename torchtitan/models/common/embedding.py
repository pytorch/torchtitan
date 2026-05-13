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

from torchtitan.distributed.spmd_state import is_spmd_active, spmd_state
from torchtitan.protocols.module import Module

__all__ = ["Embedding"]


class Embedding(nn.Embedding, Module):
    """Configurable nn.Embedding with optional vocab-parallel TP support.

    Without TP, forward is standard ``F.embedding``.
    With SPMD TP, each rank looks up its local vocab shard, masks out-of-range
    tokens, and reduces partial outputs to the configured TP output type.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int
        embedding_dim: int
        tp_out_type: spmd.PerMeshAxisSpmdType | None = None

    def __init__(self, config: Config):
        super().__init__(config.num_embeddings, config.embedding_dim)
        self.tp_out_type = config.tp_out_type

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        tp_pg = spmd_state().get_pg("tp") if is_spmd_active() else None
        if tp_pg is None:
            return F.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

        assert self.tp_out_type is not None
        chunk_size = weight.shape[0]
        offset = dist.get_rank(tp_pg) * chunk_size
        mask = (input >= offset) & (input < offset + chunk_size)
        local_input = (input - offset).clamp(0, chunk_size - 1)
        out = F.embedding(
            local_input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        out = out * mask.unsqueeze(-1).to(out.dtype)
        bwd = {"op_dtype": torch.float32} if out.dtype != torch.float32 else None
        return spmd.redistribute(
            out,
            tp_pg,
            src=spmd.P,
            dst=self.tp_out_type,
            backward_options=bwd,
        )
