# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.protocols.module import Module


class Embedding(nn.Embedding, Module):
    """
    Configurable embedding with optional local vocab-parallel execution.
    TODO(pianpwk): rename to VocabParallelEmbedding
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int
        embedding_dim: int
        padding_idx: int | None = None

    def __init__(self, config: Config):
        super().__init__(
            config.num_embeddings, config.embedding_dim, padding_idx=config.padding_idx
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run vocab-parallel embedding when the active mesh has a TP group."""
        # Weight tying may replace the standard [V, D] embedding parameter with
        # a Linear's [1, V, D] parameter. Both represent the same lookup table.
        weight_VD = self.weight.flatten(0, -2)
        tp_group = spmd_mesh_group("tp")
        if tp_group is None:
            return F.embedding(
                input,
                weight_VD,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

        tp_size = dist.get_world_size(tp_group)
        local_num_embeddings = weight_VD.shape[0]
        chunk_size = (self.num_embeddings + tp_size - 1) // tp_size
        offset = dist.get_rank(tp_group) * chunk_size
        mask = (input >= offset) & (input < offset + local_num_embeddings)
        local_input = (input - offset).clamp(0, local_num_embeddings - 1)
        # padding_idx is global; only its owning shard should suppress gradients.
        local_padding_idx = None
        if (
            self.padding_idx is not None
            and offset <= self.padding_idx < offset + local_num_embeddings
        ):
            local_padding_idx = self.padding_idx - offset
        out = F.embedding(
            local_input,
            weight_VD,
            local_padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return out * mask.unsqueeze(-1).to(out.dtype)


__all__ = ["Embedding"]
