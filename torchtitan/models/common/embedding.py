# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.distributed.spmd_types import current_spmd_mesh, spmd_mesh_size
from torchtitan.protocols.module import Module


class Embedding(nn.Embedding, Module):
    """Configurable embedding with optional local vocab-parallel execution."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int
        embedding_dim: int

    def __init__(self, config: Config):
        super().__init__(config.num_embeddings, config.embedding_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Runs vocab-parallel embedding when the current mesh has a TP axis."""
        weight = self.weight
        tp_size = spmd_mesh_size("tp")
        if tp_size == 1 or isinstance(weight, DTensor):
            return F.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

        mesh = current_spmd_mesh()
        assert mesh is not None
        tp_pg = mesh.get_group("tp")
        chunk_size = (self.num_embeddings + tp_size - 1) // tp_size
        offset = dist.get_rank(tp_pg) * chunk_size
        mask = (input >= offset) & (input < offset + weight.shape[0])
        local_input = (input - offset).clamp(0, weight.shape[0] - 1)
        out = F.embedding(
            local_input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return out * mask.unsqueeze(-1).to(out.dtype)


__all__ = ["Embedding"]
