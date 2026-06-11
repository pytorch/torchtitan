# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.protocols.module import Module

if TYPE_CHECKING:
    from torchtitan.distributed import ParallelDims


class Embedding(nn.Embedding, Module):
    """Configurable embedding with optional local vocab-parallel execution."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int
        embedding_dim: int

    def __init__(self, config: Config):
        super().__init__(config.num_embeddings, config.embedding_dim)
        self.tp_group: dist.ProcessGroup | None = None

    def parallelize(self, parallel_dims: "ParallelDims") -> None:
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        if tp_mesh is not None:
            self.tp_group = tp_mesh.get_group("tp")
        super().parallelize(parallel_dims)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Runs vocab-parallel embedding when the module has a TP group."""
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        if self.tp_group is None:
            return F.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

        # TODO(pianpwk): Once DTensor backend is removed, delete ``tp_group`` and
        # use ``current_spmd_mesh().get_group("tp")`` here instead.
        tp_pg = self.tp_group
        tp_size = dist.get_world_size(tp_pg)
        weight = weight.to_local() if isinstance(weight, DTensor) else weight
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
