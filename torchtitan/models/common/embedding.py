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

from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.distributed.utils import device_mesh_axis_coordinate, get_spmd_backend
from torchtitan.protocols.module import Module

if TYPE_CHECKING:
    from torchtitan.distributed import ParallelDims


class Embedding(nn.Embedding, Module):
    """
    Configurable embedding with optional local vocab-parallel execution.
    TODO(pianpwk): rename to VocabParallelEmbedding
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int
        embedding_dim: int

    def __init__(self, config: Config):
        super().__init__(config.num_embeddings, config.embedding_dim)
        self.tp_group: dist.ProcessGroup | None = None

    def parallelize(self, parallel_dims: "ParallelDims") -> None:
        # TODO(pianpwk): delete and rely on `current_spmd_mesh().get_group("tp")`
        # once the partial_dtensor backend is removed.
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        if tp_mesh is not None:
            self.tp_group = tp_mesh.get_group("tp")
        super().parallelize(parallel_dims)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Runs vocab-parallel embedding when the module has a TP group."""
        parameter = self.weight
        weight = parameter.to_local() if isinstance(parameter, DTensor) else parameter
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

        tp_pg = self.tp_group
        tp_size = dist.get_world_size(tp_pg)
        weight = weight.to_local() if isinstance(weight, DTensor) else weight
        chunk_size = (self.num_embeddings + tp_size - 1) // tp_size
        if isinstance(parameter, DTensor):
            tp_rank = device_mesh_axis_coordinate(parameter.device_mesh, "tp")
        elif get_spmd_backend() == "spmd_types":
            mesh = current_spmd_mesh()
            assert mesh is not None
            tp_rank = device_mesh_axis_coordinate(mesh, "tp")
        else:
            tp_rank = dist.get_rank(tp_pg)
        offset = tp_rank * chunk_size
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
