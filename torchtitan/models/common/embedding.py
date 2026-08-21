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
from torchtitan.protocols.module import Module

if TYPE_CHECKING:
    from torchtitan.distributed import ParallelDims


def get_tp_rank(tp_group: dist.ProcessGroup) -> int | torch.SymInt:
    """Return the TP rank, using a runtime symbol only during CooR tracing.

    ``DeviceMesh._sym_get_coordinate`` returns the concrete coordinate in eager
    execution and emits a runtime coordinate op only under compile-on-one-rank
    fake tracing.
    """
    mesh = current_spmd_mesh()
    if mesh is None:
        return dist.get_rank(tp_group)

    mesh_axis_names = mesh.mesh_dim_names
    assert mesh_axis_names is not None, "DeviceMesh must have named axes"
    if "tp" not in mesh_axis_names:
        raise ValueError(
            f"TP rank requires a 'tp' mesh axis, but got {mesh_axis_names}."
        )
    return mesh._sym_get_coordinate(mesh_axis_names.index("tp"))


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
        # TODO(pianpwk): delete and rely on `get_mesh_pg("tp")`
        # once the partial_dtensor backend is removed.
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        if tp_mesh is not None:
            self.tp_group = tp_mesh.get_group("tp")
        super().parallelize(parallel_dims)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Runs vocab-parallel embedding when the module has a TP group."""
        weight = (
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        tp_group = self.tp_group
        if tp_group is None:
            return F.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

        tp_rank = get_tp_rank(tp_group)
        tp_size = dist.get_world_size(tp_group)
        weight = weight.to_local() if isinstance(weight, DTensor) else weight
        chunk_size = (self.num_embeddings + tp_size - 1) // tp_size
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
