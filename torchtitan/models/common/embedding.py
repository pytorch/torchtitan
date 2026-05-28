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

from torchtitan.distributed.spmd_state import current_mesh, is_spmd_active
from torchtitan.protocols.module import Module

__all__ = ["Embedding"]


class Embedding(nn.Embedding, Module):
    """Configurable nn.Embedding.

    Uses diamond inheritance (nn.Embedding + Module) so that:
    - The module hierarchy stays flat (no extra wrapper layer).
    - All nn.Embedding logic (forward, state_dict, etc.) is reused as-is.
    - The Module protocol is satisfied and ``build()`` is inherited from
      ``Configurable.Config``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int
        embedding_dim: int
        enable_sp: bool | None = None

    def __init__(self, config: Config):
        super().__init__(config.num_embeddings, config.embedding_dim)
        self.enable_sp = config.enable_sp

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        tp_pg = None
        mesh = current_mesh() if is_spmd_active() else None
        if mesh is not None:
            assert mesh.mesh_dim_names is not None
            if "tp" in mesh.mesh_dim_names:
                tp_pg = mesh.get_group("tp")

        # standard F.embedding
        if mesh is None or tp_pg is None or dist.get_world_size(tp_pg) == 1:
            return F.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

        # vocab-parallel embedding w/ sharded vocab dim; allreduce over masked embeddings.
        assert self.enable_sp is not None
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
        tp_out_type = spmd.S(1) if self.enable_sp else spmd.I
        return spmd.redistribute(
            out,
            tp_pg,
            src=spmd.P,
            dst=tp_out_type,
            backward_options=bwd,
        )
