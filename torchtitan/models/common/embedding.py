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

    def __init__(self, config: Config):
        super().__init__(config.num_embeddings, config.embedding_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run vocab-parallel embedding when the active mesh has a TP group."""
        tp_group = spmd_mesh_group("tp")
        if tp_group is None:
            return F.embedding(
                input,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

        tp_size = dist.get_world_size(tp_group)
        chunk_size = (self.num_embeddings + tp_size - 1) // tp_size
        offset = dist.get_rank(tp_group) * chunk_size
        mask = (input >= offset) & (input < offset + self.weight.shape[0])
        local_input = (input - offset).clamp(0, self.weight.shape[0] - 1)
        # padding_idx is global; only its owning shard should suppress gradients.
        local_padding_idx = None
        if (
            self.padding_idx is not None
            and offset <= self.padding_idx < offset + self.weight.shape[0]
        ):
            local_padding_idx = self.padding_idx - offset
        # V: local vocabulary rows; D: embedding dimension.
        weight_VD = self.weight
        if self.scale_grad_by_freq:
            # Nonlocal tokens need a separate row to avoid inflating the
            # frequency of real tokens at the clamped vocabulary boundaries.
            local_input = torch.where(mask, local_input, weight_VD.shape[0])
            weight_VD = F.pad(weight_VD, (0, 0, 0, 1))
        out = F.embedding(
            local_input,
            weight_VD,
            local_padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        if self.scale_grad_by_freq and self.max_norm is not None:
            # F.embedding renormalizes in place; preserve that on the parameter
            # after the lookup used a padded copy.
            # TODO: As with F.embedding, FSDP2 resharding discards forward-time
            # max_norm updates to the transient all-gathered parameter.
            with torch.no_grad():
                self.weight.copy_(weight_VD[:-1])
        return out * mask.unsqueeze(-1).to(out.dtype)


__all__ = ["Embedding"]
