# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.models.base.module import Module
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    token_id_placement,
)

_TOKEN_IDS = token_id_placement()
_PARTIAL_OUTPUT = dense_activation_placement(tp=spmd.P, cp=spmd.S(0))


class VocabParallelEmbedding(nn.Embedding, Module):
    """Token embedding whose weight is vocab-sharded across TP.

    With a TP group, each rank looks up only the token ids in its vocab shard,
    giving a TP-``Partial`` output, then reduces it to the decoder residual
    layout: ``Partial -> Shard(0)`` is a reduce-scatter under sequence
    parallelism, and ``Partial -> Invariant`` is an all-reduce without it.
    Without TP it is a plain embedding lookup.
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
        tp_group = spmd_mesh_group(MeshAxisName.TP)
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
        output = self._local_lookup_vocab_shard(input)
        return spmd.redistribute(
            output,
            tp_group,
            src=spmd.P,
            dst=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
            backward_options={"op_dtype": output.dtype},
        )

    # The masked lookup indexes the vocab shard with bare tensor arithmetic, so
    # it runs as a local region: TP-replicated token ids in, TP-partial
    # embeddings out.
    @spmd.local_map(
        in_types=(None, (_TOKEN_IDS.local_type, _TOKEN_IDS.partition_spec)),
        out_types=(_PARTIAL_OUTPUT.local_type, _PARTIAL_OUTPUT.partition_spec),
    )
    def _local_lookup_vocab_shard(self, input: torch.Tensor) -> torch.Tensor:
        return self._lookup_vocab_shard(input)

    def _lookup_vocab_shard(self, input: torch.Tensor) -> torch.Tensor:
        """Return this rank's TP-partial lookup: the embeddings of the tokens in
        its vocab shard, and zeros for every other token."""
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        assert tp_group is not None
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
        out = F.embedding(
            local_input,
            self.weight,
            local_padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return out * mask.unsqueeze(-1).to(out.dtype)


__all__ = ["VocabParallelEmbedding"]
