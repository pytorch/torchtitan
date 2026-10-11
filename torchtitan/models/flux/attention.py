# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffix legend:
#   B = batch, L = sequence length, T = packed tokens, H = attention heads,
#   K = query/key head dimension, V = value head dimension,
#   N = number of sequence boundaries (B + 1).

from dataclasses import dataclass, field

import torch

from torchtitan.models.common.attention import (
    InnerAttention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)


class FluxInnerAttention(InnerAttention):
    """Fold Flux batches into independent bidirectional Varlen sequences.

    The outer Flux blocks retain their batch axis for per-sample conditioning.
    CP gathers keys and values before this boundary, so their sequence length
    can differ from the local query length.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        varlen_attention: VarlenInnerAttention.Config = field(
            default_factory=lambda: VarlenInnerAttention.Config(window_size=(-1, -1))
        )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.varlen_attention = config.varlen_attention.build()

    def forward(
        self,
        q_BLHK: torch.Tensor,
        k_BLHK: torch.Tensor,
        v_BLHV: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_len = q_BLHK.shape[:2]
        kv_len = k_BLHK.shape[1]
        if k_BLHK.shape[0] != batch_size or v_BLHV.shape[:2] != (batch_size, kv_len):
            raise ValueError(
                "Flux attention requires matching Q/K/V batches and K/V lengths."
            )

        sequence_ids_N = torch.arange(
            batch_size + 1, device=q_BLHK.device, dtype=torch.int32
        )
        attention_metadata = VarlenAttentionMetadata(
            cu_seq_q=sequence_ids_N * query_len,
            cu_seq_k=sequence_ids_N * kv_len,
            max_q=query_len,
            max_k=kv_len,
        )
        out_THV = self.varlen_attention(
            q_BLHK.flatten(0, 1),
            k_BLHK.flatten(0, 1),
            v_BLHV.flatten(0, 1),
            attention_metadata=attention_metadata,
        )
        return out_THV.unflatten(0, (batch_size, query_len))
