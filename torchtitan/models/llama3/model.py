# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.tools.logging import logger


class Llama3TransformerBlock(TransformerBlock):
    """
    Llama3 TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        dim (int): Model dimension.
        n_layers (int): Total number of layers.
        config (Llama3TransformerBlock.Config): Block configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        depth_init: bool = True

    def __init__(self, config: Config, *, layer_id: int, dim: int, n_layers: int):
        super().__init__()
        self.attention = config.attention.build(dim=dim)
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build(dim=dim)
        self.attention_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=config.norm_eps)

        if config.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, **kwargs):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Llama3Model(Decoder):
    """
    Llama3Model Module

    Args:
        config (Llama3Model.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        n_layers: int = 32
        vocab_size: int = 128256
        layer: TransformerBlock.Config

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )
            # Sync rope max_seq_len
            import dataclasses as _dc

            self.rope = _dc.replace(self.rope, max_seq_len=seq_len)

            if (
                parallelism.context_parallel_degree > 1
                and self.layer.attention.attn_backend == "varlen"
            ):
                raise NotImplementedError(
                    f"Context Parallel only supports SDPA and FlexAttention."
                    f"Got attn_backend='{self.layer.attention.attn_backend}'. "
                    f"Varlen attention is not supported with CP."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_dense_model_nparams_and_flops(
                self,
                model,
                self.layer.attention.n_heads,
                2 * (self.dim // self.layer.attention.n_heads),
                seq_len,
            )
