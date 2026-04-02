# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.models.common.attention import (
    AttentionMasksType,
    GQAttention,
    VarlenAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.tools.logging import logger


class Qwen3TransformerBlock(TransformerBlock):
    """
    Qwen3 TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        dim (int): Model dimension.
        n_layers (int): Total number of layers.
        config (Qwen3TransformerBlock.Config): Block configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

        self.attention = config.attention.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        x = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )

        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Qwen3Model(Decoder):
    """
    Qwen3Model Module

    Args:
        config (Qwen3Model.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 1024
        vocab_size: int = 151936
        enable_weight_tying: bool = False

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:

            training = trainer_config.training
            parallelism = trainer_config.parallelism
            debug = trainer_config.debug
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )
            # Sync rope max_seq_len
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            for layer_cfg in self.layers:
                if layer_cfg.moe is not None:
                    layer_cfg.moe.router._debug_force_load_balance = (
                        debug.moe_force_load_balance
                    )

            if parallelism.context_parallel_degree > 1 and isinstance(
                self.layers[0].attention.inner_attention, VarlenAttention.Config
            ):
                raise NotImplementedError(
                    "Context Parallel only supports SDPA and FlexAttention. "
                    "Varlen attention is not supported with CP."
                )

            if self.enable_weight_tying and parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "Weight tying is not supported with Pipeline Parallel."
                )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                n_heads = self.layers[0].attention.n_heads
                # pyrefly: ignore [missing-attribute]
                n_kv_heads = self.layers[0].attention.n_kv_heads or n_heads
                if n_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_heads ({n_heads})."
                    )
                if n_kv_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_kv_heads ({n_kv_heads})."
                    )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:

            assert isinstance(self.layers[0].attention, GQAttention.Config)
            assert self.layers[0].attention.head_dim is not None
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layers[0].attention.n_heads,
                2 * self.layers[0].attention.head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.enable_weight_tying = config.enable_weight_tying

        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.output.weight

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        if self.enable_weight_tying:
            # Re-tie before init: on meta device the __init__ tying may
            # not have worked correctly.
            assert self.tok_embeddings is not None and self.output is not None
            self.tok_embeddings.weight = self.output.weight

        super().init_states(buffer_device=buffer_device)
