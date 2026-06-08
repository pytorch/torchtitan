# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.models.common.attention import (
    AttentionMasksType,
    GQAttention,
    VarlenAttention,
)
from torchtitan.models.common.config_utils import update_moe_aux_loss_configs
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_moe_model_nparams_and_flops


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
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        x = x + self.attention(self.attention_norm(x), attention_masks, positions)

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
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            update_moe_aux_loss_configs(
                self.layers,
                pp_enabled=parallelism.pipeline_parallel_degree > 1,
                local_global_batch_size=kwargs["local_global_batch_size"],
                global_batch_size=kwargs["global_batch_size"],
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

            from torchtitan.models.qwen3.sharding import set_qwen3_sharding_config

            set_qwen3_sharding_config(
                self,
                loss_parallel=not parallelism.disable_loss_parallel,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
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
            self.tok_embeddings.weight = self.lm_head.weight

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        if self.enable_weight_tying:
            # Re-tie before init: on meta device the __init__ tying may
            # not have worked correctly.
            assert self.tok_embeddings is not None and self.lm_head is not None
            self.tok_embeddings.weight = self.lm_head.weight

        super().init_states(buffer_device=buffer_device)
