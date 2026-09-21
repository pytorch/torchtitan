# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock

from .state_dict_adapter import Qwen3StateDictAdapter


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
        *,
        padding_mask: torch.Tensor | None = None,
    ):
        x = x + self.attention(self.attention_norm(x), attention_masks, positions)

        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x), padding_mask_T=padding_mask)
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Qwen3Model(Decoder):
    state_dict_adapter_cls = Qwen3StateDictAdapter

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallel_dims) -> None:
        from torchtitan.components.optimizer import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallel_dims)

    """
    Qwen3Model Module

    Args:
        config (Qwen3Model.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 1024
        vocab_size: int = 151936

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            from torchtitan.models.qwen3.sharding import set_qwen3_sharding_config

            set_qwen3_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )
