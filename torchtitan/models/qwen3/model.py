# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from dataclasses import dataclass, field

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType, GQAttention
from torchtitan.models.common.decoder import (
    Decoder,
    DecoderStateInitializer,
    TransformerBlock,
)
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.state_initializer import StateInitializer
from torchtitan.tools.logging import logger


class Qwen3TransformerBlockStateInitializer(StateInitializer):
    @dataclass(kw_only=True, slots=True)
    class Config(StateInitializer.Config):
        depth_init: bool = True

    def init_states(self, module, *, buffer_device=None) -> None:
        for norm in (module.attention_norm, module.ffn_norm):
            norm.init_states()
        module.attention.init_states()
        if module.moe_enabled:
            module.moe.init_states(buffer_device=buffer_device)
        else:
            module.feed_forward.init_states()


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
        moe_enabled: bool = False
        state_initializer: StateInitializer.Config = field(
            default_factory=lambda: Qwen3TransformerBlockStateInitializer.Config(
                depth_init=True
            )
        )

    def __init__(self, config: Config, *, layer_id: int, dim: int, n_layers: int):
        super().__init__(config)

        assert isinstance(
            config.state_initializer, Qwen3TransformerBlockStateInitializer.Config
        )
        if config.state_initializer.depth_init:
            weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            weight_init_std = 0.02 / (2 * n_layers) ** 0.5

        # Replace init_std on output projections before building
        attn_cfg = config.attention.replace_state_init_field(init_std=weight_init_std)
        self.attention = attn_cfg.build(dim=dim)

        self.moe_enabled = config.moe_enabled
        if self.moe_enabled:
            assert config.moe is not None
            moe_cfg = config.moe.replace_state_init_field(init_std=weight_init_std)
            self.moe = moe_cfg.build(dim=dim)
        else:
            assert config.feed_forward is not None
            ff_cfg = config.feed_forward.replace_state_init_field(
                init_std=weight_init_std
            )
            self.feed_forward = ff_cfg.build(dim=dim)

        self.attention_norm = config.attention_norm.build(normalized_shape=dim)
        self.ffn_norm = config.ffn_norm.build(normalized_shape=dim)

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


class Qwen3DecoderStateInitializer(DecoderStateInitializer):
    @dataclass(kw_only=True, slots=True)
    class Config(DecoderStateInitializer.Config):
        enable_weight_tying: bool = False

    def __init__(self, config: Config):
        self.enable_weight_tying = config.enable_weight_tying

    def init_states(self, module, *, buffer_device=None) -> None:
        if self.enable_weight_tying:
            # since when the model is initialized on meta device,
            # the tying in the __init__ may not have worked correctly
            # we ensure the weights are tied here
            assert module.tok_embeddings is not None and module.output is not None
            module.tok_embeddings.weight = module.output.weight
        super().init_states(module, buffer_device=buffer_device)


class Qwen3Model(Decoder):
    """
    Qwen3Model Module

    Args:
        config (Qwen3Model.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 1024
        n_layers: int = 28
        vocab_size: int = 151936
        enable_weight_tying: bool = False
        layer: TransformerBlock.Config
        state_initializer: StateInitializer.Config = field(
            default_factory=lambda: Qwen3DecoderStateInitializer.Config(
                enable_weight_tying=False
            )
        )

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
            import dataclasses as _dc

            self.rope = _dc.replace(self.rope, max_seq_len=seq_len)

            if self.layer.moe is not None:
                self.layer.moe._debug_force_load_balance = debug.moe_force_load_balance

            if (
                parallelism.context_parallel_degree > 1
                and self.layer.attention.attn_backend == "varlen"
            ):
                raise NotImplementedError(
                    f"Context Parallel only supports SDPA and FlexAttention."
                    f"Got attn_backend='{self.layer.attention.attn_backend}'. "
                    f"Varlen attention is not supported with CP."
                )

            if self.enable_weight_tying and parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "Weight tying is not supported with Pipeline Parallel."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            assert isinstance(self.layer.attention, GQAttention.Config)
            assert self.layer.attention.head_dim is not None
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layer.attention.n_heads,
                2 * self.layer.attention.head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.enable_weight_tying = config.enable_weight_tying

        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.output.weight

        # Rebuild state_initializer to pick up enable_weight_tying
        si_config = Qwen3DecoderStateInitializer.Config(
            enable_weight_tying=config.enable_weight_tying
        )
        self._state_initializer = Qwen3DecoderStateInitializer(si_config)
