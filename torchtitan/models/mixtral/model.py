# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.common.attention import (
    AttentionMasksType,
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_sliding_window_mask_mod,
    GQAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


class MixtralTransformerBlock(TransformerBlock):
    """
    Mixtral TransformerBlock with MoE feed-forward on every layer.

    Args:
        config (MixtralTransformerBlock.Config): Block configuration.
        layer_id (int): Identifier for the layer.
        dim (int): Model dimension.
        n_layers (int): Total number of layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        depth_init: bool = True

    def __init__(self, config: Config, *, layer_id: int, dim: int, n_layers: int):
        super().__init__()

        self.attention = config.attention.build(dim=dim)

        assert config.moe is not None, "Mixtral requires MoE config for all layers"
        self.moe = config.moe.build(dim=dim)
        self.moe_enabled = True

        self.attention_norm = config.attention_norm.build(normalized_shape=dim)
        self.ffn_norm = config.ffn_norm.build(normalized_shape=dim)

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
        x = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        x = x + self.moe(self.ffn_norm(x))
        return x

    def init_weights(self, **kwargs):
        buffer_device: torch.device | None = kwargs.get("buffer_device")
        for norm in (self.attention_norm, self.ffn_norm):
            norm.init_weights()
        self.attention.init_weights(self.weight_init_std)
        self.moe.init_weights(
            init_std=self.weight_init_std, buffer_device=buffer_device
        )


class MixtralModel(Decoder):
    """
    Mixtral-8x7B transformer with top-2 MoE routing on every layer.

    Args:
        config (MixtralModel.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        n_layers: int = 32
        vocab_size: int = 32000
        sliding_window: int | None = None
        layer: TransformerBlock.Config

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
                    f"Sequence length {seq_len} exceeds original maximum "
                    f"{self.rope.max_seq_len}."
                )
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            assert self.layer.moe is not None
            if self.layer.moe.experts.use_grouped_mm and not has_cuda_capability(9, 0):
                logger.warning(
                    "Failed to use grouped mm, which is only supported "
                    "on SM90 or later",
                )
                self.layer.moe.experts.use_grouped_mm = False

            self.layer.moe.router._debug_force_load_balance = (
                debug.moe_force_load_balance
            )

            if parallelism.expert_parallel_comm_backend == "deepep":
                from torchtitan.models.common.moe.moe_deepep import DeepEPMoE

                init_kwargs = {
                    f.name: getattr(self.layer.moe, f.name)
                    for f in dataclasses.fields(self.layer.moe)
                    if f.init
                }
                self.layer.moe = DeepEPMoE.Config(**init_kwargs)

            if (
                parallelism.context_parallel_degree > 1
                and self.layer.attention.attn_backend == "varlen"
            ):
                raise NotImplementedError(
                    "Context Parallel only supports SDPA and FlexAttention. "
                    f"Got attn_backend='{self.layer.attention.attn_backend}'. "
                    "Varlen attention is not supported with CP."
                )

            if (
                self.sliding_window is not None
                and self.layer.attention.attn_backend != "flex"
            ):
                raise ValueError(
                    "Sliding window attention requires attn_backend='flex'. "
                    f"Got attn_backend='{self.layer.attention.attn_backend}'."
                )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                n_heads = self.layer.attention.n_heads
                # pyrefly: ignore [missing-attribute]
                n_kv_heads = self.layer.attention.n_kv_heads or n_heads
                if n_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide "
                        f"n_heads ({n_heads})."
                    )
                if n_kv_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide "
                        f"n_kv_heads ({n_kv_heads})."
                    )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            assert isinstance(self.layer.attention, GQAttention.Config)
            # pyrefly: ignore [missing-attribute]
            head_dim = self.layer.attention.head_dim or (
                self.dim // self.layer.attention.n_heads
            )
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layer.attention.n_heads,
                2 * head_dim,
                seq_len,
            )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        # pyrefly: ignore [missing-attribute]
        if self.config.sliding_window is None:
            return super().get_attention_masks(input_batch, tokenizer, extra_inputs)

        # Sliding window requires flex attention (validated in update_from_config)
        mask_mods = [
            get_causal_mask_mod(),
            # pyrefly: ignore [missing-attribute, bad-argument-type]
            get_sliding_window_mask_mod(self.config.sliding_window),
        ]

        match self.attn_config.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = input_batch.shape[0]
                assert tokenizer.eos_id is not None
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case _:
                raise ValueError(
                    f"Unknown attention mask type: "
                    f"{self.attn_config.attn_mask_type}"
                )

        return create_attention_mask(
            and_masks(*mask_mods),
            B,
            None,
            input_batch.shape[1],
            input_batch.shape[1],
        )
