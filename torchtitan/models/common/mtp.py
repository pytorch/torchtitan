# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.protocols.module import Module, ModuleList


def roll_mtp_sequence(
    sequence: torch.Tensor,
    *,
    shift: int,
    positions: torch.Tensor | None = None,
    fill_value: int = 0,
    return_valid_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Left-roll an MTP sequence while preserving packed-document boundaries.

    Without ``positions``, this is a regular left shift with the tail filled.
    With reset-style positions, each document slice is shifted independently:
    ``[A0, A1, A2, B0, B1]`` becomes ``[A1, A2, fill, B1, fill]``.
    """
    if shift < 0:
        raise ValueError(f"MTP roll shift must be non-negative, got {shift}.")
    if shift == 0:
        if return_valid_mask:
            return sequence, torch.ones_like(sequence, dtype=torch.bool)
        return sequence

    seq_len = sequence.shape[1]
    rolled = torch.full_like(sequence, fill_value)
    valid_mask = torch.zeros_like(sequence, dtype=torch.bool)
    if shift >= seq_len:
        return (rolled, valid_mask) if return_valid_mask else rolled

    source = sequence[:, shift:]
    if positions is None:
        rolled[:, : seq_len - shift] = source
        valid_mask[:, : seq_len - shift] = True
        return (rolled, valid_mask) if return_valid_mask else rolled

    if positions.shape[1] < seq_len:
        raise ValueError(
            f"MTP positions need at least {seq_len} tokens, got {positions.shape[1]}."
        )
    valid_tokens = (
        positions[:, shift:seq_len]
        == positions[:, : seq_len - shift] + shift
    )
    rolled[:, : seq_len - shift] = torch.where(
        valid_tokens,
        source,
        torch.full_like(source, fill_value),
    )
    valid_mask[:, : seq_len - shift] = valid_tokens
    return (rolled, valid_mask) if return_valid_mask else rolled


class MTPTransformerBlock(TransformerBlock):
    """Generic MTP block for decoder-only transformer models.

    The block implements the DeepSeek-V3 style fusion:

    ``eh_proj(cat(enorm(shifted_embedding), hnorm(previous_hidden)))``

    followed by one regular transformer block.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        enorm: RMSNorm.Config
        hnorm: RMSNorm.Config
        eh_proj: Linear.Config
        mtp_norm: RMSNorm.Config

    def __init__(
        self,
        config: Config,
        *,
        detach_heads: bool = False,
    ):
        super().__init__()
        self.detach_heads = detach_heads
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.enorm = config.enorm.build()
        self.hnorm = config.hnorm.build()
        self.eh_proj = config.eh_proj.build()
        self.mtp_norm = config.mtp_norm.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        input_offset: torch.Tensor,
        prev_embed: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        if self.detach_heads:
            input_offset = input_offset.detach()
        h = self.eh_proj(
            torch.cat([self.enorm(input_offset), self.hnorm(prev_embed)], dim=-1)
        )
        h = h + self.attention(self.attention_norm(h), attention_masks, positions)
        if self.moe_enabled:
            h = h + self.moe(self.ffn_norm(h))
        else:
            h = h + self.feed_forward(self.ffn_norm(h))
        return self.mtp_norm(h)


class MTPDecoder(Decoder):
    """Decoder variant that owns MTP layers.

    MTP is kept as model behavior: the main decoder consumes the normal input
    sequence, and each MTP layer predicts one extra depth from internally shifted
    token embeddings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        num_mtp_layers: int = 0
        mtp_layer_config: TransformerBlock.Config | None = None
        detach_heads: bool = False

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            MTPDecoder.materialize_mtp_config(self)

    @staticmethod
    def materialize_mtp_config(config: "MTPDecoder.Config") -> None:
        """Materialize MTP layer config before model-specific sharding."""
        if config.num_mtp_layers <= 0:
            return

        inner_cfg = (
            config.mtp_layer_config
            if config.mtp_layer_config is not None
            else config.layers[-1]
        )
        if isinstance(inner_cfg, MTPTransformerBlock.Config):
            return
        config.mtp_layer_config = MTPTransformerBlock.Config(
            attention=inner_cfg.attention,
            feed_forward=inner_cfg.feed_forward,
            moe=inner_cfg.moe,
            attention_norm=inner_cfg.attention_norm,
            ffn_norm=inner_cfg.ffn_norm,
            enorm=RMSNorm.Config(normalized_shape=config.dim),
            hnorm=RMSNorm.Config(normalized_shape=config.dim),
            eh_proj=Linear.Config(
                in_features=config.dim * 2,
                out_features=config.dim,
                bias=False,
            ),
            mtp_norm=RMSNorm.Config(normalized_shape=config.dim),
        )

    def __init__(self, config: Config):
        self.materialize_mtp_config(config)
        super().__init__(config)
        if config.num_mtp_layers <= 0:
            self.mtp_layers = None
            return

        if not isinstance(config.mtp_layer_config, MTPTransformerBlock.Config):
            raise ValueError(
                "MTPDecoder requires Config.mtp_layer_config to be "
                "materialized by MTPDecoder.Config.update_from_config()."
            )
        self.mtp_layers = ModuleList(
            [
                MTPTransformerBlock(
                    config.mtp_layer_config,
                    detach_heads=config.detach_heads,
                )
                for _ in range(config.num_mtp_layers)
            ]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ):
        if self.mtp_layers is None:
            return super().forward(tokens, positions, attention_masks)
        if self.tok_embeddings is None:
            raise ValueError("MTP decoder forward requires token embeddings.")

        # Keep this aligned with Decoder.forward(), but preserve the pre-norm
        # hidden state because MTP consumes the last decoder-layer output.
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(h, attention_masks, positions)

        mtp_prev_embed = h
        h = self.norm(h) if self.norm is not None else h

        mtp_outputs = []
        for depth, layer in enumerate(self.mtp_layers, 1):
            offset_tokens, valid_offset_tokens = roll_mtp_sequence(
                tokens,
                shift=depth,
                positions=positions,
                fill_value=0,
                return_valid_mask=True,
            )
            input_offset = self.tok_embeddings(offset_tokens)
            valid_offset_tokens = valid_offset_tokens.unsqueeze(-1).to(
                dtype=mtp_prev_embed.dtype
            )
            mtp_prev_embed = mtp_prev_embed * valid_offset_tokens
            mtp_prev_embed = layer(
                input_offset,
                mtp_prev_embed,
                attention_masks,
                positions,
            )
            mtp_outputs.append(mtp_prev_embed)

        outputs = [h] + mtp_outputs
        if self._skip_lm_head:
            return outputs
        return [
            self.lm_head(item) if self.lm_head is not None else item
            for item in outputs
        ]
