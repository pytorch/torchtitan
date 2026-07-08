# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.protocols.module import Module, ModuleList


@dataclass(frozen=True)
class MTPSequenceSlices:
    """Aligned main and shifted views for multi-token prediction."""

    main: torch.Tensor
    offsets: tuple[torch.Tensor, ...]

    @property
    def num_offsets(self) -> int:
        return len(self.offsets)


@dataclass(frozen=True)
class MTPContextParallelInput:
    """Main and shifted MTP tensors to shard together for context parallel."""

    main_inputs: torch.Tensor
    main_labels: torch.Tensor
    mtp_inputs: torch.Tensor
    mtp_labels: torch.Tensor


class MTPTransformerBlock(Module):
    """Generic MTP block for decoder-only transformer models.

    The block implements the DeepSeek-V3 style fusion:

    ``eh_proj(cat(enorm(shifted_embedding), hnorm(previous_hidden)))``

    followed by one regular transformer block.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        inner_block_config: TransformerBlock.Config
        enorm: RMSNorm.Config
        hnorm: RMSNorm.Config
        eh_proj: Linear.Config
        final_norm: RMSNorm.Config

        @classmethod
        def from_inner_block_config(
            cls,
            inner_block_config: TransformerBlock.Config,
            *,
            dim: int,
        ) -> "MTPTransformerBlock.Config":
            if isinstance(inner_block_config, cls):
                return inner_block_config
            # Keep the inner transformer block intact and wrap only the MTP
            # fusion/projection modules around it. This lets model-specific
            # block construction stay in the original block config.
            return cls(
                attention=inner_block_config.attention,
                feed_forward=inner_block_config.feed_forward,
                moe=inner_block_config.moe,
                attention_norm=inner_block_config.attention_norm,
                ffn_norm=inner_block_config.ffn_norm,
                inner_block_config=inner_block_config,
                enorm=RMSNorm.Config(normalized_shape=dim),
                hnorm=RMSNorm.Config(normalized_shape=dim),
                eh_proj=Linear.Config(
                    in_features=dim * 2,
                    out_features=dim,
                    bias=False,
                ),
                final_norm=RMSNorm.Config(normalized_shape=dim),
            )

    def __init__(
        self,
        config: Config,
        *,
        detach_heads: bool = False,
    ):
        super().__init__()
        self.detach_heads = detach_heads
        self.enorm = config.enorm.build()
        self.hnorm = config.hnorm.build()
        self.eh_proj = config.eh_proj.build()
        self.inner = config.inner_block_config.build()
        self.final_norm = config.final_norm.build()

    @property
    def moe_enabled(self) -> bool:
        # FSDP treats each item in model.layers as the MoE owner. MTP wraps the
        # real transformer block in ``inner``, so proxy the MoE surface here to
        # keep FSDP's existing MoE branch unchanged.
        return getattr(self.inner, "moe_enabled", False)

    @property
    def moe(self):
        # See ``moe_enabled`` above: this exposes ``inner.moe.experts`` to the
        # shared FSDP expert-sharding logic while still sharding the outer MTP
        # block as one unit.
        return self.inner.moe

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
        h = self.inner(h, attention_masks, positions)
        return self.final_norm(h)


class MTPBlock(Module):
    """Prepared-input MTP block for decoder-only transformer models.

    The dataloader provides ``[B, S + K]`` tokens. ``Decoder`` embeds the K
    shifted token views and this block applies one MTP layer for each depth.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_mtp_layers: int = 0
        loss_scaling_factor: float = 0.3
        inner_block_config: TransformerBlock.Config | None = None
        detach_heads: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.mtp_config = config
        if not isinstance(config.inner_block_config, MTPTransformerBlock.Config):
            raise ValueError(
                "MTPBlock requires MTPBlock.Config.inner_block_config to be "
                "materialized by Decoder.Config.update_from_config()."
            )
        layer_config = config.inner_block_config
        self.layers = ModuleList(
            [
                MTPTransformerBlock(
                    layer_config,
                    detach_heads=config.detach_heads,
                )
                for _ in range(config.num_mtp_layers)
            ]
        )

    @staticmethod
    def mtp_layer_config(
        inner_block_config: TransformerBlock.Config,
        *,
        dim: int,
    ) -> MTPTransformerBlock.Config:
        return MTPTransformerBlock.Config.from_inner_block_config(
            inner_block_config,
            dim=dim,
        )

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_depths(self) -> int:
        return len(self.layers)

    def prepare_main_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return split_mtp_sequence(
            tokens,
            num_mtp_modules=self.num_depths,
        ).main

    def forward(
        self,
        tokens: torch.Tensor,
        h: torch.Tensor,
        tok_embeddings: Module,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        mtp_slices = split_mtp_sequence(
            tokens,
            num_mtp_modules=self.num_depths,
        )
        input_offsets = tuple(
            tok_embeddings(offset_tokens)
            for offset_tokens in mtp_slices.offsets
        )

        outputs = []
        for layer, input_offset in zip(self.layers, input_offsets, strict=True):
            h = layer(input_offset, h, attention_masks, positions)
            outputs.append(h)
        return outputs


def split_mtp_sequence(
    sequence: torch.Tensor,
    *,
    num_mtp_modules: int,
) -> MTPSequenceSlices:
    """Split ``[B, S + K]`` into main ``[B, S]`` and K shifted ``[B, S]`` views."""
    if num_mtp_modules <= 0:
        return MTPSequenceSlices(main=sequence, offsets=())

    seq_len = sequence.shape[1] - num_mtp_modules
    if seq_len <= 0:
        raise ValueError(
            f"Sequence length ({sequence.shape[1]}) must be greater than "
            f"num_mtp_modules ({num_mtp_modules})."
        )
    return MTPSequenceSlices(
        main=sequence[:, :seq_len],
        offsets=tuple(
            sequence[:, offset : offset + seq_len]
            for offset in range(1, num_mtp_modules + 1)
        ),
    )


def trim_mtp_positions(
    positions: torch.Tensor | None,
    *,
    num_mtp_modules: int,
) -> torch.Tensor | None:
    """Return positions for the main sequence when MTP extra tokens are present."""
    if positions is None or num_mtp_modules <= 0:
        return positions
    return positions[:, :-num_mtp_modules]


def split_mtp_context_parallel_input(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_mtp_modules: int,
) -> MTPContextParallelInput:
    """Split extended MTP input/label tensors before CP sharding.

    The dataloader provides ``[B, S + K]`` tensors. For CP, we shard the main
    view and the shifted view together, then append the local shifted tail back
    so each rank still holds ``local_S + K`` tokens for model/loss slicing.
    """
    if num_mtp_modules <= 0:
        raise ValueError("num_mtp_modules must be positive for MTP CP splitting.")
    return MTPContextParallelInput(
        main_inputs=inputs[:, :-num_mtp_modules],
        main_labels=labels[:, :-num_mtp_modules],
        mtp_inputs=inputs[:, num_mtp_modules:],
        mtp_labels=labels[:, num_mtp_modules:],
    )


def merge_mtp_context_parallel_input(
    mtp_input: MTPContextParallelInput,
    *,
    num_mtp_modules: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append the local shifted tail back after CP sharding."""
    if num_mtp_modules <= 0:
        raise ValueError("num_mtp_modules must be positive for MTP CP merging.")
    inputs = torch.cat(
        [
            mtp_input.main_inputs,
            mtp_input.mtp_inputs[:, -num_mtp_modules:],
        ],
        dim=-1,
    )
    labels = torch.cat(
        [
            mtp_input.main_labels,
            mtp_input.mtp_labels[:, -num_mtp_modules:],
        ],
        dim=-1,
    )
    return inputs, labels


def mtp_labels_for_depth(
    labels: torch.Tensor,
    *,
    depth: int,
    seq_len: int,
) -> torch.Tensor:
    """Return labels aligned to one MTP prediction depth."""
    if depth < 0:
        raise ValueError(f"MTP depth must be non-negative, got {depth}.")
    end = depth + seq_len
    if labels.shape[1] < end:
        raise ValueError(
            f"MTP labels need at least {end} tokens for depth {depth}, "
            f"got {labels.shape[1]}."
        )
    return labels[:, depth:end]
