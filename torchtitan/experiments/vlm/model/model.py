# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import einops as E
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.linear import Linear
from torchtitan.models.llama3 import Llama3Model as Llama3
from torchtitan.protocols.module import Module

from .args import SpecialTokens
from .siglip2 import VisionTransformer


def _scatter_img_tokens(h_BSD, tokens_BS, i_NLD, i_mask_NL, img_id):
    B, S, D = h_BSD.shape
    # Where are the image tokens in LLM input, make broadcastable with h_BSD
    img_mask_h_BSD = E.repeat(tokens_BS == img_id, "b s -> b s 1")
    # Only get valid (non-padded) tokens, result are flatten
    i_flatten = torch.masked_select(i_NLD, mask=i_mask_NL.unsqueeze(-1))

    assert i_flatten.numel() // D == img_mask_h_BSD.sum(), (
        f"Different number of visual embeddings {i_flatten.numel() // D} "
        f"with placeholder in input token embeddings {img_mask_h_BSD.sum()}"
    )
    h_BSD.masked_scatter_(mask=img_mask_h_BSD, source=i_flatten)
    return h_BSD


class Projector(Module):
    """Project the Encoder embedding to the LLM embedding."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w1: Linear.Config
        w2: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.w1 = config.w1.build()
        self.w2 = config.w2.build()

    def forward(self, x_NLD: torch.Tensor):
        x_NLD = self.w1(x_NLD)
        x_NLD = nn.functional.silu(x_NLD)
        x_NLD = self.w2(x_NLD)
        return x_NLD


class Llama3Siglip2Transformer(Llama3):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3.Config):
        encoder: VisionTransformer.Config
        projector: Projector.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        self.encoder = VisionTransformer(config.encoder)
        self.projector = Projector(config.projector)

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        masks = super().get_attention_masks(input_batch, tokenizer, extra_inputs)
        assert isinstance(masks, BlockMask)
        if self.encoder is not None:
            encoder_masks = self.encoder.get_attention_masks(
                input_batch, tokenizer, extra_inputs
            )
            assert isinstance(encoder_masks, BlockMask)
        return {"llama3_masks": masks, "encoder_masks": encoder_masks}

    def forward(
        self,
        tokens: torch.Tensor,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        special_tokens: SpecialTokens,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h_BSD = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        if self.encoder is not None:
            assert (
                attention_masks is not None
            ), "encoder only allows FlexAttention, so the llama3 must use FlexAttention as well."
            grid_hw = grid_thw[:, :, 1:]  # Siglip2 only support image hw
            pixel_masks = E.reduce(grid_hw != -1, "n l hw -> n l", reduction="all")
            i_NLD = self.encoder(
                pixel_values, pixel_masks, grid_hw, attention_masks["encoder_masks"]
            )
            i_NLD = self.projector(i_NLD)
            h_BSD = _scatter_img_tokens(
                h_BSD, tokens, i_NLD, pixel_masks, special_tokens.img_id
            )

        for layer in self.layers.values():
            h_BSD = layer(
                h_BSD, self.freqs_cis, attention_masks["llama3_masks"], positions
            )

        h_BSD = self.norm(h_BSD) if self.norm else h_BSD
        output = self.output(h_BSD) if self.output else h_BSD
        return output
