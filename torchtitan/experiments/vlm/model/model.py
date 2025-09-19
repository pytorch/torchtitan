# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import einops as E
import torch
from torch import nn

from torchtitan.models.llama3 import Transformer as Llama3

from ..datasets.mm_datasets import SpecialTokens

from .args import Llama3Siglip2ModelArgs
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


class Projector(nn.Module):
    """Project the Encoder embedding to the LLM embedding."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, in_dim)
        self.w2 = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def forward(self, x_NLD: torch.Tensor):
        x_NLD = self.w1(x_NLD)
        x_NLD = nn.functional.silu(x_NLD)
        x_NLD = self.w2(x_NLD)
        return x_NLD

    def init_weights(self):
        nn.init.xavier_uniform_(self.w1.weight)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        nn.init.xavier_uniform_(self.w2.weight)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)


class Llama3Siglip2Transformer(Llama3):
    def __init__(self, model_args: Llama3Siglip2ModelArgs):
        super().__init__(model_args)
        self.model_args = model_args
        self.encoder = VisionTransformer(model_args.encoder)
        self.projector = Projector(
            in_dim=model_args.encoder.dim, out_dim=model_args.dim
        )

    def init_weights(self, buffer_device=None):
        super().init_weights(buffer_device=buffer_device)
        if self.encoder is not None:
            self.encoder.init_weights()
        if self.projector is not None:
            self.projector.init_weights()

    def forward(
        self,
        tokens: torch.Tensor,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        special_tokens: SpecialTokens,
        input_batch: torch.Tensor | None = None,
    ):
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h_BSD = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        if self.encoder is not None:
            grid_hw = grid_thw[:, :, 1:]  # Siglip2 only support image hw
            pixel_masks = E.reduce(grid_hw != -1, "n l hw -> n l", reduction="all")
            i_NLD = self.encoder(pixel_values, pixel_masks, grid_hw)
            i_NLD = self.projector(i_NLD)
            h_BSD = _scatter_img_tokens(
                h_BSD, tokens, i_NLD, pixel_masks, special_tokens.img_id
            )

        for layer in self.layers.values():
            h_BSD = layer(h_BSD, self.freqs_cis)

        h_BSD = self.norm(h_BSD) if self.norm else h_BSD
        output = self.output(h_BSD) if self.output else h_BSD
        return output
