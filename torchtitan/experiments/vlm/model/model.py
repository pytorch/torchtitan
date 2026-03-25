# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import einops as E
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.linear import Linear
from torchtitan.models.llama3 import Llama3Model as Llama3
from torchtitan.protocols.module import Module

from .args import Siglip2Config, SpecialTokens
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

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        linear_config = Linear.Config(bias=True)
        self.w1 = linear_config.build(in_features=in_dim, out_features=in_dim)
        self.w2 = linear_config.build(in_features=in_dim, out_features=out_dim)

    def forward(self, x_NLD: torch.Tensor):
        x_NLD = self.w1(x_NLD)
        x_NLD = nn.functional.silu(x_NLD)
        x_NLD = self.w2(x_NLD)
        return x_NLD


def _set_param_init_recursive(module: nn.Module) -> None:
    """Set param_init on all Module descendants that have Linear or norm children.

    Walks the module tree and assigns xavier_uniform for weights, zeros for biases,
    and ones for norm weights. This is used for experiment modules that don't use
    the Config.expand() system.
    """
    for child in module.modules():
        if isinstance(child, Module) and child._param_init is None:
            has_params = any(True for _ in child.parameters(recurse=False))
            if has_params:
                # Check if this is a norm-like module
                names = [n for n, _ in child.named_parameters(recurse=False)]
                if names == ["weight"] and hasattr(child, "normalized_shape"):
                    child._param_init = {"weight": nn.init.ones_}
                else:
                    child._param_init = {
                        "weight": nn.init.xavier_uniform_,
                        "bias": nn.init.zeros_,
                    }


class Llama3Siglip2Transformer(Llama3):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3.Config):
        encoder: Siglip2Config = field(default_factory=Siglip2Config)

    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        self.encoder = VisionTransformer(config.encoder)
        self.projector = Projector(in_dim=config.encoder.dim, out_dim=config.dim)

        # Set param_init on encoder and projector sub-modules.
        # These are experiment-level modules that don't use Config.expand().
        _set_param_init_recursive(self.encoder)
        _set_param_init_recursive(self.projector)

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
            h_BSD = layer(h_BSD, self.freqs_cis, attention_masks["llama3_masks"])

        h_BSD = self.norm(h_BSD) if self.norm else h_BSD
        output = self.output(h_BSD) if self.output else h_BSD
        return output
