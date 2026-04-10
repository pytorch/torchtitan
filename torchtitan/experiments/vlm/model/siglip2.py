# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import einops as E
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.common.attention import (
    AttentionMasksType,
    create_attention_mask,
    FlexAttention,
    get_causal_mask_mod,
    get_document_mask_mod,
)
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module, ModuleDict

LayerNorm = Module.from_nn_module(nn.LayerNorm)


def resize_positional_embeddings(
    pos_embs_HWD: torch.Tensor,
    spatial_shapes_N2: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """
    Resize the learned 2D positional embeddings to image-specific size and pad to a fixed size.

    Args:
        pos_embs_HWD (`torch.Tensor`):
            Position embeddings of shape (height, width, embed_dim)
        spatial_shapes (`torch.LongTensor`):
            Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
        max_length (`int`):
            Maximum length of the positional embeddings to pad resized positional embeddings to

    Returns:
        `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
    """
    _, _, D = pos_embs_HWD.shape
    B, _ = spatial_shapes_N2.shape

    resized_embs_BLD = torch.empty(
        (B, max_length, D),
        device=pos_embs_HWD.device,
        dtype=pos_embs_HWD.dtype,
    )

    # TODO: group images by size, and do interpolate,
    # or cache the interpolate output so we do this once per size
    for i in range(B):
        height, width = spatial_shapes_N2[i].tolist()
        if (height + width) == 0:  # Skip empty padding images
            continue

        resized_emb = F.interpolate(
            E.rearrange(pos_embs_HWD, "h w d -> 1 d h w"),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        resized_emb_LD = E.rearrange(resized_emb, "1 d h w -> (h w) d")
        resized_embs_BLD[i, : int(height * width)] = resized_emb_LD

    return resized_embs_BLD


class VisionEmbeddings(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        patch_embedding: Linear.Config
        position_embedding: Embedding.Config
        n_pos_embs: int

    def __init__(self, config: Config):
        super().__init__()
        self.patch_embedding = config.patch_embedding.build()
        self.position_embedding = config.position_embedding.build()
        self.n_pos_embs = config.n_pos_embs

    def forward(self, pixels_NLD: torch.Tensor, grid_hw: torch.Tensor) -> torch.Tensor:
        # Apply patch embeddings to already patchified pixel values
        patch_embeds_NLD = self.patch_embedding(pixels_NLD)

        # Get positional resized and padded positional embeddings
        pos_emb_HWD = self.position_embedding.weight.reshape(
            self.n_pos_embs, self.n_pos_embs, -1
        )
        spatial_h = E.reduce(grid_hw[:, :, 0], "n l -> n", reduction="max") + 1
        spatial_w = E.reduce(grid_hw[:, :, 1], "n l -> n", reduction="max") + 1
        spatial_shapes = torch.stack([spatial_h, spatial_w], dim=-1).long()
        resized_positional_embeddings = resize_positional_embeddings(
            pos_emb_HWD,
            spatial_shapes,
            max_length=pixels_NLD.shape[1],
        )
        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds_NLD + resized_positional_embeddings
        return embeddings


class Attention(Module):
    """Multi-head attention module for vision transformer."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        qkv_proj: Linear.Config
        out_proj: Linear.Config
        n_heads: int
        dim: int

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads

        self.q_proj = config.qkv_proj.build()
        self.k_proj = config.qkv_proj.build()
        self.v_proj = config.qkv_proj.build()
        self.out_proj = config.out_proj.build()

        self.inner_attention = FlexAttention.Config().build()

    def forward(self, x: torch.Tensor, attention_masks: AttentionMasksType):
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Use self.head_dim instead of `n_heads` to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = E.rearrange(xq, "b l (h d) -> b l h d", d=self.head_dim)
        xk = E.rearrange(xk, "b l (h d) -> b l h d", d=self.head_dim)
        xv = E.rearrange(xv, "b l (h d) -> b l h d", d=self.head_dim)

        assert isinstance(attention_masks, BlockMask)
        output = self.inner_attention(xq, xk, xv, attention_masks=attention_masks)
        output = E.rearrange(output, "b l h d -> b l (h d)").contiguous()

        return self.out_proj(output)


class FeedForward(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        fc1: Linear.Config
        fc2: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = config.fc1.build()
        self.fc2 = config.fc2.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


class TransformerLayer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        self_attn: Attention.Config
        mlp: FeedForward.Config
        layer_norm_eps: float = 1e-6
        dim: int

    def __init__(self, config: Config):
        super().__init__()
        self.layer_norm1 = LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.self_attn = Attention(config.self_attn)
        self.layer_norm2 = LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config.mlp)

    def forward(
        self, x: torch.Tensor, attention_masks: AttentionMasksType
    ) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x), attention_masks)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class VisionTransformer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        embeddings: VisionEmbeddings.Config
        layers: list[TransformerLayer.Config]
        n_channels: int = 3
        patch_size: int = 16
        layer_norm_eps: float = 1e-6
        attn_mask_type: str = "causal"

    def __init__(self, config: Config):
        super().__init__()
        self.attn_mask_type = config.attn_mask_type
        self.eos_id = 11

        self.embeddings = VisionEmbeddings(config.embeddings)

        self.layers = ModuleDict()
        for i, layer_config in enumerate(config.layers):
            self.layers[str(i)] = TransformerLayer(layer_config)

        self.post_layernorm = LayerNorm(config.dim, eps=config.layer_norm_eps)

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:

        # TODO: this is duplicated in the main model forward.
        # TODO: is this really required? Can we call this `get_attention_masks`
        # inside the main model forward? At that time PP should already split the
        # grid_thw correctly.
        grid_hw = extra_inputs["grid_thw"][:, :, 1:]  # Siglip2 only support image hw
        pixel_masks = E.reduce(grid_hw != -1, "n l hw -> n l", reduction="all")

        mask_mods = [get_causal_mask_mod()]
        match self.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = pixel_masks.shape[0]
                mask_mods.append(get_document_mask_mod(pixel_masks, tokenizer.eos_id))
            case _:
                raise ValueError(f"Unknown attention mask type: {self.attn_mask_type}")
        return create_attention_mask(
            and_masks(*mask_mods), B, None, pixel_masks.shape[1], pixel_masks.shape[1]
        )

    def forward(
        self,
        pixel_values_NLD: torch.FloatTensor,
        pixel_masks_NL: torch.BoolTensor,
        grid_hw: torch.LongTensor,
        attention_masks: AttentionMasksType,
    ):
        h = self.embeddings(pixel_values_NLD, grid_hw)

        for layer in self.layers.values():
            h = layer(h, attention_masks)
        h = self.post_layernorm(h)

        return h
