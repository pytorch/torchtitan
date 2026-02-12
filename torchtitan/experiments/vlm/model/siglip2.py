# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import einops as E
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.common import trunc_normal_
from torchtitan.models.common.attention import (
    AttentionMasksType,
    create_attention_mask,
    FlexAttentionWrapper,
    get_causal_mask_mod,
    get_document_mask_mod,
)

from .args import Siglip2Config


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


class VisionEmbeddings(nn.Module):
    def __init__(self, args: Siglip2Config):
        super().__init__()
        self.patch_embedding = nn.Linear(
            in_features=args.n_channels * args.patch_size * args.patch_size,
            out_features=args.dim,
        )
        self.position_embedding = nn.Embedding(args.n_pos_embs**2, args.dim)
        self.n_pos_embs = args.n_pos_embs

    def init_weights(self):
        trunc_normal_(self.patch_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight)

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


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (Transformer.Config): Model configuration arguments.

    Attributes:
        n_heads (int): Number of query heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, args: Siglip2Config):
        super().__init__()
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)

        self.inner_attention = FlexAttentionWrapper()

    def forward(self, x: torch.Tensor, attention_masks: AttentionMasksType):
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Use self.head_dim instead of `n_heads` to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = E.rearrange(xq, "b l (h d) -> b h l d", d=self.head_dim)
        xk = E.rearrange(xk, "b l (h d) -> b h l d", d=self.head_dim)
        xv = E.rearrange(xv, "b l (h d) -> b h l d", d=self.head_dim)

        assert isinstance(attention_masks, BlockMask)
        output = self.inner_attention(xq, xk, xv, block_mask=attention_masks)
        output = E.rearrange(output, "b h l d -> b l (h d)").contiguous()

        return self.out_proj(output)

    def init_weights(self):
        for linear in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            trunc_normal_(linear.weight, mean=0.0, std=0.02)


class FeedForward(nn.Module):
    def __init__(self, args: Siglip2Config):
        super().__init__()
        self.fc1 = nn.Linear(args.dim, args.ffn_dim)
        self.fc2 = nn.Linear(args.ffn_dim, args.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x

    def init_weights(self):
        trunc_normal_(self.fc1.weight, mean=0.0, std=0.02)
        trunc_normal_(self.fc2.weight, mean=0.0, std=0.02)


class TransformerLayer(nn.Module):
    def __init__(self, args: Siglip2Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(args.dim, eps=args.layer_norm_eps)
        self.self_attn = Attention(args)
        self.layer_norm2 = nn.LayerNorm(args.dim, eps=args.layer_norm_eps)
        self.mlp = FeedForward(args)

    def forward(
        self, x: torch.Tensor, attention_masks: AttentionMasksType
    ) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x), attention_masks)
        x = x + self.mlp(self.layer_norm2(x))
        return x

    def init_weights(self):
        self.layer_norm1.reset_parameters()
        self.layer_norm2.reset_parameters()
        self.self_attn.init_weights()
        self.mlp.init_weights()


class VisionTransformer(nn.Module):
    def __init__(self, args: Siglip2Config):
        super().__init__()
        self.args = args
        self.eos_id = 11

        self.embeddings = VisionEmbeddings(args)
        self.layers = nn.ModuleDict(
            {str(idx): TransformerLayer(args) for idx in range(args.n_layers)}
        )
        self.post_layernorm = nn.LayerNorm(args.dim, eps=args.layer_norm_eps)

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
        match self.args.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = pixel_masks.shape[0]
                mask_mods.append(get_document_mask_mod(pixel_masks, tokenizer.eos_id))
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.args.attn_mask_type}"
                )
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

    def init_weights(self):
        self.embeddings.init_weights()
        for layer in self.layers.values():
            layer.init_weights()
        self.post_layernorm.reset_parameters()
