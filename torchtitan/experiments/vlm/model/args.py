# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.components.tokenizer import HuggingFaceTokenizer


@dataclass
class SpecialTokens:
    img_token: str
    img_id: int
    boi_token: str
    boi_id: int
    eoi_token: str
    eoi_id: int
    pad_token: str
    pad_id: int
    ignore_id: int = -100  # Pytorch F.cross_entropy default

    @classmethod
    def from_tokenizer(cls, tokenizer: HuggingFaceTokenizer):
        SPECIAL_TOKENS_MAP = {
            "img": "<|image|>",
            "boi": "<|begin_of_image|>",
            "eoi": "<|end_of_image|>",
            "pad": "<|pad|>",
        }
        added_tokens = tokenizer.tokenizer.get_added_tokens_decoder()
        token_to_id = {tok.content: tok_id for tok_id, tok in added_tokens.items()}
        special_tokens_dict = {}
        for prefix, tok in SPECIAL_TOKENS_MAP.items():
            special_tokens_dict[f"{prefix}_token"] = tok
            special_tokens_dict[f"{prefix}_id"] = token_to_id[tok]
        return cls(**special_tokens_dict)


@dataclass
class Siglip2Config:
    dim: int = 768
    ffn_dim: int = 3072
    n_layers: int = 12
    n_heads: int = 12

    n_pos_embs: int = 16  # Number of positional embeddings per h&w
    n_channels: int = 3  # RGB channels
    patch_size: int = 16
    spatial_merge_size: int = 1

    layer_norm_eps: float = 1e-6
    attn_backend: str = "flex"
    attn_mask_type: str = "causal"
