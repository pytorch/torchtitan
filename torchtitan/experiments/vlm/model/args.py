# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.models.llama3 import TransformerModelArgs as Llama3Args


@dataclass
class Siglip2ModelArgs:
    dim: int = 768
    ffn_dim: int = 3072
    n_layers: int = 12
    n_heads: int = 12

    n_pos_embs: int = 16  # Number of positional embeddings per h&w
    n_channels: int = 3  # RGB channels
    patch_size: int = 16
    spatial_merge_size: int = 1

    layer_norm_eps: float = 1e-6
    use_flex_attn: bool = True
    attn_mask_type: str = "causal"


@dataclass
class Llama3Siglip2ModelArgs(Llama3Args):
    encoder: Siglip2ModelArgs = field(default_factory=Siglip2ModelArgs)
