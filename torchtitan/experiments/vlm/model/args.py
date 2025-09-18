# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.config.job_config import JobConfig
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

    layer_norm_eps: float = 1e-6
    use_flex_attn: bool = True
    attn_mask_type: str = "causal"


@dataclass
class Llama3Siglip2ModelArgs(Llama3Args):
    encoder: Siglip2ModelArgs = field(default_factory=Siglip2ModelArgs)
    decoder: Llama3Args = field(default_factory=Llama3Args)
    img_token_id: int = 1998

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        super().update_from_config(job_config, **kwargs)
        self.img_token_id = job_config.special_tokens.img_id
