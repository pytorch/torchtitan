# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional, Union
import os

from torch import nn
from torchtitan.config import JobConfig
from torchtitan.protocols import BaseModelArgs
from torchtitan.tools.logging import logger
from transformers.models.llama.configuration_llama import LlamaConfig


@dataclass
class HFTransformerModelArgs(LlamaConfig, BaseModelArgs):
    # Torchtitan naming
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 128256
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    
    max_seq_len: int = 2048
    depth_init: bool = True
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 0
    
    # HF args
    attn_implementation: str = "eager"

    def update_from_config(self, job_config: JobConfig):
        
        #TODO(3outeille): clean this mess once grad norm is stabilized
        default_args = HFTransformerModelArgs()

        args_to_override = {}
        for key in default_args.__dict__:
            if hasattr(self, key):
                current_value = getattr(self, key)
                default_value = getattr(default_args, key)
                if current_value != default_value:
                    args_to_override[key] = current_value

        hf_model_config = LlamaConfig.from_pretrained(
            job_config.model.name,
            attn_implementation=self.attn_implementation,
        )
        # n_layers = 32
        self.__dict__.update(hf_model_config.__dict__)

        # num_hidden_layers = 16

        # Update TT args with HF args (for keys that exist in both but differ in namings)
        self.dim = self.hidden_size
        self.n_layers = self.num_hidden_layers
        self.n_heads = self.num_attention_heads
        self.n_kv_heads = self.num_key_value_heads
        self.norm_eps = self.rms_norm_eps
        self.max_seq_len = self.max_position_embeddings
        self.eos_id = self.eos_token_id

        # n_layers = 16
        
        self.__dict__.update(args_to_override)
        
        # n_layers = 2
        # num_hidden_layers = 16

        # Update HF args with TT override args because HF modeling uses HF args and not TT args
        # TODO(3outeille): find a cleaner way to handle the mapping
        self.hidden_size = self.dim
        self.num_hidden_layers = self.n_layers
        self.num_attention_heads = self.n_heads
        self.num_key_value_heads = self.n_kv_heads
        self.rms_norm_eps = self.norm_eps
        self.max_position_embeddings = self.max_seq_len
        self.eos_token_id = self.eos_id
        
        # n_layers = 2
        # num_hidden_layers = 2

        print(self)
        self.use_cache = False
        return self

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        l, h, q, t = self.n_layers, self.n_heads, self.dim // self.n_heads, seq_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return nparams, num_flops_per_token
