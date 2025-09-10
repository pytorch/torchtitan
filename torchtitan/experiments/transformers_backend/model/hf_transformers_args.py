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
    attn_implementation: str = "sdpa"

    passed_args: dict = field(init=False, repr=False, default_factory=dict)

    def update_from_config(self, job_config: JobConfig):
        
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
        
        self.__dict__.update(self.passed_args)
        
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
        
        # Match torchtitan parameter counts
        self.tie_word_embeddings = False
        self.attention_bias = False
        self.mlp_bias = False

        # Match torchtitan intermediate size calculation
        ffn_hidden_size = 4 * self.hidden_size
        ffn_hidden_size = int(2 * ffn_hidden_size / 3)
        if self.ffn_dim_multiplier is not None:
            ffn_hidden_size = int(self.ffn_dim_multiplier * ffn_hidden_size)
        self.intermediate_size = self.multiple_of * (
            (ffn_hidden_size + self.multiple_of - 1) // self.multiple_of
        )
        # Forced it as HF has config.head_dim and the modeling retrieves it instead of doing config.hidden_size // config.num_attention_heads
        self.head_dim = self.dim // self.num_attention_heads
        
        # n_layers = 2
        # num_hidden_layers = 2

        self.use_cache = False
        return self

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())

        layer_params = {}  # int -> int
        embedding_params = 0
        norm_params = 0
        lm_head_params = 0
        misc_params = {}

        for name, p in model.named_parameters():
            if "model.embed_tokens" in name:
                embedding_params += p.numel()
            elif "model.layers." in name:
                try:
                    layer_num = int(name.split("layers.")[1].split(".")[0])
                    if layer_num not in layer_params:
                        layer_params[layer_num] = 0
                    layer_params[layer_num] += p.numel()
                except (ValueError, IndexError):
                    # Should not happen with standard HF llama names
                    component = "misc_layer_parts"
                    if component not in misc_params:
                        misc_params[component] = 0
                    misc_params[component] += p.numel()
            elif "model.norm" in name:
                norm_params += p.numel()
            elif "lm_head" in name:
                lm_head_params += p.numel()
            else:
                # Catch anything else
                component = name.split(".")[0]
                if component not in misc_params:
                    misc_params[component] = 0
                misc_params[component] += p.numel()

        logger.info("Parameter breakdown:")
        logger.info(f"  - embedding: {embedding_params:,} parameters")
        for layer_num in sorted(layer_params.keys()):
            params = layer_params[layer_num]
            logger.info(f"  - layer_{layer_num}: {params:,} parameters")
        logger.info(f"  - final_norm: {norm_params:,} parameters")
        logger.info(f"  - lm_head: {lm_head_params:,} parameters")
        if misc_params:
            for name, params in misc_params.items():
                logger.info(f"  - {name} (misc): {params:,} parameters")

        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        l, h, q, t = self.n_layers, self.n_heads, self.dim // self.n_heads, seq_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return nparams, num_flops_per_token
