# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Gemma-4 Model Implementation
# Based on Google DeepMind's Gemma-4 architecture with hybrid attention

from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)


class Gemma4TransformerBlock(TransformerBlock):
    """
    Gemma-4 TransformerBlock Module with hybrid attention support.
    
    Combines sliding-window attention with global attention (final layer is always global).
    
    Args:
        layer_id (int): Identifier for the layer.
        config (Gemma4TransformerBlock.Config): Block configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # Gemma-4 specific: whether this layer uses global attention
        # (typically only the last layer)
        use_global_attention: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.use_global_attention = config.use_global_attention
        self.attention = config.attention.build()
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        h = x + self.attention(self.attention_norm(x), attention_masks, positions)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Gemma4Model(Decoder):
    """
    Gemma4 Model - Google DeepMind's multimodal language model with hybrid attention.
    
    Key features:
    - Hybrid attention: sliding-window for most layers + global for final layer
    - Supports up to 256K context length
    - Efficient memory usage via KV cache optimizations
    - Available in multiple sizes: 2B, 4B, 12B, 26B (MoE), 31B
    
    Args:
        config (Gemma4Model.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 3584  # Hidden dimension for 12B variant
        vocab_size: int = 262144  # Gemma-4 vocabulary size
        # Sliding window size for local attention (5:1 ratio per spec)
        sliding_window_size: int = 4096
        # Whether to enable sliding window attention
        enable_sliding_window: bool = True

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            from torchtitan.models.gemma4.sharding import set_gemma4_sharding_config

            set_gemma4_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            attention_op_flops = 0
            
            # Gemma-4 uses hybrid attention: sliding window + occasional global
            for layer_idx, layer in enumerate(self.layers):
                attention = layer.attention
                head_dim = (
                    attention.head_dim
                    if attention.head_dim is not None
                    else attention.dim // attention.n_heads
                )
                
                # Use sliding window size for most layers, full seq_len for last layer
                is_last_layer = layer_idx == len(self.layers) - 1
                attn_seq_len = seq_len if is_last_layer else min(seq_len, self.sliding_window_size)
                
                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=head_dim,
                    v_head_dim=head_dim,
                    seq_len=attn_seq_len,
                )
            
            return nparams, 6 * active_nparams + attention_op_flops
