# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Nemotron-3 Nano: Hybrid Mamba-Transformer MoE Model
# Based on NVIDIA's Nemotron-3 architecture with efficient sparse MoE

from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common import Linear
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
)


class NemotronTransformerBlock(TransformerBlock):
    """
    Nemotron-3 Nano TransformerBlock: Hybrid Mamba-Transformer layer.
    
    Alternates between:
    - Mamba-2 blocks (state-space models) for efficient long-range dependencies
    - Transformer blocks (GQA) for local context and multi-head attention
    
    This hybrid design provides:
    - Linear complexity for Mamba blocks
    - Strong local modeling from Transformer blocks
    - Efficient memory usage and high throughput
    
    Args:
        layer_id (int): Layer identifier
        config (NemotronTransformerBlock.Config): Block configuration
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # Nemotron-specific: whether this layer is a Mamba block (True) or Transformer (False)
        is_mamba_block: bool = False
        # For Mamba blocks: state dimension
        mamba_state_dim: int = 16
        mamba_conv_dim: int = 4096
        mamba_input_projection: Linear.Config | None = None
        mamba_output_projection: Linear.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.is_mamba_block = config.is_mamba_block
        self.moe_enabled = not self.is_mamba_block and config.moe is not None
        self.mamba_state_dim = config.mamba_state_dim
        
        if self.is_mamba_block:
            # Mamba block: state-space model
            # In a full implementation, this would use S6 (Selective State Spaces for In-Context Reasoning)
            # For now, we model it as a simplified linear state transition
            self.norm = config.attention_norm.build()
            # Simplified Mamba-like component
            self.input_projection = config.mamba_input_projection.build()
            self.output_projection = config.mamba_output_projection.build()
        else:
            # Transformer block: attention + FFN
            self.attention = config.attention.build()
            self.attention_norm = config.attention_norm.build()
            
            if config.feed_forward is not None:
                self.feed_forward = config.feed_forward.build()
                self.ffn_norm = config.ffn_norm.build()
            elif config.moe is not None:
                self.moe = config.moe.build()
                self.ffn_norm = config.ffn_norm.build()
            else:
                raise ValueError("Either feed_forward or moe must be provided for Transformer blocks")

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        if self.is_mamba_block:
            import torch.nn.functional as F
            # Mamba block forward: norm -> linear proj -> state transition -> output proj
            x_norm = self.norm(x)
            h = F.silu(self.input_projection(x_norm))
            h_out = self.output_projection(h)  # Simplified for now
            return x + h_out
        else:
            # Transformer block forward: attention + FFN/MoE
            h = x + self.attention(self.attention_norm(x), attention_masks, positions)
            
            if hasattr(self, 'feed_forward'):
                out = h + self.feed_forward(self.ffn_norm(h))
            else:  # MoE
                out = h + self.moe(self.ffn_norm(h))
            
            return out

    def reset_parameters(self) -> None:
        pass
class Nemotron3NanoModel(Decoder):
    """
    Nemotron-3 Nano: Hybrid Mamba-Transformer Mixture-of-Experts model.
    
    Key specifications:
    - Architecture: Alternating Mamba-2 + Transformer (GQA) layers
    - Parameters: 31.6B total, 3.2B activated per token (MoE)
    - Experts: 128 total, 6 activated (granular top-k routing)
    - Context: Up to 1 million tokens (1M)
    - Training: 25 trillion tokens with 2-phase curriculum
    
    Performance characteristics:
    - 3.3x higher inference throughput vs GPT-OSS-20B, Qwen3-30B-A3B
    - Superior performance on code, math, reasoning, chat, long-context tasks
    - Supports multi-environment RL post-training
    
    Args:
        config (Nemotron3NanoModel.Config): Model configuration
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096  # Hidden dimension
        vocab_size: int = 262144  # Tokenizer vocab size
        # MoE configuration
        num_experts: int = 128
        top_k_experts: int = 6  # Granular: 6 out of 128 experts
        # Context and position encodings
        max_context_length: int = 1000000  # 1M tokens
        # Mamba configuration
        mamba_state_dim: int = 16
        mamba_conv_dim: int = 4096

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            from torchtitan.models.nemotron_nano.sharding import set_nemotron_sharding_config

            set_nemotron_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            # 31.6B total parameters, 3.2B activated per token
            nparams, _ = get_nparams_and_active_nparams(model)
            # Approximate FLOPs: 6 * active_params + attention/Mamba compute
            active_params = nparams * (self.top_k_experts / self.num_experts)
            return nparams, int(6 * active_params)
