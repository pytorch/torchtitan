# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torch import nn
from torchtitan.config import JobConfig
from torchtitan.protocols.model import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class LFM2ModelArgs(BaseModelArgs):
    """Model arguments for LFM2 (Liquid Foundation Model 2).

    LFM2 is a hybrid model combining:
    - Short-range convolutions (LIV - Linear Input-Varying)
    - Grouped Query Attention (GQA)
    - SwiGLU activation functions
    - RMSNorm normalization
    """

    # Architecture parameters
    vocab_size: int = 32768
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_conv_blocks: int = 10
    num_attention_blocks: int = 6
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    conv_kernel_size: int = 3
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0

    # Training parameters
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    initializer_range: float = 0.02

    # Token IDs
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    eos_id: int = 2  # For compatibility with TorchTitan

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        """Update model args from training config."""
        seq_len = job_config.training.seq_len
        if seq_len > self.max_position_embeddings:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_position_embeddings}."
            )
        self.max_position_embeddings = seq_len

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        """Calculate model parameters and FLOPs.

        For LFM2, we calculate FLOPs based on the hybrid architecture:
        - Conv blocks contribute differently than attention blocks
        - Only attention blocks use the head_dims formula
        """
        import torch.nn as nn

        # Count total parameters
        nparams = sum(p.numel() for p in model.parameters())

        # Count embedding parameters
        nparams_embedding = 0
        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            # LFM2Model wraps LFM2ForCausalLM which wraps LFM2Model
            inner_model = model.model.model
            if hasattr(inner_model, 'embed_tokens'):
                nparams_embedding = sum(p.numel() for p in inner_model.embed_tokens.parameters())

        # Calculate FLOPs
        # For simplicity, we use a similar formula to dense models but account for
        # the fact that only num_attention_blocks use attention
        head_dim = self.hidden_size // self.num_attention_heads

        # FLOPs from non-embedding parameters (6x for forward+backward)
        num_flops_per_token = 6 * (nparams - nparams_embedding)

        # Additional FLOPs from attention blocks (using the head_dims formula)
        # Only the attention blocks contribute to this
        num_flops_per_token += (
            6 * self.num_attention_blocks * self.num_attention_heads * (2 * head_dim) * seq_len
        )

        # Subtract embedding parameters if weight tying is enabled
        if self.tie_word_embeddings:
            nparams = nparams - nparams_embedding

        return nparams, num_flops_per_token
