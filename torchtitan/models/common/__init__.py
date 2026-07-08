# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .attention import (
    BaseQKVLinear,
    create_attention_mask,
    create_varlen_metadata_for_document,
    FlexAttention,
    FusedQKVLinear,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
    get_fixed_block_mask_mod,
    get_sliding_window_mask_mod,
    GQAttention,
    QKVLinear,
    ScaledDotProductAttention,
    VarlenAttention,
    VarlenMetadata,
)
from .decoder import Decoder, TransformerBlock
from .embedding import Embedding
from .feed_forward import compute_ffn_hidden_dim, FeedForward
from .moe import MoE
from .mtp import MTPBlock, MTPTransformerBlock
from .nn_modules import (
    Conv1d,
    Conv2d,
    GELU,
    GroupNorm,
    Identity,
    LayerNorm,
    Linear,
    RMSNorm,
    SiLU,
)
from .rope import ComplexRoPE, CosSinRoPE, RoPE

__all__ = [
    "Conv1d",
    "Conv2d",
    "ComplexRoPE",
    "CosSinRoPE",
    "create_attention_mask",
    "create_varlen_metadata_for_document",
    "Decoder",
    "Embedding",
    "FeedForward",
    "FlexAttention",
    "BaseQKVLinear",
    "FusedQKVLinear",
    "GELU",
    "get_causal_mask_mod",
    "get_document_mask_mod",
    "get_efficient_causal_mask_mod_for_packed_document",
    "get_fixed_block_mask_mod",
    "get_sliding_window_mask_mod",
    "GQAttention",
    "GroupNorm",
    "Identity",
    "LayerNorm",
    "Linear",
    "MoE",
    "MTPBlock",
    "MTPTransformerBlock",
    "QKVLinear",
    "RMSNorm",
    "RoPE",
    "ScaledDotProductAttention",
    "SiLU",
    "TransformerBlock",
    "VarlenAttention",
    "VarlenMetadata",
    "compute_ffn_hidden_dim",
]
