# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .activation import ActivationFn, SiTUGLU, SwiGLU
from .attention import (
    create_attention_mask,
    create_varlen_metadata_for_document,
    FlexInnerAttention,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
    get_fixed_block_mask_mod,
    get_sliding_window_mask_mod,
    GQAttention,
    InnerAttention,
    QKVLinear,
    ScaledDotProductInnerAttention,
    VarlenInnerAttention,
    VarlenMetadata,
)
from .decoder import Decoder, TransformerBlock
from .embedding import Embedding
from .feed_forward import compute_ffn_hidden_dim, FeedForward, SigmoidGatedFeedForward
from .linear import (
    AllGatherLinear,
    Linear,
    LinearReduceScatter,
    RouterGateLinear,
    ScaledBiasRowwiseLinear,
)
from .moe import MicrobatchWiseLoadBalanceLoss, MoE
from .nn_modules import (
    Conv1d,
    Conv2d,
    GELU,
    GroupNorm,
    Identity,
    LayerNorm,
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
    "SigmoidGatedFeedForward",
    "FlexInnerAttention",
    "QKVLinear",
    "GELU",
    "get_causal_mask_mod",
    "get_document_mask_mod",
    "get_efficient_causal_mask_mod_for_packed_document",
    "get_fixed_block_mask_mod",
    "get_sliding_window_mask_mod",
    "GQAttention",
    "GroupNorm",
    "Identity",
    "InnerAttention",
    "LayerNorm",
    "AllGatherLinear",
    "Linear",
    "LinearReduceScatter",
    "MoE",
    "MicrobatchWiseLoadBalanceLoss",
    "RMSNorm",
    "RoPE",
    "RouterGateLinear",
    "ScaledBiasRowwiseLinear",
    "ScaledDotProductInnerAttention",
    "SiLU",
    "ActivationFn",
    "SiTUGLU",
    "SwiGLU",
    "TransformerBlock",
    "VarlenInnerAttention",
    "VarlenMetadata",
    "compute_ffn_hidden_dim",
]
