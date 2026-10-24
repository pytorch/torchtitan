# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping

from . import attention as _attention, mla as _mla
from .attention import (  # noqa: F401
    Attention,
    create_attention_mask,
    create_varlen_metadata_for_document,
    FlexAttentionMetadata,
    FlexInnerAttention,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
    get_fixed_block_mask_mod,
    get_sliding_window_mask_mod,
    GQAttention,
    InnerAttention,
    local_head_split,
    QKVLinear,
    SlidingWindowFlexInnerAttention,
    SlidingWindowVarlenInnerAttention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from .linear_attention import LinearAttention, LinearAttentionMetadata
from .mla import materialize_mla_kv, MLAAttention  # noqa: F401

AttentionMetadata = (
    FlexAttentionMetadata | VarlenAttentionMetadata | LinearAttentionMetadata
)
AttentionMetadataMap = Mapping[type[InnerAttention], AttentionMetadata]

__all__ = [
    *_attention.__all__,
    *_mla.__all__,
    "AttentionMetadata",
    "AttentionMetadataMap",
    "LinearAttention",
    "LinearAttentionMetadata",
]
