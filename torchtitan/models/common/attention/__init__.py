# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping

from .attention import (  # noqa: F401
    __all__ as _attention_all,
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
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from .linear_attention import LinearAttention, LinearAttentionMetadata

AttentionMetadata = (
    FlexAttentionMetadata | VarlenAttentionMetadata | LinearAttentionMetadata
)
AttentionMetadataMap = Mapping[type[InnerAttention], AttentionMetadata]

__all__ = [
    *_attention_all,
    "AttentionMetadata",
    "AttentionMetadataMap",
    "LinearAttention",
    "LinearAttentionMetadata",
]
