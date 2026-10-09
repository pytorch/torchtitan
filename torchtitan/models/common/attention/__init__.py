# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping

from torchtitan.protocols.module import Module

from . import attention as _attention, mla as _mla
from .attention import (  # noqa: F401
    AttentionBackend,
    BaseAttention,
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
    ScaledDotProductInnerAttention,
    SlidingWindowFlexInnerAttention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from .kda import LinearAttentionMetadata
from .mla import (  # noqa: F401
    materialize_mla_kv,
    MLAAttention,
    register_mla_wkv_b_state_dict_hooks,
)

AttentionMetadata = (
    FlexAttentionMetadata | VarlenAttentionMetadata | LinearAttentionMetadata
)
AttentionMetadataMap = Mapping[type[Module], AttentionMetadata]

__all__ = [
    *_attention.__all__,
    *_mla.__all__,
    "AttentionMetadata",
    "AttentionMetadataMap",
    "LinearAttentionMetadata",
]
