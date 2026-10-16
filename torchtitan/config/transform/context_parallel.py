# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel transform."""

from collections.abc import Mapping
from dataclasses import dataclass, replace

import spmd_types as spmd

from torchtitan.models.base.module import Module

from torchtitan.models.common.attention import Attention, InnerAttention
from torchtitan.models.common.attention.cp_attention import CPInnerAttention
from torchtitan.models.common.attention.cp_mla import (
    AllGatherKVMLAFlexAttention,
    UlyssesMLAAttention,
)
from torchtitan.models.common.decoder_sharding import mla_kv_projection_sharding_config

from .base import convert_config_type, ModelConfigTransform

__all__ = ["ContextParallelTransform"]


@dataclass(kw_only=True, slots=True)
class ContextParallelTransform(ModelConfigTransform):
    """Convert configured inner-attention types to CP inner-attention types."""

    inner_attention_map: Mapping[type[InnerAttention], type[InnerAttention]]
    """Map each inner-attention type to its CP inner-attention type."""

    def __post_init__(self) -> None:
        if not self.inner_attention_map:
            raise ValueError("inner_attention_map must not be empty.")
        for inner_attention, cp_inner_attention in self.inner_attention_map.items():
            if not issubclass(inner_attention, InnerAttention):
                raise ValueError(
                    f"{inner_attention.__qualname__} must inherit InnerAttention."
                )
            if not issubclass(cp_inner_attention, CPInnerAttention):
                raise ValueError(
                    f"{cp_inner_attention.__qualname__} must inherit CPInnerAttention."
                )

    def transform(self, model: Module.Config) -> Module.Config:
        # Restrict traversal to decoder layers so replicated vision attention
        # is not replaced with a token-sharded CP backend.
        layers = getattr(model, "layers", None)
        if not isinstance(layers, list):
            raise ValueError("ContextParallelTransform requires decoder layers.")

        mtp_layers = getattr(model, "mtp_layers", None) or []
        for layer in layers + mtp_layers:
            for _, attention, _, _ in layer.traverse(Attention.Config):
                existing = attention.inner_attention
                inner_attention = existing._owner
                assert inner_attention is not None and issubclass(
                    inner_attention, InnerAttention
                )
                if inner_attention not in self.inner_attention_map:
                    raise ValueError(
                        "No CP inner attention configured for "
                        f"{inner_attention.__qualname__}."
                    )
                converted = convert_config_type(
                    existing, self.inner_attention_map[inner_attention]
                )
                if isinstance(
                    converted,
                    (AllGatherKVMLAFlexAttention.Config, UlyssesMLAAttention.Config),
                ):
                    # All-gather precedes the projection; Ulysses all-to-all follows it.
                    cp = (
                        spmd.R
                        if isinstance(converted, AllGatherKVMLAFlexAttention.Config)
                        else spmd.S(0)
                    )
                    converted = replace(
                        converted,
                        wkv_b=replace(
                            converted.wkv_b,
                            sharding_config=mla_kv_projection_sharding_config(cp=cp),
                        ),
                    )
                attention.inner_attention = converted
        return model
