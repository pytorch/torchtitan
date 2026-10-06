# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel transform."""

from collections.abc import Mapping
from dataclasses import dataclass

from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.common.attention.cp_attention import CPInnerAttention
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform, ModelConfigTransformContext

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
                    f"{cp_inner_attention.__qualname__} must inherit "
                    "CPInnerAttention."
                )

    def transform(
        self,
        model: Module.Config,
        *,
        context: ModelConfigTransformContext | None = None,
    ) -> Module.Config:
        del context

        def convert(
            traversed: InnerAttention.Config,
            parent: object,
            field_name: str | int,
        ) -> None:
            inner_attention = traversed._owner
            assert inner_attention is not None and issubclass(
                inner_attention, InnerAttention
            )
            if inner_attention not in self.inner_attention_map:
                raise ValueError(
                    "No CP inner attention configured for "
                    f"{inner_attention.__qualname__}."
                )
            converted = convert_config_type(
                traversed,
                self.inner_attention_map[inner_attention],
            )
            assert isinstance(
                field_name, str
            ), "Inner-attention configs must be stored in named decoder fields."
            setattr(parent, field_name, converted)

        # Restrict traversal to decoder layers so replicated vision attention
        # is not replaced with a token-sharded CP backend.
        layers = getattr(model, "layers", None)
        if not isinstance(layers, list):
            raise ValueError("ContextParallelTransform requires decoder layers.")

        mtp_layers = getattr(model, "mtp_layers", None) or []
        for layer in layers + mtp_layers:
            for _, traversed, parent, field_name in layer.traverse(
                InnerAttention.Config
            ):
                assert parent is not None and field_name is not None
                convert(traversed, parent, field_name)
        return model
