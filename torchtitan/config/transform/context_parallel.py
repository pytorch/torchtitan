# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel transform."""

from collections.abc import Mapping
from dataclasses import dataclass

from torchtitan.models.common.attention import AttentionBackend
from torchtitan.models.common.attention.cp_attention import CPAttention
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform, ModelConfigTransformContext

__all__ = ["ContextParallelTransform"]


@dataclass(kw_only=True, slots=True)
class ContextParallelTransform(ModelConfigTransform):
    """Convert configured attention backends to CP attention backends."""

    inner_attention_map: Mapping[type[AttentionBackend], type[AttentionBackend]]
    """Map each attention backend type to its CP backend type."""

    def __post_init__(self) -> None:
        if not self.inner_attention_map:
            raise ValueError("inner_attention_map must not be empty.")
        for attention_backend, cp_attention_backend in self.inner_attention_map.items():
            if not issubclass(attention_backend, AttentionBackend):
                raise ValueError(
                    f"{attention_backend.__qualname__} must inherit AttentionBackend."
                )
            if not issubclass(cp_attention_backend, CPAttention):
                raise ValueError(
                    f"{cp_attention_backend.__qualname__} must inherit CPAttention."
                )
            if not issubclass(cp_attention_backend, AttentionBackend):
                raise ValueError(
                    f"{cp_attention_backend.__qualname__} must inherit "
                    "AttentionBackend."
                )

    def transform(
        self,
        model: Module.Config,
        *,
        context: ModelConfigTransformContext | None = None,
    ) -> Module.Config:
        del context

        def convert(
            traversed: AttentionBackend.Config,
            parent: object,
            field_name: str | int,
        ) -> None:
            attention_backend = traversed._owner
            assert attention_backend is not None and issubclass(
                attention_backend, AttentionBackend
            )
            if attention_backend not in self.inner_attention_map:
                raise ValueError(
                    "No CP attention backend configured for "
                    f"{attention_backend.__qualname__}."
                )
            converted = convert_config_type(
                traversed,
                self.inner_attention_map[attention_backend],
            )
            assert isinstance(
                field_name, str
            ), "Attention backend configs must be stored in named decoder fields."
            setattr(parent, field_name, converted)

        # Restrict traversal to decoder layers so replicated vision attention
        # is not replaced with a token-sharded CP backend.
        layers = getattr(model, "layers", None)
        if not isinstance(layers, list):
            raise ValueError("ContextParallelTransform requires decoder layers.")

        mtp_layers = getattr(model, "mtp_layers", None) or []
        for layer in layers + mtp_layers:
            for _, traversed, parent, field_name in layer.traverse(
                AttentionBackend.Config
            ):
                assert parent is not None and field_name is not None
                convert(traversed, parent, field_name)
        return model
