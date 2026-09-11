# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel transform."""

from dataclasses import dataclass
from typing import cast

from torchtitan.models.common.attention import BaseAttention, InnerAttention
from torchtitan.models.common.cp_attention import CPInnerAttention
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform

__all__ = ["ContextParallelTransform"]


@dataclass(kw_only=True, slots=True)
class ContextParallelTransform(ModelConfigTransform):
    """Replace inner-attention configs with context-parallel implementations.

    TODO: Replace the config-type mapping with a selector over each attention
    instance, such as its FQN and mask key. Multiple attention instances can
    share one local config type while only some of their masks require K/V
    all-gather; the current mapping assigns all such instances the same CP
    backend.
    """

    inner_attention: dict[type[Module.Config], type[Module]]
    """Map each local config type to its CP backend implementation."""

    def __post_init__(self) -> None:
        for config_type, replacement in self.inner_attention.items():
            if not issubclass(config_type, Module.Config):
                raise ValueError(
                    f"{config_type.__qualname__} must inherit Module.Config."
                )
            if not issubclass(replacement, CPInnerAttention):
                raise ValueError(
                    f"{replacement.__qualname__} must inherit CPInnerAttention."
                )

    def transform(self, model: Module.Config) -> Module.Config:
        for config_type, replacement in self.inner_attention.items():
            for _, traversed, parent, field_name in model.traverse(config_type):
                if issubclass(config_type, InnerAttention.Config) and not (
                    isinstance(parent, BaseAttention.Config)
                    and field_name == "inner_attention"
                ):
                    continue
                converted = convert_config_type(
                    cast(Module.Config, traversed), replacement
                )
                # A matched config can be the root, a list item, or a field.
                if parent is None:
                    model = converted
                elif isinstance(parent, list):
                    assert isinstance(field_name, int)
                    parent[field_name] = converted
                else:
                    assert isinstance(field_name, str)
                    setattr(parent, field_name, converted)
        return model
