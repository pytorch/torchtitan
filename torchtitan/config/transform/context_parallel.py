# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel transform."""

from dataclasses import dataclass
from typing import cast

from torchtitan.models.common.cp_attention import CPInnerAttention
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform

__all__ = ["ContextParallelTransform"]


@dataclass(kw_only=True, slots=True)
class ContextParallelTransform(ModelConfigTransform):
    """Replace inner-attention configs with context-parallel implementations.

    TODO: Replace config-type keys with an instance-aware selection
    mechanism. Selective gather may require different CP behavior for attention
    instances that share an inner-attention config type but use different masks.
    """

    inner_attention: dict[type[Module.Config], type[Module]]
    """Map each local config type to its CP backend implementation."""

    exclude_fqn_prefixes: tuple[str, ...] = ()
    """Subtrees that keep their local attention, e.g. a vision tower whose
    tokens are not sharded on the cp axis."""

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
            for fqn, traversed, parent, field_name in model.traverse(config_type):
                if any(
                    fqn == prefix or fqn.startswith(prefix + ".")
                    for prefix in self.exclude_fqn_prefixes
                ):
                    continue
                converted = convert_config_type(
                    cast(Module.Config, traversed), replacement
                )
                if parent is None:
                    model = converted
                elif isinstance(parent, list):
                    assert isinstance(field_name, int)
                    parent[field_name] = converted
                else:
                    assert isinstance(field_name, str)
                    setattr(parent, field_name, converted)
        return model
