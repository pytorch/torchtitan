# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel transform."""

from dataclasses import dataclass
from typing import cast

from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.cp_attention import CPInnerAttention
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform

__all__ = ["ContextParallelTransform", "KDAContextParallelTransform"]


@dataclass(kw_only=True, slots=True)
class ContextParallelTransform(ModelConfigTransform):
    """Run attention under context parallelism.

    Replace every inner attention with ``inner_attention`` while preserving its
    config.

    TODO(fegin): support one kernel per attention type, for models that mix
    them.
    """

    inner_attention: type[Module]
    """Replacement inner attention; must inherit ``CPInnerAttention``."""

    def __post_init__(self) -> None:
        if not issubclass(self.inner_attention, CPInnerAttention):
            raise ValueError(
                f"{self.inner_attention.__qualname__} must inherit CPInnerAttention."
            )

    def transform(self, model: Module.Config) -> Module.Config:
        for _, traversed, _, _ in model.traverse(BaseAttention.Config):
            # traverse returns the base config type.
            attention = cast(BaseAttention.Config, traversed)
            attention.inner_attention = convert_config_type(
                attention.inner_attention, self.inner_attention
            )
        return model


@dataclass(kw_only=True, slots=True)
class KDAContextParallelTransform(ModelConfigTransform):
    """Install the context-parallel inner KDA implementation."""

    def transform(self, model: Module.Config) -> Module.Config:
        from torchtitan.models.kimi_k3.cp_kda import ContextParallelInnerKDA
        from torchtitan.models.kimi_k3.kda import KDA

        for _, traversed, _, _ in model.traverse(KDA.Config):
            kda = cast(KDA.Config, traversed)
            kda.inner_kda = convert_config_type(kda.inner_kda, ContextParallelInnerKDA)
        return model
