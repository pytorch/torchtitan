# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel transform."""

from dataclasses import dataclass, field
from typing import cast

from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.cp_attention import ContextParallelKernel
from torchtitan.protocols.module import Module

from .base import ModelTransform, retype_node

__all__ = ["ContextParallelTransform"]


class ContextParallelTransform(ModelTransform):
    """Run attention under context parallelism.

    Replace every attention kernel with ``kernel`` while preserving its config.

    TODO(fegin): support one kernel per attention type, for models that mix
    them.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ModelTransform.Config):
        kernel: type[Module]
        """CP attention kernel; must inherit ``ContextParallelKernel``."""

        kernel_config_overrides: dict[str, object] = field(default_factory=dict)
        """Values for config fields defined by the CP kernel."""

    def transform(self, model: Module.Config) -> Module.Config:
        kernel = self.config.kernel
        if not issubclass(kernel, ContextParallelKernel):
            raise ValueError(
                f"{kernel.__qualname__} must inherit ContextParallelKernel."
            )
        for _, traversed, _, _ in model.traverse(BaseAttention.Config):
            # traverse returns the base config type.
            attention = cast(BaseAttention.Config, traversed)
            attention.inner_attention = retype_node(
                attention.inner_attention,
                kernel,
                **self.config.kernel_config_overrides,
            )
        return model
