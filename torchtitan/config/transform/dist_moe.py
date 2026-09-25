# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-config transforms for the optional DistMoE routed-expert backend."""

from dataclasses import dataclass, field, replace
from typing import Literal

from dist_moe import (
    DistMoeBlockScaledConfig,
    DistMoeBlockScaledFormat,
    DistMoeBlockScaledKernelConfig,
)

from torchtitan.components.dist_moe import (
    DistMoeBackendConfig,
    DistMoeRoutedExperts,
    MXFP8DistMoeRoutedExperts,
)
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform


__all__ = ["DistMoeTransform", "MXFP8DistMoeTransform"]


def _replace_traversed_config(model, entry, replacement):
    """Install a replacement at the location returned by config traversal."""
    _fqn, _config, parent, attr = entry
    if parent is None:
        return replacement
    if isinstance(parent, list):
        parent[attr] = replacement
    else:
        setattr(parent, attr, replacement)
    return model


@dataclass(kw_only=True, slots=True)
class DistMoeTransform(ModelConfigTransform):
    """Replace stock routed experts with the DistMoE backend."""

    backend: DistMoeBackendConfig = field(default_factory=DistMoeBackendConfig)

    def transform(self, model: Module.Config) -> Module.Config:
        targets = list(model.traverse(RoutedExperts.Config))
        for entry in targets:
            config = entry[1]
            if type(config) is not RoutedExperts.Config:
                raise TypeError(
                    "DistMoE cannot convert a specialized RoutedExperts.Config; "
                    "express model-specific behavior through the common "
                    "routed-expert contract"
                )
            replacement = convert_config_type(config, DistMoeRoutedExperts)
            assert isinstance(replacement, DistMoeRoutedExperts.Config)
            replacement.backend = self.backend
            model = _replace_traversed_config(model, entry, replacement)
        return model


@dataclass(kw_only=True, slots=True)
class MXFP8DistMoeTransform(ModelConfigTransform):
    """Upgrade BF16 DistMoE configs to the native MXFP8 module variant."""

    run_after = (DistMoeTransform,)

    pipeline: Literal["staged", "mega"] = "staged"
    fast_math: bool = False
    kernel_config: DistMoeBlockScaledKernelConfig | None = None

    def transform(self, model: Module.Config) -> Module.Config:
        targets = list(model.traverse(DistMoeRoutedExperts.Config))
        for entry in targets:
            config = entry[1]
            if type(config) is not DistMoeRoutedExperts.Config:
                continue
            block_scaled = DistMoeBlockScaledConfig(
                format=DistMoeBlockScaledFormat.MXFP8_E4M3,
                fast_math=self.fast_math,
                pipeline=self.pipeline,
                kernel_config=self.kernel_config,
            )
            converted = convert_config_type(config, MXFP8DistMoeRoutedExperts)
            assert isinstance(converted, MXFP8DistMoeRoutedExperts.Config)
            replacement = replace(
                converted,
                backend=replace(
                    config.backend,
                    bf16_grouped_gemm_preset=None,
                    block_scaled=block_scaled,
                ),
            )
            model = _replace_traversed_config(model, entry, replacement)
        return model
