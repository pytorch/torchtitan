# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transforms for the standalone Dist-MoE routed-expert backend."""

from dataclasses import dataclass, field, replace
from typing import Any

import dist_moe

from torchtitan.components.runtime import TrainingRuntime
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts, DistMoeRuntime
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.protocols.module import Module
from torchtitan.quantization.mxfp8.dist_moe import MXFP8DistMoeRoutedExperts

from .base import convert_config_type, ModelConfigTransform


__all__ = ["DistMoeTransform", "MXFP8DistMoeTransform"]


def _replace_routed_experts(
    model: Module.Config,
    replacement_type: type[DistMoeRoutedExperts],
    **values: Any,
) -> Module.Config:
    """Replace every stock routed-expert config with one Dist-MoE variant."""
    for fqn, source, parent, attr in list(model.traverse(RoutedExperts.Config)):
        if type(source) is not RoutedExperts.Config:
            raise TypeError(
                "Dist-MoE requires an unmodified RoutedExperts.Config source; "
                f"got {type(source).__qualname__} at {fqn!r}"
            )
        replacement = replace(
            convert_config_type(source, replacement_type),
            **values,
        )
        if parent is None:
            model = replacement
        elif isinstance(parent, list):
            assert isinstance(attr, int)
            parent[attr] = replacement
        else:
            assert isinstance(attr, str)
            setattr(parent, attr, replacement)
    return model


@dataclass(kw_only=True, slots=True)
class DistMoeTransform(ModelConfigTransform):
    """Replace stock routed experts with native BF16 Dist-MoE execution.

    Args:
        runtime: Rank-wide memory, scratch, VMM, and PP activation-slot policy.
        inplace_wgrad_accum: Whether Dist-MoE writes W13/W2 gradients directly
            into existing standard ``parameter.grad`` buffers. Dist-MoE
            derives the owners from the logical expert weights. Keep this
            disabled with GraphTrainer until its Dist-MoE WGRAD fusion pass is
            available.
        bf16_grouped_gemm_preset: Optional expert override for the annex BF16
            FPROP/DGRAD grouped-GEMM schedule. ``None`` uses production defaults.
    """

    runtime: DistMoeRuntime.Config = field(default_factory=DistMoeRuntime.Config)
    inplace_wgrad_accum: bool = False
    bf16_grouped_gemm_preset: dist_moe.Bf16GroupedGemmPreset | None = None

    def transform(self, model: Module.Config) -> Module.Config:
        """Replace each stock routed-expert config directly with BF16 Dist-MoE."""
        return _replace_routed_experts(
            model,
            DistMoeRoutedExperts,
            inplace_wgrad_accum=self.inplace_wgrad_accum,
            bf16_grouped_gemm_preset=self.bf16_grouped_gemm_preset,
        )

    def runtime_configs(self) -> tuple[TrainingRuntime.Config, ...]:
        """Register the one rank-wide runtime required by Dist-MoE modules."""
        return (self.runtime,)


@dataclass(kw_only=True, slots=True)
class MXFP8DistMoeTransform(ModelConfigTransform):
    """Replace stock routed experts directly with native MXFP8 Dist-MoE.

    Args:
        runtime: Rank-wide memory, scratch, VMM, and PP activation-slot policy.
        inplace_wgrad_accum: Whether Dist-MoE writes W13/W2 gradients directly
            into existing standard ``parameter.grad`` buffers. Dist-MoE
            derives the owners from the logical expert weights. Keep this
            disabled with GraphTrainer until its Dist-MoE WGRAD fusion pass is
            available.
        block_scaled_config: Native annex MXFP8 policy. ``pipeline`` selects
            staged or fused Mega execution, ``fast_math`` controls approximate
            SwiGLU sigmoid math, and ``kernel_config`` is an expert-only CuTe
            tuning override.
    """

    runtime: DistMoeRuntime.Config = field(default_factory=DistMoeRuntime.Config)
    inplace_wgrad_accum: bool = False
    block_scaled_config: dist_moe.BlockScaledConfig = field(
        default_factory=dist_moe.BlockScaledConfig
    )

    def __post_init__(self) -> None:
        if self.block_scaled_config.format is not dist_moe.BlockScaledFormat.MXFP8_E4M3:
            raise ValueError("MXFP8DistMoeTransform requires MXFP8_E4M3")

    def transform(self, model: Module.Config) -> Module.Config:
        """Replace each stock routed-expert config directly with MXFP8 Dist-MoE."""
        return _replace_routed_experts(
            model,
            MXFP8DistMoeRoutedExperts,
            inplace_wgrad_accum=self.inplace_wgrad_accum,
            block_scaled_config=self.block_scaled_config,
        )

    def runtime_configs(self) -> tuple[TrainingRuntime.Config, ...]:
        """Register the one rank-wide runtime required by Dist-MoE modules."""
        return (self.runtime,)


DistMoeTransform.conflicts_with = (MXFP8DistMoeTransform,)
MXFP8DistMoeTransform.conflicts_with = (DistMoeTransform,)
