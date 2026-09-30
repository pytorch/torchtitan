# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transforms for the standalone Dist-MoE routed-expert backend."""

from dataclasses import dataclass
from typing import Any, Literal

import dist_moe

from torchtitan.models.common.activation import SwiGLU

from torchtitan.models.common.dist_moe import (
    DistMoeRoutedExperts,
    MXFP8DistMoeRoutedExperts,
)
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.protocols.module import Module
from .base import ModelConfigTransform
from .lora import LoRATransform


__all__ = ["DistMoeTransform"]


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
        if (
            type(source.w13) is not GroupedLinear.Config
            or type(source.w2) is not GroupedLinear.Config
            or type(source.activation_fn) is not SwiGLU.Config
        ):
            raise TypeError(
                "Dist-MoE requires stock GroupedLinear W13/W2 projections and SwiGLU"
            )
        if not isinstance(source.token_dispatcher, AllToAllTokenDispatcher.Config):
            raise ValueError(
                "Dist-MoE requires the standard all-to-all source config; "
                "the annex replaces its runtime dispatch and combine"
            )
        replacement = replacement_type.Config(
            param_init=source.param_init,
            sharding_config=source.sharding_config,
            w13=source.w13,
            w2=source.w2,
            top_k=source.token_dispatcher.top_k,
            output_postprocess=source.output_postprocess,
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
    """Replace stock routed experts with BF16 or MXFP8 Dist-MoE execution.

    Args:
        expert_precision: Expert compute precision. BF16 is the high-precision
            default; MXFP8 selects the annex's asynchronous block-scaled path.
        inplace_wgrad_accum: Whether Dist-MoE writes W13/W2 gradients directly
            into existing standard ``parameter.grad`` buffers. Dist-MoE
            derives the owners from the logical expert weights. Keep this
            disabled with GraphTrainer until its Dist-MoE WGRAD fusion pass is
            available.
        bf16_grouped_gemm_preset: Optional expert override for the annex BF16
            FPROP/DGRAD grouped-GEMM schedule. ``None`` uses production defaults.
        block_scaled_config: MXFP8 kernel policy. ``None`` selects the annex
            defaults when ``expert_precision`` is ``"mxfp8"``.
    """

    expert_precision: Literal["bf16", "mxfp8"] = "bf16"
    inplace_wgrad_accum: bool = False
    bf16_grouped_gemm_preset: dist_moe.Bf16GroupedGemmPreset | None = None
    block_scaled_config: dist_moe.BlockScaledConfig | None = None

    def __post_init__(self) -> None:
        if self.expert_precision == "bf16":
            if self.block_scaled_config is not None:
                raise ValueError(
                    "block_scaled_config requires expert_precision='mxfp8'"
                )
        elif self.expert_precision == "mxfp8":
            if self.bf16_grouped_gemm_preset is not None:
                raise ValueError(
                    "bf16_grouped_gemm_preset requires expert_precision='bf16'"
                )
            block_scaled_config = (
                self.block_scaled_config or dist_moe.BlockScaledConfig()
            )
            if block_scaled_config.format is not dist_moe.BlockScaledFormat.MXFP8_E4M3:
                raise ValueError("Dist-MoE MXFP8 requires the MXFP8_E4M3 format")
        else:
            raise ValueError(
                f"unsupported Dist-MoE expert precision {self.expert_precision!r}"
            )

    def transform(self, model: Module.Config) -> Module.Config:
        """Replace each stock routed-expert config with the selected precision."""
        if self.expert_precision == "bf16":
            return _replace_routed_experts(
                model,
                DistMoeRoutedExperts,
                inplace_wgrad_accum=self.inplace_wgrad_accum,
                bf16_grouped_gemm_preset=self.bf16_grouped_gemm_preset,
            )
        return _replace_routed_experts(
            model,
            MXFP8DistMoeRoutedExperts,
            inplace_wgrad_accum=self.inplace_wgrad_accum,
            block_scaled_config=(
                self.block_scaled_config or dist_moe.BlockScaledConfig()
            ),
        )


DistMoeTransform.conflicts_with = (LoRATransform,)
