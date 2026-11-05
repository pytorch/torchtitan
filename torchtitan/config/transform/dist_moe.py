# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transforms for the standalone Dist-MoE routed-expert backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from torchtitan.models.base.module import Module

from torchtitan.models.common.activation import ClampedSwiGLU, SwiGLU
from torchtitan.models.common.dist_moe import (
    _dist_moe as dist_moe,
    DistMoeRoutedExperts,
    MXFP8DistMoeRoutedExperts,
    NVFP4FwdBF16BwdDistMoeRoutedExperts,
)
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moe.token_dispatcher import AllToAllTokenDispatcher
from .base import ModelConfigTransform


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
        ):
            raise TypeError("Dist-MoE requires stock GroupedLinear W13/W2 projections")
        if isinstance(source.activation_fn, SwiGLU.Config):
            activation = "swiglu"
            swiglu_alpha = None
            swiglu_limit = None
        elif isinstance(source.activation_fn, ClampedSwiGLU.Config):
            activation = "swiglu_clamped"
            swiglu_alpha = source.activation_fn.alpha
            swiglu_limit = source.activation_fn.limit
        else:
            raise TypeError(
                "Dist-MoE supports SwiGLU.Config or ClampedSwiGLU.Config; "
                f"got {type(source.activation_fn).__qualname__} at {fqn!r}"
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
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
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
    """Replace stock routed experts with a selected Dist-MoE precision.

    Expert activation semantics come from each source
    ``RoutedExperts.Config.activation_fn``. The transform translates supported
    source activations to the Annex kernel policy instead of owning an
    independent activation setting.

    Args:
        expert_precision: Expert compute precision. BF16 is the high-precision
            default. MXFP8 uses native block-scaled forward and backward;
            NVFP4 forward with BF16 backward uses cached NVFP4 forward weights
            and the annex's BF16 STE backward.
        inplace_wgrad_accum: Whether Dist-MoE writes W13/W2 gradients directly
            into existing standard ``parameter.grad`` buffers. This is enabled
            by default; Dist-MoE derives the owners from the logical expert
            weights and exposes the destination to graph tracing.
        bf16_grouped_gemm_preset: Optional expert override for the annex BF16
            FPROP/DGRAD grouped-GEMM schedule. ``None`` uses production defaults.
        block_scaled_config: Block-scaled kernel policy. ``None`` selects the
            format-specific defaults for MXFP8 or NVFP4.
        mxfp8_mega_fused_inference: Prepare MXFP8 weights in the
            gate/up interleaved format so the Mega kernel can fuse SwiGLU into
            the W13 GEMM, for inference only. Quantized weights and scales match
            training numerics exactly. Requires
            ``expert_precision="mxfp8"`` and ``block_scaled_config.pipeline="mega"``.
    """

    expert_precision: Literal["bf16", "mxfp8", "nvfp4_fwd_bf16_bwd"] = "bf16"
    inplace_wgrad_accum: bool = True
    bf16_grouped_gemm_preset: dist_moe.Bf16GroupedGemmPreset | None = None
    block_scaled_config: dist_moe.BlockScaledConfig | None = None
    mxfp8_mega_fused_inference: bool = False

    def __post_init__(self) -> None:
        if self.expert_precision == "bf16":
            if self.block_scaled_config is not None:
                raise ValueError(
                    "block_scaled_config requires a block-scaled expert precision"
                )
            if self.mxfp8_mega_fused_inference:
                raise ValueError(
                    "mxfp8_mega_fused_inference requires " "expert_precision='mxfp8'"
                )
        elif self.expert_precision in ("mxfp8", "nvfp4_fwd_bf16_bwd"):
            if self.bf16_grouped_gemm_preset is not None:
                raise ValueError(
                    "bf16_grouped_gemm_preset requires expert_precision='bf16'"
                )
            block_scaled_config = (
                self.block_scaled_config
                or dist_moe.BlockScaledConfig(
                    format=(
                        dist_moe.BlockScaledFormat.MXFP8_E4M3
                        if self.expert_precision == "mxfp8"
                        else dist_moe.BlockScaledFormat.NVFP4
                    )
                )
            )
            expected_format = (
                dist_moe.BlockScaledFormat.MXFP8_E4M3
                if self.expert_precision == "mxfp8"
                else dist_moe.BlockScaledFormat.NVFP4
            )
            if block_scaled_config.format is not expected_format:
                raise ValueError(
                    f"Dist-MoE {self.expert_precision} requires the "
                    f"{expected_format.name} format"
                )
            if self.mxfp8_mega_fused_inference and self.expert_precision != "mxfp8":
                raise ValueError(
                    "mxfp8_mega_fused_inference requires " "expert_precision='mxfp8'"
                )
            if (
                self.mxfp8_mega_fused_inference
                and block_scaled_config.pipeline != "mega"
            ):
                raise ValueError(
                    "mxfp8_mega_fused_inference requires "
                    "block_scaled_config.pipeline='mega'"
                )
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
        if self.expert_precision == "mxfp8":
            return _replace_routed_experts(
                model,
                MXFP8DistMoeRoutedExperts,
                inplace_wgrad_accum=self.inplace_wgrad_accum,
                block_scaled_config=(
                    self.block_scaled_config or dist_moe.BlockScaledConfig()
                ),
                mega_fused_inference=self.mxfp8_mega_fused_inference,
            )
        return _replace_routed_experts(
            model,
            NVFP4FwdBF16BwdDistMoeRoutedExperts,
            inplace_wgrad_accum=self.inplace_wgrad_accum,
            block_scaled_config=(
                self.block_scaled_config
                or dist_moe.BlockScaledConfig(format=dist_moe.BlockScaledFormat.NVFP4)
            ),
        )
