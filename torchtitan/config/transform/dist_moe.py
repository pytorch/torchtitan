# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-config transforms for the DistMoE routed-expert backend."""

from dataclasses import dataclass, replace
from typing import Literal

from dist_moe import Bf16GroupedGemmPreset, BlockScaledKernelConfig, VmmConfig

from torchtitan.components.dist_moe import (
    DistMoeRoutedExperts,
    MXFP8DistMoeRoutedExperts,
)
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform


__all__ = ["DistMoeTransform", "MXFP8DistMoeTransform"]


def _replace_traversed_config(model, entry, replacement):
    """Install a replacement at the location returned by config traversal.

    Traversal identifies a config by its parent and attribute. Root configs have
    no parent and are replaced by returning the new value; list-owned configs use
    their integer index; all other configs use their dataclass attribute name.

    Args:
        model: Current root model configuration.
        entry: Traversal tuple containing FQN, value, parent, and parent key.
        replacement: Config to install at the traversed location.

    Returns:
        The original root after an in-place nested replacement, or the
        replacement itself when the traversed value was the root.
    """
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
    """Replace standard routed experts with the BF16 DistMoE backend."""

    device_scratch_capacity_factor: float = 1.0
    saved_activation_buffer_bytes: int | None = None
    activation_slot_policy: Literal["auto", "microbatch", "stage_microbatch"] = "auto"
    num_activation_slots: int | None = None
    vmm: VmmConfig | None = None
    num_sms: int | None = None
    bf16_grouped_gemm_preset: Bf16GroupedGemmPreset | None = None
    wgrad_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    inplace_wgrad_accum: bool = False

    def transform(self, model: Module.Config) -> Module.Config:
        """Replace each stock routed-expert config with the BF16 backend."""
        targets = list(model.traverse(RoutedExperts.Config))
        for entry in targets:
            config = entry[1]
            assert isinstance(config, RoutedExperts.Config)
            if type(config) is not RoutedExperts.Config:
                raise TypeError(
                    "DistMoE cannot convert a specialized RoutedExperts.Config; "
                    "express model-specific behavior through the common "
                    "routed-expert contract"
                )
            converted = convert_config_type(config, DistMoeRoutedExperts)
            assert isinstance(converted, DistMoeRoutedExperts.Config)
            replacement = replace(
                converted,
                device_scratch_capacity_factor=self.device_scratch_capacity_factor,
                saved_activation_buffer_bytes=self.saved_activation_buffer_bytes,
                activation_slot_policy=self.activation_slot_policy,
                num_activation_slots=self.num_activation_slots,
                vmm=self.vmm,
                num_sms=self.num_sms,
                bf16_grouped_gemm_preset=self.bf16_grouped_gemm_preset,
                wgrad_dtype=self.wgrad_dtype,
                inplace_wgrad_accum=self.inplace_wgrad_accum,
            )
            assert isinstance(replacement, DistMoeRoutedExperts.Config)
            model = _replace_traversed_config(model, entry, replacement)
        return model


@dataclass(kw_only=True, slots=True)
class MXFP8DistMoeTransform(ModelConfigTransform):
    """Upgrade BF16 DistMoE configs to the native MXFP8 module variant."""

    run_after = (DistMoeTransform,)

    pipeline: Literal["staged", "mega"] = "staged"
    fast_math: bool = False
    kernel_config: BlockScaledKernelConfig | None = None

    def transform(self, model: Module.Config) -> Module.Config:
        """Upgrade each BF16 DistMoE config to asynchronous MXFP8."""
        targets = list(model.traverse(DistMoeRoutedExperts.Config))
        for entry in targets:
            config = entry[1]
            if type(config) is not DistMoeRoutedExperts.Config:
                continue
            converted = convert_config_type(config, MXFP8DistMoeRoutedExperts)
            assert isinstance(converted, MXFP8DistMoeRoutedExperts.Config)
            replacement = replace(
                converted,
                pipeline=self.pipeline,
                fast_math=self.fast_math,
                kernel_config=self.kernel_config,
            )
            assert isinstance(replacement, MXFP8DistMoeRoutedExperts.Config)
            model = _replace_traversed_config(model, entry, replacement)
        return model
