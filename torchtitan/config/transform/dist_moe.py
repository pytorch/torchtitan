# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-config transforms for the DistMoE routed-expert backend."""

from dataclasses import dataclass, replace
from typing import Literal

from dist_moe import DistMoeBlockScaledKernelConfig

from torchtitan.components.dist_moe import (
    DistMoeRoutedExperts,
    MXFP8DistMoeRoutedExperts,
)
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform


__all__ = ["DistMoeTransform", "MXFP8DistMoeTransform"]


def _replace_config(model, entry, replacement):
    """Replace one traversed config and return the possibly replaced root."""
    _fqn, _config, parent, attr = entry
    if parent is None:
        return replacement
    if isinstance(parent, list):
        parent[attr] = replacement
    else:
        setattr(parent, attr, replacement)
    return model


def _validate_source(config: RoutedExperts.Config) -> None:
    """Reject model behavior that DistMoE cannot preserve."""
    if type(config) is not RoutedExperts.Config:
        raise TypeError(
            "DistMoE cannot convert a specialized RoutedExperts.Config; express "
            "model-specific behavior through the common routed-expert contract"
        )
    if (
        type(config.w13) is not GroupedLinear.Config
        or type(config.w2) is not GroupedLinear.Config
        or type(config.activation_fn) is not SwiGLU.Config
    ):
        raise TypeError(
            "DistMoE requires the stock grouped-linear projections and SwiGLU"
        )
    if not isinstance(config.token_dispatcher, AllToAllTokenDispatcher.Config):
        raise ValueError(
            "DistMoE owns expert communication and requires the standard "
            "all-to-all routed-expert config"
        )
    postprocess = config.expert_output_postprocess
    owner = None if postprocess is None else postprocess._owner
    if postprocess is not None and not callable(
        getattr(owner, "to_dist_moe_postprocess", None)
    ):
        raise TypeError(
            f"{type(postprocess).__qualname__} cannot run inside DistMoE; its "
            "module must define to_dist_moe_postprocess()"
        )


@dataclass(kw_only=True, slots=True)
class DistMoeTransform(ModelConfigTransform):
    """Replace standard routed experts with the BF16 DistMoE backend."""

    max_routing_imbalance_factor: float = 1.0
    device_memory_budget_bytes: int | None = None
    activation_slot_policy: Literal["auto", "microbatch", "stage_microbatch"] = "auto"
    num_activation_slots: int | None = None
    vmm_host_scratch_imbalance_factor: float | None = None
    prefetch_vmm: bool = False
    num_sms: int | None = None
    kernel_config: str | None = None
    wgrad_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    inplace_wgrad_accum: bool = False

    def transform(self, model: Module.Config) -> Module.Config:
        targets = list(model.traverse(RoutedExperts.Config))
        for entry in targets:
            config = entry[1]
            assert isinstance(config, RoutedExperts.Config)
            _validate_source(config)
            converted = convert_config_type(config, DistMoeRoutedExperts)
            assert isinstance(converted, DistMoeRoutedExperts.Config)
            replacement = replace(
                converted,
                max_routing_imbalance_factor=self.max_routing_imbalance_factor,
                device_memory_budget_bytes=self.device_memory_budget_bytes,
                activation_slot_policy=self.activation_slot_policy,
                num_activation_slots=self.num_activation_slots,
                vmm_host_scratch_imbalance_factor=(
                    self.vmm_host_scratch_imbalance_factor
                ),
                prefetch_vmm=self.prefetch_vmm,
                num_sms=self.num_sms,
                kernel_config=self.kernel_config,
                wgrad_dtype=self.wgrad_dtype,
                inplace_wgrad_accum=self.inplace_wgrad_accum,
            )
            assert isinstance(replacement, DistMoeRoutedExperts.Config)
            model = _replace_config(model, entry, replacement)
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
            converted = convert_config_type(config, MXFP8DistMoeRoutedExperts)
            assert isinstance(converted, MXFP8DistMoeRoutedExperts.Config)
            replacement = replace(
                converted,
                pipeline=self.pipeline,
                fast_math=self.fast_math,
                kernel_config=self.kernel_config,
            )
            assert isinstance(replacement, MXFP8DistMoeRoutedExperts.Config)
            model = _replace_config(model, entry, replacement)
        return model
