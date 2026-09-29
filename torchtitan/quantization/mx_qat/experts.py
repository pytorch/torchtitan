# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP4-weight/MXFP8-activation QAT for grouped linears."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchtitan.models.common.linear import GroupedLinear

try:
    from torchao.prototype.qat import MXFakeQuantizeConfig
except ImportError:
    # Keep non-QAT recipes importable when optional TorchAO is not installed.
    from typing import Any as MXFakeQuantizeConfig


def weight_config():
    from torchao.prototype.qat import MXFakeQuantizeConfig

    return MXFakeQuantizeConfig(dtype=torch.float4_e2m1fn_x2)


def activation_config():
    from torchao.prototype.qat import MXFakeQuantizeConfig

    return MXFakeQuantizeConfig(dtype=torch.float8_e4m3fn)


_mx_qat_grouped_linear_cache: dict[type["GroupedLinear"], type["GroupedLinear"]] = {}


def _get_mx_qat_grouped_linear_cls(
    parent_cls: type["GroupedLinear"],
) -> type["GroupedLinear"]:
    """Return a grouped-linear subclass using stateless TorchAO MX QAT."""
    if getattr(parent_cls, "_mx_qat", False):
        return parent_cls
    if parent_cls in _mx_qat_grouped_linear_cache:
        return _mx_qat_grouped_linear_cache[parent_cls]
    from torchtitan.models.common.linear import GroupedLinear

    if parent_cls._grouped_mm is not GroupedLinear._grouped_mm:
        raise ValueError(
            f"MX QAT cannot replace an existing grouped-MM override on {parent_cls.__name__}"
        )

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class MXQATGroupedLinear(parent_cls):
        _mx_qat = True

        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            weight_fake_quant_config: "MXFakeQuantizeConfig" = field(
                default_factory=weight_config
            )
            activation_fake_quant_config: "MXFakeQuantizeConfig" = field(
                default_factory=activation_config
            )

        def __init__(self, config: Config):
            super().__init__(config)
            self._weight_fake_quant_config = config.weight_fake_quant_config
            self._activation_fake_quant_config = config.activation_fake_quant_config

        def _grouped_mm(
            self,
            *,
            input_RI: torch.Tensor,
            weight_EOI: torch.Tensor,
            offsets_E: torch.Tensor,
        ) -> torch.Tensor:
            from torchao.prototype.qat import mx_fake_quantized_grouped_mm

            return mx_fake_quantized_grouped_mm(
                input_RI.bfloat16(),
                weight_EOI.bfloat16(),
                offsets_E,
                self._activation_fake_quant_config,
                self._weight_fake_quant_config,
            )

    MXQATGroupedLinear.__name__ = f"MXQAT{parent_cls.__name__}"
    MXQATGroupedLinear.__qualname__ = f"MXQAT{parent_cls.__name__}"
    _mx_qat_grouped_linear_cache[parent_cls] = MXQATGroupedLinear
    return MXQATGroupedLinear
