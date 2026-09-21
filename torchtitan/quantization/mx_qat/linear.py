# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Weight-only MX QAT for ordinary Linear projections."""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .experts import MXFakeQuantizeConfig, weight_config

_linear_cache: dict[type, type] = {}


def _get_mx_qat_linear_cls(parent_cls: type) -> type:
    if getattr(parent_cls, "_mx_qat", False):
        return parent_cls
    if parent_cls in _linear_cache:
        return _linear_cache[parent_cls]
    if parent_cls.forward is not torch.nn.Linear.forward:
        raise ValueError(
            f"MX QAT cannot replace custom forward of {parent_cls.__name__}"
        )
    parent_config_cls = parent_cls.Config

    class MXQATLinear(parent_cls):
        _mx_qat = True

        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):
            weight_fake_quant_config: "MXFakeQuantizeConfig" = field(
                default_factory=weight_config
            )

        def __init__(self, config: Config):
            super().__init__(config)
            self._weight_fake_quant_config = config.weight_fake_quant_config

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            from torchao.prototype.qat import mx_fake_quantize

            return F.linear(
                input,
                mx_fake_quantize(self.weight, self._weight_fake_quant_config),
                self.bias,
            )

    MXQATLinear.__name__ = f"MXQAT{parent_cls.__name__}"
    MXQATLinear.__qualname__ = MXQATLinear.__name__
    _linear_cache[parent_cls] = MXQATLinear
    return MXQATLinear
