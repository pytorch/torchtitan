# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP4-weight/MXFP8-activation QAT for grouped experts."""

from dataclasses import dataclass

import torch


_mx_qat_experts_cache: dict[type, type] = {}


def _get_mx_qat_grouped_experts_cls(parent_cls: type) -> type:
    """Return a grouped-expert subclass using stateless TorchAO MX QAT."""
    if parent_cls in _mx_qat_experts_cache:
        return _mx_qat_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class MXQATGroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            weight_block_size: int = 32
            activation_block_size: int = 32

        def __init__(self, config: Config):
            super().__init__(config)
            from torchao.prototype.qat import MXFakeQuantizeConfig

            self._weight_fake_quant_config = MXFakeQuantizeConfig(
                dtype=torch.float4_e2m1fn_x2,
                block_size=config.weight_block_size,
            )
            self._activation_fake_quant_config = MXFakeQuantizeConfig(
                dtype=torch.float8_e4m3fn,
                block_size=config.activation_block_size,
            )

        def _grouped_mm(self, *, A, weight_EOI, offs):
            from torchao.prototype.qat import mx_fake_quantized_grouped_mm

            return mx_fake_quantized_grouped_mm(
                A.bfloat16(),
                weight_EOI.bfloat16(),
                offs,
                self._activation_fake_quant_config,
                self._weight_fake_quant_config,
            )

    MXQATGroupedExperts.__name__ = f"MXQAT{parent_cls.__name__}"
    MXQATGroupedExperts.__qualname__ = f"MXQAT{parent_cls.__name__}"
    _mx_qat_experts_cache[parent_cls] = MXQATGroupedExperts
    return MXQATGroupedExperts
