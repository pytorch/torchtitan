# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torchtitan.config import Configurable
from torchtitan.tools.logging import logger


class QATConverter(Configurable):
    """Replace nn.Linear with FakeQuantizedLinear for quantization-aware training.

    Uses torchao's FakeQuantizedLinear to simulate int4 weight quantization during
    training. The fake quantization is applied in the forward pass so the model
    learns to compensate for quantization error.

    When composed with LoRA (QATConverter listed before LoRAConverter in converters),
    LoRA will inherit from FakeQuantizedLinear so base weights are fake-quantized
    while LoRA adapters stay full-precision.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dtype: Literal["int4", "int8"] = "int4"
        """Data type for fake quantization. Supported: 'int4', 'int8'."""

        group_size: int = 256
        """Group size for per-group weight quantization.
        Must divide in_features of all Linear layers in the model."""

    def __init__(self, config: Config, **kwargs):
        self.dtype = config.dtype
        self.group_size = config.group_size
        logger.info(
            f"QAT training active (dtype={self.dtype}, group_size={self.group_size})"
        )

    def convert(self, model: nn.Module) -> None:
        from torchao.quantization.qat import FakeQuantizedLinear, IntxFakeQuantizeConfig

        dtype_map = {
            "int4": torch.int4,  # pyre-ignore[16]
            "int8": torch.int8,
        }
        torch_dtype = dtype_map[self.dtype]

        weight_config = IntxFakeQuantizeConfig(
            dtype=torch_dtype,
            group_size=self.group_size,
            is_symmetric=True,
        )

        def _replace_recursive(parent: nn.Module) -> None:
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    fq = FakeQuantizedLinear.from_linear(
                        child, weight_config=weight_config
                    )
                    setattr(parent, name, fq)
                else:
                    _replace_recursive(child)

        _replace_recursive(model)
        logger.info(
            "Swapped to FakeQuantizedLinear layers "
            f"(dtype={self.dtype}, group_size={self.group_size})"
        )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        pass
