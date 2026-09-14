# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.models.common.linear import (
    Linear,
    StructuredLinear,
    StructuredLinearBase,
)
from torchtitan.protocols.module import Module


try:
    from torchao.float8.float8_linear import (
        Float8Linear as TorchAOFloat8Linear,
        matmul_with_hp_or_float8_args,
    )

    class Float8Linear(TorchAOFloat8Linear, Module):
        """Inherits from Module (not Linear) to satisfy the Module protocol
        (init_states, _param_init) while avoiding MRO conflicts with
        Linear.__init__. Config still inherits from Linear.Config for
        field compatibility.
        """

        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            """Drop-in replacement for Linear.Config that builds Float8Linear."""

            _torchao_config: object = None

        def __init__(self, config: Config):
            TorchAOFloat8Linear.__init__(
                self,
                config.in_features,
                config.out_features,
                bias=config.bias,
                config=config._torchao_config,
            )

    class Float8StructuredLinear(TorchAOFloat8Linear, StructuredLinearBase, Module):
        """Float8 linear whose parameter retains structured output dimensions."""

        @dataclass(kw_only=True, slots=True)
        class Config(StructuredLinear.Config):
            _torchao_config: object = None

        def __init__(self, config: Config):
            TorchAOFloat8Linear.__init__(
                self,
                config.in_features,
                config.out_features,
                bias=config.bias,
                config=config._torchao_config,
            )
            self.output_shape = config.output_shape
            self.weight = torch.nn.Parameter(
                self.weight.detach().reshape(*config.output_shape, config.in_features),
                requires_grad=self.weight.requires_grad,
            )
            if self.bias is not None:
                self.bias = torch.nn.Parameter(
                    self.bias.detach().reshape(*config.output_shape),
                    requires_grad=self.bias.requires_grad,
                )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if torch.is_autocast_enabled():
                input = input.to(torch.get_autocast_gpu_dtype())
            output = matmul_with_hp_or_float8_args.apply(
                input,
                self.weight.flatten(0, -2).t(),
                self.linear_mm_config,
                self.config,
            )
            if self.bias is not None:
                output = output + self.bias.flatten().to(output.dtype)
            return output.unflatten(-1, self.weight.shape[:-1])

except ImportError:
    Float8Linear = None
    Float8StructuredLinear = None


_float8_experts_cache: dict[type, type] = {}


def _get_float8_grouped_experts_cls(parent_cls: type) -> type:
    """Get or create a Float8-quantized subclass of *parent_cls*.

    Works for any ``GroupedExperts`` subclass (e.g. gpt-oss variants).
    The returned class has a proper ``_owner`` set by ``__init_subclass__``.
    """
    if parent_cls in _float8_experts_cache:
        return _float8_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class Float8GroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            pass

        def __init__(self, config: Config):
            super().__init__(config)
            from torchao.prototype.moe_training.config import Float8TrainingOpConfig

            self._float8_op_config = Float8TrainingOpConfig()

        def _grouped_mm(self, *, A, weight_EOI, offs):
            from torchao.prototype.moe_training.utils import (
                _quantize_then_scaled_grouped_mm,
            )

            return _quantize_then_scaled_grouped_mm(
                A,
                weight_EOI.bfloat16().transpose(-2, -1),
                config=self._float8_op_config,
                offs=offs,
            )

    Float8GroupedExperts.__name__ = f"Float8{parent_cls.__name__}"
    Float8GroupedExperts.__qualname__ = f"Float8{parent_cls.__name__}"
    _float8_experts_cache[parent_cls] = Float8GroupedExperts
    return Float8GroupedExperts
