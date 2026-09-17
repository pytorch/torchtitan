# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
)
from torchtitan.protocols.module import Module


try:
    from torchao.float8.float8_linear import Float8Linear as TorchAOFloat8Linear

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

        def _linear(self, input: torch.Tensor) -> torch.Tensor:
            return TorchAOFloat8Linear.forward(self, input)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self._linear(input)

    class Float8ColumnParallelLinear(ColumnParallelLinear, Float8Linear):
        """Float8 projection with a synchronous column-parallel boundary."""

        @dataclass(kw_only=True, slots=True)
        class Config(Float8Linear.Config, ColumnParallelLinear.Config):
            pass

        # ColumnParallelLinear appears before Float8Linear in the MRO so its
        # forward owns communication. Delegate construction and local compute
        # explicitly to Float8Linear instead of falling back to BF16 Linear.
        def __init__(self, config: Config):
            Float8Linear.__init__(self, config)  # pyrefly: ignore[bad-argument-count]

        def _linear(self, input: torch.Tensor) -> torch.Tensor:
            return Float8Linear._linear(  # pyrefly: ignore[missing-attribute]
                self, input
            )

    class Float8RowParallelLinear(RowParallelLinear, Float8Linear):
        """Float8 projection with a synchronous row-parallel boundary."""

        @dataclass(kw_only=True, slots=True)
        class Config(Float8Linear.Config, RowParallelLinear.Config):
            pass

        # RowParallelLinear appears before Float8Linear in the MRO so its
        # forward owns communication. Keep Float8 initialization and compute
        # explicit for the same reason as the column-parallel variant above.
        def __init__(self, config: Config):
            Float8Linear.__init__(self, config)  # pyrefly: ignore[bad-argument-count]

        def _linear(self, input: torch.Tensor) -> torch.Tensor:
            return Float8Linear._linear(  # pyrefly: ignore[missing-attribute]
                self, input
            )

except ImportError:
    Float8Linear = None
    Float8ColumnParallelLinear = None
    Float8RowParallelLinear = None


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
