# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.common.linear import GroupedLinear, Linear
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

except ImportError:
    Float8Linear = None


_float8_grouped_linear_cache: dict[type, type] = {}


def _get_float8_grouped_linear_cls(parent_cls: type[GroupedLinear]) -> type:
    """Get or create a Float8-quantized subclass of *parent_cls*.

    Works for any ``GroupedLinear`` subclass (e.g. GPT-OSS projections).
    The returned class has a proper ``_owner`` set by ``__init_subclass__``.
    """
    if parent_cls in _float8_grouped_linear_cache:
        return _float8_grouped_linear_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class Float8GroupedLinear(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            pass

        def __init__(self, config: Config):
            super().__init__(config)
            from torchao.prototype.moe_training.config import Float8TrainingOpConfig

            self._float8_op_config = Float8TrainingOpConfig()

        def _grouped_mm(self, *, input_RI, weight_EOI, offsets_E):
            from torchao.prototype.moe_training.utils import (
                _quantize_then_scaled_grouped_mm,
            )

            return _quantize_then_scaled_grouped_mm(
                input_RI,
                weight_EOI.bfloat16().transpose(-2, -1),
                config=self._float8_op_config,
                offs=offsets_E,
            )

    Float8GroupedLinear.__name__ = f"Float8{parent_cls.__name__}"
    Float8GroupedLinear.__qualname__ = f"Float8{parent_cls.__name__}"
    _float8_grouped_linear_cache[parent_cls] = Float8GroupedLinear
    return Float8GroupedLinear
