# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 grouped-linear building blocks."""

from dataclasses import dataclass

_mxfp8_grouped_linear_cache: dict[type, type] = {}


def _get_mxfp8_grouped_linear_cls(parent_cls: type) -> type:
    """Get or create an MXFP8-quantized subclass of *parent_cls*.

    Works for any ``GroupedLinear`` subclass. The returned class has a proper
    ``_owner`` set by ``__init_subclass__``.

    The subclass overrides ``_grouped_mm`` to call torchao's
    ``_quantize_then_scaled_grouped_mm``.
    """
    if parent_cls in _mxfp8_grouped_linear_cache:
        return _mxfp8_grouped_linear_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class MXFP8GroupedLinear(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            recipe_name: str = "mxfp8_rceil"

        def __init__(self, config: Config):
            super().__init__(config)
            from torchao.prototype.moe_training.config import (
                MXFP8TrainingOpConfig,
                MXFP8TrainingRecipe,
            )

            recipe = MXFP8TrainingRecipe(config.recipe_name)
            self._mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)

        def _grouped_mm(self, *, input_RI, weight_EOI, offsets_E):
            from torchao.prototype.moe_training.utils import (
                _quantize_then_scaled_grouped_mm,
            )

            return _quantize_then_scaled_grouped_mm(
                input_RI,
                weight_EOI.bfloat16().transpose(-2, -1),
                config=self._mxfp8_op_config,
                offs=offsets_E,
            )

    MXFP8GroupedLinear.__name__ = f"MXFP8{parent_cls.__name__}"
    MXFP8GroupedLinear.__qualname__ = f"MXFP8{parent_cls.__name__}"
    _mxfp8_grouped_linear_cache[parent_cls] = MXFP8GroupedLinear
    return MXFP8GroupedLinear
