# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [Note] Getting the 'torchao' package:
# This script requires the 'torchao' package to function correctly.
# Please ensure you have this package installed from the appropriate repository.
# You can obtain it from https://github.com/pytorch/ao by following the
# installation instructions.

# Note: Performance
# The quantization modules are intended to be ran under `torch.compile`` for competitive performance

from dataclasses import dataclass, field
from functools import partial
from importlib.util import find_spec

import torch
import torch._inductor.config
import torch.nn as nn
from torchtitan.models.common.linear import Linear
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .module_utils import inject_module_protocol
from .utils import module_filter_fn

# Mapping from quantization type to the pad_multiple needed for grouped GEMMs.
# FP8: 16 byte alignment / 1 byte per elem = 16 elements.
# MXFP8: scaling block size is (1 x 32), so contracting dim must be divisible by 32.
PAD_MULTIPLE_MAP: dict[str, int] = {
    "float8": 16,
    "mxfp8": 32,
}

AUTO_FILTER_SMALL_KN_FLAG = "auto_filter_small_kn"


@dataclass
class QuantizationConfig:
    """Encapsulates all quantization settings for a model.

    Pass to ``model_registry(quantization=...)`` to apply quantization
    at config construction time.
    """

    float8_recipe: str | None = None
    """Float8 recipe for linear layers ("rowwise", "tensorwise", "rowwise_with_gw_hp")."""

    float8_filter_fqns: list[str] | None = None
    """FQNs of linear modules to skip float8 conversion."""

    float8_emulate: bool = False
    """Use software emulation for float8 (test only, no SM89 required)."""

    float8_moe_fqns: list[str] | None = None
    """FQNs of MoE modules to apply FP8 grouped GEMMs (e.g. ["experts"])."""

    mxfp8_fqns: list[str] | None = None
    """FQNs of MoE modules to apply MXFP8 grouped GEMMs (e.g. ["experts"])."""

    mxfp8_recipe: str = "mxfp8_rceil"
    """MXFP8 recipe name."""

    mxfp8_pad_token_groups: bool = True
    """If True, TorchAO pads token groups for MXFP8. False when EP handles padding."""

    def apply(self, model_config) -> None:
        """Apply all configured quantization to the model config."""
        if self.float8_recipe is not None:
            convert_to_float8(
                model_config,
                recipe_name=self.float8_recipe,
                filter_fqns=self.float8_filter_fqns,
                emulate=self.float8_emulate,
            )
        if self.float8_moe_fqns is not None:
            convert_moe_to_float8(
                model_config,
                fqns=self.float8_moe_fqns,
            )
        if self.mxfp8_fqns is not None:
            convert_moe_to_mxfp8(
                model_config,
                fqns=self.mxfp8_fqns,
                recipe_name=self.mxfp8_recipe,
                pad_token_groups_for_grouped_mm=self.mxfp8_pad_token_groups,
            )

    @property
    def has_quantization(self) -> bool:
        return (
            self.float8_recipe is not None
            or self.float8_moe_fqns is not None
            or self.mxfp8_fqns is not None
        )


def convert_to_float8(
    model_config,
    *,
    recipe_name: str = "rowwise",
    filter_fqns: list[str] | None = None,
    emulate: bool = False,
) -> None:
    """Convert Linear.Config instances in the model config to produce Float8Linear.

    Walks the model config tree and sets ``_convert_fn`` on matching
    ``Linear.Config`` instances so that ``build()`` produces
    ``Float8Linear`` modules directly.

    Args:
        model_config: The model config to walk.
        recipe_name: Float8 recipe name ("tensorwise", "rowwise", "rowwise_with_gw_hp").
        filter_fqns: FQNs of modules to skip. Dims not divisible by 16 are always skipped.
        emulate: If True, use software emulation instead of hardware float8.
    """
    if filter_fqns is None:
        filter_fqns = []

    if not has_cuda_capability(8, 9) and not emulate:
        raise ValueError(
            "Float8 is only supported on SM89 or later. "
            "To enable testing on older hardware, set emulate=True."
        )

    try:
        from torchao.float8 import Float8LinearConfig
    except ImportError as e:
        raise ImportError(
            "torchao is not installed. Please install it to use float8 linear layers."
        ) from e

    if not hasattr(Float8LinearConfig, "from_recipe_name"):
        logger.warning(
            "Failed to use Float8 with recipe lookup because the torchao version "
            "is too old, please install torchao v0.9.0 or later and try again",
        )
        return

    torchao_config = Float8LinearConfig.from_recipe_name(recipe_name)
    logger.info(f"Float8 training active with recipe {recipe_name}")

    # short-term solution for https://github.com/pytorch/pytorch/issues/150859
    if recipe_name == "rowwise":
        torch._inductor.config.emulate_precision_casts = True
        logger.debug("Set torch._inductor.config.emulate_precision_casts to True")

    # Build filter function
    clean_fqns = [f for f in filter_fqns if f != AUTO_FILTER_SMALL_KN_FLAG]
    use_auto_filter = AUTO_FILTER_SMALL_KN_FLAG in filter_fqns
    if use_auto_filter:
        try:
            from torchao.float8 import _auto_filter_for_recipe

            logger.info(
                "Using _auto_filter_for_recipe to avoid converting linear layers with dims too small "
                "to benefit from float8 training. See docs/float8.md for more info."
            )
            filter_fn = _auto_filter_for_recipe(
                recipe_name, filter_fqns=clean_fqns
            )
        except ImportError:
            logger.warning(
                "Using default module_filter_fn for float8 model conversion. "
                "To use _auto_filter_for_recipe, please install torchao nightly build."
            )
            filter_fn = partial(module_filter_fn, filter_fqns=clean_fqns)
    else:
        filter_fn = partial(module_filter_fn, filter_fqns=clean_fqns)

    # Walk config tree and set _convert_fn
    from torchao.float8.float8_linear import Float8Linear

    def convert_fn(mod: nn.Module) -> nn.Module:
        converted = Float8Linear.from_float(mod, config=torchao_config)
        if not isinstance(converted, Linear):
            inject_module_protocol(converted, Linear)
        return converted

    for fqn, linear_config in model_config.walk(Linear.Config):
        if filter_fn(linear_config, fqn):
            linear_config._convert_fn = convert_fn

    logger.info("Swapped to Float8Linear layers")


def convert_moe_to_float8(
    model_config,
    *,
    fqns: list[str],
) -> None:
    """Convert MoE expert layers in the model config to use float8 scaled grouped GEMMs.

    Walks the model config tree and sets ``_convert_fn`` on modules
    matching the target FQNs so that ``build()`` applies dynamic float8
    rowwise quantization + scaled grouped GEMMs.

    Args:
        model_config: The model config to walk.
        fqns: FQNs of MoE modules to convert (e.g. ["experts"]).
    """
    if not has_cuda_capability(8, 9):
        raise ValueError("Float8 MoE training only supported on SM89 or later.")

    from torchao.quantization.quant_api import quantize_

    try:
        from torchao.prototype.moe_training.config import Float8TrainingOpConfig
    except ImportError as e:
        raise ImportError(
            "torchao installation does not have MoE training support. Please install torchao nightly build."
        ) from e

    from torchtitan.protocols.module import Module

    def convert_fn(mod: nn.Module) -> nn.Module:
        config = Float8TrainingOpConfig()
        quantize_(mod, config=config)
        return mod

    for fqn, config in model_config.walk(Module.Config):
        if any(target_fqn in fqn for target_fqn in fqns):
            config._convert_fn = convert_fn

    logger.info(
        f"Converted MoE layers matching FQNS {fqns} "
        "to use dynamic float8 rowwise quantization with scaled grouped GEMMs"
    )


def convert_moe_to_mxfp8(
    model_config,
    *,
    fqns: list[str],
    recipe_name: str = "mxfp8_rceil",
    pad_token_groups_for_grouped_mm: bool = True,
) -> None:
    """Convert MoE layers in the model config to use MXFP8 scaled grouped GEMMs.

    Walks the model config tree and sets ``_convert_fn`` on modules
    matching the target FQNs so that ``build()`` applies dynamic MXFP8
    quantization for grouped GEMMs.

    Args:
        model_config: The model config to walk.
        fqns: FQNs of MoE modules to convert (e.g. ["experts"]).
        recipe_name: MXFP8 recipe name (default "mxfp8_rceil").
        pad_token_groups_for_grouped_mm: If True, TorchAO pads token groups.
            Set to False when EP is enabled (TorchTitan handles padding instead).
    """
    if find_spec("torchao") is None:
        raise ImportError(
            "torchao is not installed. Please install it to use MXFP8 linear layers."
        )

    assert has_cuda_capability(
        10, 0
    ), "MXFP8 is only supported on SM100 or architectures"

    from torchao.prototype.moe_training.config import (
        MXFP8TrainingOpConfig,
        MXFP8TrainingRecipe,
    )
    from torchao.quantization.quant_api import quantize_
    from torchtitan.protocols.module import Module

    def convert_fn(mod: nn.Module) -> nn.Module:
        recipe = MXFP8TrainingRecipe(recipe_name)
        mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
        mxfp8_op_config.pad_token_groups_for_grouped_mm = (
            pad_token_groups_for_grouped_mm
        )
        quantize_(mod, config=mxfp8_op_config)
        return mod

    for fqn, config in model_config.walk(Module.Config):
        if any(target_fqn in fqn for target_fqn in fqns):
            config._convert_fn = convert_fn

    logger.info(
        f"Converted layers matching FQNS {fqns} "
        f"to use dynamic {recipe_name} quantization for grouped_mm and linear ops"
    )
