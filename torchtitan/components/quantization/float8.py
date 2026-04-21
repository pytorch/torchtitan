# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial

import torch
import torch._inductor.config
import torch.nn as nn
from torchtitan.models.common.linear import Linear
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .module_utils import inject_module_protocol
from .utils import module_filter_fn

AUTO_FILTER_SMALL_KN_FLAG = "auto_filter_small_kn"


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
    matching the target FQNs so that ``build()`` replaces instances of
    nn.Parameter with MXFP8TrainingWeightWrapperTensor, to perform dynamic float8
    rowwise quantization + scaled grouped GEMMs.

    Also swaps AllToAllTokenDispatcher -> TorchAOTokenDispatcher and sets
    pad_multiple on DeepEPTokenDispatcher (hybridep) for matching FQNs.

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
