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

from dataclasses import dataclass
from functools import partial
from importlib.util import find_spec

import torch
import torch._inductor.config
import torch.nn as nn
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
    TorchAOTokenDispatcher,
)
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .module_utils import inject_module_protocol
from .utils import module_filter_fn


@dataclass(kw_only=True, slots=True)
class Float8LinearConfig(Linear.Config):
    """Drop-in replacement for Linear.Config that builds Float8Linear directly."""

    _torchao_config: object = None

    def build(self, **kwargs):
        from torchao.float8.float8_linear import Float8Linear

        instance = Float8Linear(
            self.in_features,
            self.out_features,
            bias=self.bias,
            config=self._torchao_config,
        )
        if not isinstance(instance, Linear):
            inject_module_protocol(instance, Linear)
        if self.param_init is not None:
            instance._param_init = self.param_init
        return instance


@dataclass
class QuantizationConfig:
    """Base class for quantization config passes.

    Subclasses implement ``apply()`` to transform the model config tree.
    Pass a list of these to ``model_registry(quantization=[...])``.
    """

    model_compile_enabled: bool = False
    """Whether torch.compile is enabled for the model."""

    def apply(self, model_config) -> None:
        raise NotImplementedError


@dataclass
class Float8LinearQuant(QuantizationConfig):
    """Replace matching Linear.Config with Float8LinearConfig."""

    recipe_name: str = "rowwise"
    """Float8 recipe name ("rowwise", "rowwise_with_gw_hp")."""

    filter_fqns: list[str] | None = None
    """
    List of fully qualified names of modules to skip applying float8 training to.
    nn.Linear modules with any dim size not divisible by 16 are always skipped due to hardware requirements.
    Example: filter_fqns=["attention.qkv_linear.wq", "attention.qkv_linear.wk", "attention.qkv_linear.wv", "output"]
    """

    emulate: bool = False
    """
    If True, emulation is used instead of hardware accelerated gemm. This is for test purpose only,
    as the current CI does not have sm_89 capability, required by Float8.
    Not compatible with torch.compile.
    """

    def apply(self, model_config) -> None:
        convert_to_float8(
            model_config,
            recipe_name=self.recipe_name,
            filter_fqns=self.filter_fqns,
            emulate=self.emulate,
            model_compile_enabled=self.model_compile_enabled,
        )


@dataclass
class Float8MoEQuant(QuantizationConfig):
    """Apply FP8 quantization to MoE expert grouped GEMMs."""

    def apply(self, model_config) -> None:
        convert_moe_to_float8(
            model_config,
            model_compile_enabled=self.model_compile_enabled,
        )


@dataclass
class MXFP8MoEQuant(QuantizationConfig):
    """Apply MXFP8 quantization to MoE expert grouped GEMMs."""

    recipe_name: str = "mxfp8_rceil"
    """
    Quantization recipe name for grouped GEMMs. Options: ["mxfp8_rceil"]

    - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode when computing the e8m0 scale factors.
    """

    pad_token_groups: bool = True
    """If True, TorchAO pads token groups for MXFP8 grouped GEMM.
    Set to False when EP is enabled (TorchTitan handles padding instead,
    except for DeepEP backend)."""

    def apply(self, model_config) -> None:
        convert_moe_to_mxfp8(
            model_config,
            recipe_name=self.recipe_name,
            pad_token_groups_for_grouped_mm=self.pad_token_groups,
            model_compile_enabled=self.model_compile_enabled,
        )


@dataclass
class MXFP8Quant(QuantizationConfig):
    """Apply MXFP8 quantization to modules matching FQNs (e.g. Flux blocks)."""

    fqns: list[str] | None = None
    """List of fully qualified names of modules to apply MXFP8 dynamic quantization
    on grouped GEMM operations."""

    recipe_name: str = "mxfp8_rceil"
    """
    Quantization recipe name for grouped GEMMs. Options: ["mxfp8_rceil"]

    - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode when computing the e8m0 scale factors.
    """

    pad_token_groups: bool = True
    """If True, TorchAO pads token groups for MXFP8 grouped GEMM.
    Set to False when EP is enabled (TorchTitan handles padding instead,
    except for DeepEP backend)."""

    def apply(self, model_config) -> None:
        convert_to_mxfp8(
            model_config,
            fqns=self.fqns,
            recipe_name=self.recipe_name,
            pad_token_groups_for_grouped_mm=self.pad_token_groups,
            model_compile_enabled=self.model_compile_enabled,
        )


def has_quantization(model_config) -> bool:
    """Check if any module in the model config has quantization applied."""
    has_float8_linear = any(
        isinstance(lc, Float8LinearConfig)
        for _fqn, lc, _parent, _attr in model_config.walk(Linear.Config)
    )
    has_moe_quant = any(
        config._convert_fn is not None
        for _fqn, config, _parent, _attr in model_config.walk(GroupedExperts.Config)
    )
    return has_float8_linear or has_moe_quant


def convert_to_float8(
    model_config,
    *,
    recipe_name: str = "rowwise",
    filter_fqns: list[str] | None = None,
    emulate: bool = False,
    model_compile_enabled: bool = False,
) -> None:
    """Replace matching Linear.Config with Float8LinearConfig in the model config tree.

    Args:
        model_config: The model config to walk.
        recipe_name: Float8 recipe name ("rowwise", "rowwise_with_gw_hp").
        filter_fqns: FQNs of modules to skip. Dims not divisible by 16 are always skipped.
        emulate: If True, use software emulation instead of hardware float8.
        model_compile_enabled: Whether torch.compile is enabled for the model.
    """
    if filter_fqns is None:
        filter_fqns = []

    if has_cuda_capability(8, 9) or (emulate and not model_compile_enabled):
        pass
    else:
        raise ValueError(
            "Failed to swap to Float8Linear because float8 is only supported on SM89 or later."
            "To enable testing on older hardware, set `float8.emulate` to True in eager mode.",
        )

    try:
        from torchao.float8 import Float8LinearConfig as TorchAOFloat8LinearConfig
    except ImportError as e:
        raise ImportError(
            "torchao is not installed. Please install it to use float8 linear layers."
        ) from e

    if not hasattr(TorchAOFloat8LinearConfig, "from_recipe_name"):
        logger.warning(
            "Failed to use Float8 with recipe lookup because the torchao version "
            "is too old, please install torchao v0.9.0 or later and try again",
        )
        return

    torchao_config = TorchAOFloat8LinearConfig.from_recipe_name(recipe_name)
    logger.info(f"Float8 training active with recipe {recipe_name}")

    # short-term solution for https://github.com/pytorch/pytorch/issues/150859
    if recipe_name == "rowwise":
        torch._inductor.config.emulate_precision_casts = True
        logger.debug("Set torch._inductor.config.emulate_precision_casts to True")

    # Build filter function
    clean_fqns = [f for f in filter_fqns if f != "auto_filter_small_kn"]
    use_auto_filter = "auto_filter_small_kn" in filter_fqns
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

    # Walk config tree and swap Linear.Config → Float8LinearConfig
    for fqn, linear_config, parent, attr in model_config.walk(Linear.Config):
        if filter_fn(linear_config, fqn):
            new_config = Float8LinearConfig(
                in_features=linear_config.in_features,
                out_features=linear_config.out_features,
                bias=linear_config.bias,
                param_init=linear_config.param_init,
                _torchao_config=torchao_config,
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

    logger.info("Swapped to Float8Linear layers")


def _swap_token_dispatcher(config, pad_multiple: int) -> None:
    """Swap token dispatcher config to support padded grouped GEMMs."""
    td = config.token_dispatcher
    if isinstance(td, AllToAllTokenDispatcher.Config) and not isinstance(
        td, TorchAOTokenDispatcher.Config
    ):
        config.token_dispatcher = TorchAOTokenDispatcher.Config(
            num_experts=td.num_experts,
            top_k=td.top_k,
            score_before_experts=td.score_before_experts,
            pad_multiple=pad_multiple,
        )
    elif isinstance(td, DeepEPTokenDispatcher.Config):
        if td.comm_backend == "deepep":
            raise ValueError(
                "DeepEP does not support pad_multiple. "
                "Use hybridep or standard comm backend instead."
            )
        config.token_dispatcher = DeepEPTokenDispatcher.Config(
            num_experts=td.num_experts,
            top_k=td.top_k,
            score_before_experts=td.score_before_experts,
            comm_backend=td.comm_backend,
            non_blocking_capacity_factor=td.non_blocking_capacity_factor,
            pad_multiple=pad_multiple,
        )


def convert_moe_to_float8(
    model_config,
    *,
    model_compile_enabled: bool = False,
) -> None:
    """Convert MoE expert layers in the model config to use float8 scaled grouped GEMMs.

    Walks the model config tree for GroupedExperts.Config instances and sets
    ``_convert_fn`` so that ``build()`` applies dynamic float8 rowwise
    quantization + scaled grouped GEMMs. Also swaps token dispatcher configs
    to set the appropriate pad_multiple for FP8 grouped GEMMs.
    """
    if not has_cuda_capability(8, 9):
        raise ValueError("Float8 MoE training only supported on SM89 or later.")

    if not model_compile_enabled:
        logger.warning(
            "Compile is required for high performance float8 MoE training; "
            "enable it with --compile.enable"
        )

    from torchao.quantization.quant_api import quantize_

    try:
        from torchao.prototype.moe_training.config import Float8TrainingOpConfig
    except ImportError as e:
        raise ImportError(
            "torchao installation does not have MoE training support. Please install torchao nightly build."
        ) from e

    # FP8: 16 byte alignment / 1 byte per elem = 16 elements.
    pad_multiple = 16

    def convert_fn(mod: nn.Module) -> nn.Module:
        config = Float8TrainingOpConfig()
        quantize_(mod, config=config)
        return mod

    for fqn, config, _parent, _attr in model_config.walk(GroupedExperts.Config):
        config._convert_fn = convert_fn
        _swap_token_dispatcher(config, pad_multiple)

    logger.info(
        "Converted GroupedExperts to use dynamic float8 rowwise quantization "
        "with scaled grouped GEMMs"
    )


def _make_mxfp8_convert_fn(recipe_name, pad_token_groups_for_grouped_mm):
    """Create a _convert_fn closure for MXFP8 quantization."""
    from torchao.prototype.moe_training.config import (
        MXFP8TrainingOpConfig,
        MXFP8TrainingRecipe,
    )
    from torchao.quantization.quant_api import quantize_

    def convert_fn(mod: nn.Module) -> nn.Module:
        recipe = MXFP8TrainingRecipe(recipe_name)
        mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
        mxfp8_op_config.pad_token_groups_for_grouped_mm = (
            pad_token_groups_for_grouped_mm
        )
        quantize_(mod, config=mxfp8_op_config)
        return mod

    return convert_fn


def convert_moe_to_mxfp8(
    model_config,
    *,
    recipe_name: str = "mxfp8_rceil",
    pad_token_groups_for_grouped_mm: bool = True,
    model_compile_enabled: bool = False,
) -> None:
    """Convert GroupedExperts in the model config to use MXFP8 scaled grouped GEMMs.

    Also swaps token dispatcher configs to set the appropriate pad_multiple
    for MXFP8 grouped GEMMs.
    """
    if find_spec("torchao") is None:
        raise ImportError(
            "torchao is not installed. Please install it to use MXFP8 linear layers."
        )
    assert has_cuda_capability(
        10, 0
    ), "MXFP8 is only supported on SM100 or architectures"

    if not model_compile_enabled:
        logger.warning(
            "torch.compile enablement is required for highest performance "
            "of MXFP8 dynamic quantization."
        )
    convert_fn = _make_mxfp8_convert_fn(recipe_name, pad_token_groups_for_grouped_mm)

    # MXFP8: scaling block size is (1 x 32), so contracting dim must be divisible by 32.
    pad_multiple = 32

    for _fqn, config, _parent, _attr in model_config.walk(GroupedExperts.Config):
        config._convert_fn = convert_fn
        _swap_token_dispatcher(config, pad_multiple)

    logger.info(
        f"Converted GroupedExperts to use dynamic {recipe_name} quantization "
        "for grouped_mm and linear ops"
    )


def convert_to_mxfp8(
    model_config,
    *,
    fqns: list[str] | None = None,
    recipe_name: str = "mxfp8_rceil",
    pad_token_groups_for_grouped_mm: bool = True,
    model_compile_enabled: bool = False,
) -> None:
    """Convert modules matching FQNs to use MXFP8 quantization (e.g. Flux blocks)."""
    if find_spec("torchao") is None:
        raise ImportError(
            "torchao is not installed. Please install it to use MXFP8 linear layers."
        )
    assert has_cuda_capability(
        10, 0
    ), "MXFP8 is only supported on SM100 or architectures"

    if not model_compile_enabled:
        logger.warning(
            "torch.compile enablement is required for highest performance "
            "of MXFP8 dynamic quantization."
        )
    convert_fn = _make_mxfp8_convert_fn(recipe_name, pad_token_groups_for_grouped_mm)

    for fqn, config, _parent, _attr in model_config.walk(Module.Config):
        if fqns is None or any(target_fqn in fqn for target_fqn in fqns):
            config._convert_fn = convert_fn

    logger.info(
        f"Converted modules to use dynamic {recipe_name} quantization "
        "for grouped_mm and linear ops"
    )
