# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, fields
from functools import partial
from importlib.util import find_spec
from typing import Literal

import torch
import torch._inductor.config
from torchtitan.components.quantization import QuantizationConverter
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import module_filter_fn, swap_token_dispatcher


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


class Float8LinearConverter(QuantizationConverter):
    """Replace matching Linear.Config with Float8Linear.Config."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal["rowwise", "rowwise_with_gw_hp"] = "rowwise"
        """Float8 recipe name."""

        filter_fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to skip applying float8 training to.
        nn.Linear modules with any dim size not divisible by 16 are always skipped
        due to hardware requirements.
        """

        emulate: bool = False
        """
        If True, emulation is used instead of hardware accelerated gemm.
        This is for test purpose only. Not compatible with torch.compile.
        """

    def __init__(self, config: Config):
        self.config = config

        if Float8Linear is None:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 linear layers."
            )

        cfg = self.config
        filter_fqns = cfg.filter_fqns

        if has_cuda_capability(8, 9) or (cfg.emulate and not cfg.model_compile_enabled):
            pass
        else:
            raise ValueError(
                "Failed to swap to Float8Linear because float8 is only supported on SM89 or later. "
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
            self.enabled = False
            return

        self.torchao_config = TorchAOFloat8LinearConfig.from_recipe_name(
            cfg.recipe_name
        )
        if cfg.emulate:
            self.torchao_config = TorchAOFloat8LinearConfig(emulate=True)
        logger.info(f"Float8 training active with recipe {cfg.recipe_name}")

        # short-term solution for https://github.com/pytorch/pytorch/issues/150859
        if cfg.recipe_name == "rowwise":
            torch._inductor.config.emulate_precision_casts = True
            logger.debug("Set torch._inductor.config.emulate_precision_casts to True")

        # Build filter function
        clean_fqns = [f for f in filter_fqns if f != "auto_filter_small_kn"]
        use_auto_filter = "auto_filter_small_kn" in filter_fqns
        if use_auto_filter:
            try:
                from torchao.float8 import _auto_filter_for_recipe

                logger.info(
                    "Using _auto_filter_for_recipe to avoid converting linear layers "
                    "with dims too small to benefit from float8 training. "
                    "See torchtitan/components/quantization/float8.md for more info."
                )
                self.filter_fn = _auto_filter_for_recipe(
                    cfg.recipe_name, filter_fqns=clean_fqns
                )
            except ImportError:
                logger.warning(
                    "Using default module_filter_fn for float8 model conversion. "
                    "To use _auto_filter_for_recipe, please install torchao nightly build."
                )
                self.filter_fn = partial(module_filter_fn, filter_fqns=clean_fqns)
        else:
            self.filter_fn = partial(module_filter_fn, filter_fqns=clean_fqns)

        self.enabled = True

    def convert(self, model_config) -> None:
        if not self.enabled:
            return

        assert Float8Linear is not None
        for fqn, linear_config, parent, attr in model_config.traverse(Linear.Config):
            if self.filter_fn(linear_config, fqn):
                new_config = Float8Linear.Config(
                    in_features=linear_config.in_features,
                    out_features=linear_config.out_features,
                    bias=linear_config.bias,
                    param_init=linear_config.param_init,
                    _torchao_config=self.torchao_config,
                )
                if isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info("Swapped to Float8Linear layers")


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
            from torchao.quantization.quant_api import quantize_

            quantize_(
                self,
                config=Float8TrainingOpConfig(),
                filter_fn=lambda mod, _fqn: isinstance(mod, GroupedExperts),
            )

    Float8GroupedExperts.__name__ = f"Float8{parent_cls.__name__}"
    Float8GroupedExperts.__qualname__ = f"Float8{parent_cls.__name__}"
    _float8_experts_cache[parent_cls] = Float8GroupedExperts
    return Float8GroupedExperts


class Float8GroupedExpertsConverter(QuantizationConverter):
    """Apply FP8 quantization to MoE expert grouped GEMMs."""

    # FP8: 16 byte alignment / 1 byte per elem = 16 elements.
    PAD_MULTIPLE = 16

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        pass

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 MoE training."
            )

        if not has_cuda_capability(8, 9):
            raise ValueError("Float8 MoE training only supported on SM89 or later.")

        if not self.config.model_compile_enabled:
            logger.warning(
                "Compile is required for high performance float8 MoE training; "
                "enable it with --compile.enable"
            )

    def convert(self, model_config) -> None:
        for _fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            swap_token_dispatcher(config, self.PAD_MULTIPLE)
            base_module_cls = type(config)._owner
            quantized_cls = _get_float8_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            "Converted GroupedExperts to use dynamic float8 rowwise quantization "
            "with scaled grouped GEMMs"
        )
