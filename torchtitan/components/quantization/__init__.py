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

from dataclasses import dataclass, field, fields
from functools import partial
from importlib.util import find_spec
from typing import ClassVar, Literal

import torch
import torch._inductor.config
from torchtitan.config import Configurable
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import module_filter_fn, swap_token_dispatcher


class Float8Linear:
    """Namespace for Float8-quantized Linear config.

    The diamond-inherited module class (TorchAOFloat8Linear + Linear) is
    created lazily on first build to avoid a top-level torchao import.
    """

    _module_cls: ClassVar[type | None] = None

    @classmethod
    def _get_module_cls(cls):
        if cls._module_cls is None:
            from torchao.float8.float8_linear import (
                Float8Linear as TorchAOFloat8Linear,
            )

            class Float8LinearModule(TorchAOFloat8Linear, Linear):
                def __init__(self, *args, **kwargs):
                    TorchAOFloat8Linear.__init__(self, *args, **kwargs)

            cls._module_cls = Float8LinearModule
        return cls._module_cls

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Drop-in replacement for Linear.Config that builds Float8Linear."""

        _torchao_config: object = None

        def build(self, **kwargs):
            module_cls = Float8Linear._get_module_cls()
            instance = module_cls(
                self.in_features,
                self.out_features,
                bias=self.bias,
                config=self._torchao_config,
            )
            if self.param_init is not None:
                instance._param_init = self.param_init
            return instance


class MXFP8Linear:
    """Namespace for MXFP8-quantized Linear config."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Builds Linear then applies MXFP8 quantization via quantize_()."""

        _recipe_name: str = "mxfp8_rceil"
        _pad_token_groups: bool = True

        def build(self, **kwargs):
            from torchao.prototype.moe_training.config import (
                MXFP8TrainingOpConfig,
                MXFP8TrainingRecipe,
            )
            from torchao.quantization.quant_api import quantize_

            instance = Linear.Config.build(self, **kwargs)
            param_init = getattr(instance, "_param_init", None)
            recipe = MXFP8TrainingRecipe(self._recipe_name)
            mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
            mxfp8_op_config.pad_token_groups_for_grouped_mm = self._pad_token_groups
            quantize_(instance, config=mxfp8_op_config)
            if param_init is not None:
                instance._param_init = param_init
            return instance


class Float8GroupedExperts:
    """Float8-quantized GroupedExperts config factory.

    Dynamically creates Config subclasses that override ``build()`` to apply
    float8 quantization, preserving model-specific Config fields
    (e.g. GptOssGroupedExperts.Config.swiglu_limit).
    """

    _cache: ClassVar[dict[type, type]] = {}

    @classmethod
    def from_config(cls, config: GroupedExperts.Config) -> GroupedExperts.Config:
        base_cls = type(config)
        if base_cls not in cls._cache:

            @dataclass(kw_only=True, slots=True)
            class Config(base_cls):
                _is_quantized: ClassVar[bool] = True

                def build(self, **kwargs):
                    from torchao.prototype.moe_training.config import (
                        Float8TrainingOpConfig,
                    )
                    from torchao.quantization.quant_api import quantize_

                    instance = base_cls.build(self, **kwargs)
                    param_init = getattr(instance, "_param_init", None)
                    quantize_(instance, config=Float8TrainingOpConfig())
                    if param_init is not None:
                        instance._param_init = param_init
                    return instance

            cls._cache[base_cls] = Config

        ConfigCls = cls._cache[base_cls]
        return ConfigCls(
            **{f.name: getattr(config, f.name) for f in fields(config)}
        )


class MXFP8GroupedExperts:
    """MXFP8-quantized GroupedExperts config factory.

    Dynamically creates Config subclasses that override ``build()`` to apply
    MXFP8 quantization, preserving model-specific Config fields.
    """

    _cache: ClassVar[dict[type, type]] = {}

    @classmethod
    def from_config(
        cls,
        config: GroupedExperts.Config,
        *,
        recipe_name: str = "mxfp8_rceil",
        pad_token_groups: bool = True,
    ) -> GroupedExperts.Config:
        base_cls = type(config)
        if base_cls not in cls._cache:

            @dataclass(kw_only=True, slots=True)
            class Config(base_cls):
                _is_quantized: ClassVar[bool] = True
                _recipe_name: str = "mxfp8_rceil"
                _pad_token_groups: bool = True

                def build(self, **kwargs):
                    from torchao.prototype.moe_training.config import (
                        MXFP8TrainingOpConfig,
                        MXFP8TrainingRecipe,
                    )
                    from torchao.quantization.quant_api import quantize_

                    instance = base_cls.build(self, **kwargs)
                    param_init = getattr(instance, "_param_init", None)
                    recipe = MXFP8TrainingRecipe(self._recipe_name)
                    mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
                    mxfp8_op_config.pad_token_groups_for_grouped_mm = (
                        self._pad_token_groups
                    )
                    quantize_(instance, config=mxfp8_op_config)
                    if param_init is not None:
                        instance._param_init = param_init
                    return instance

            cls._cache[base_cls] = Config

        ConfigCls = cls._cache[base_cls]
        return ConfigCls(
            **{f.name: getattr(config, f.name) for f in fields(config)},
            _recipe_name=recipe_name,
            _pad_token_groups=pad_token_groups,
        )


class QuantizationConverter(Configurable):
    """Base class for quantization converters.

    Subclasses define a nested Config and implement ``convert()``
    to transform the model config tree.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        model_compile_enabled: bool = False
        """Whether torch.compile is enabled for the model."""

    def convert(self, model_config) -> None:
        raise NotImplementedError


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

    def convert(self, model_config) -> None:
        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 linear layers."
            )

        cfg = self.config
        filter_fqns = cfg.filter_fqns

        if has_cuda_capability(8, 9) or (
            cfg.emulate and not cfg.model_compile_enabled
        ):
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

        torchao_config = TorchAOFloat8LinearConfig.from_recipe_name(cfg.recipe_name)
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
                    "See docs/float8.md for more info."
                )
                filter_fn = _auto_filter_for_recipe(
                    cfg.recipe_name, filter_fqns=clean_fqns
                )
            except ImportError:
                logger.warning(
                    "Using default module_filter_fn for float8 model conversion. "
                    "To use _auto_filter_for_recipe, please install torchao nightly build."
                )
                filter_fn = partial(module_filter_fn, filter_fqns=clean_fqns)
        else:
            filter_fn = partial(module_filter_fn, filter_fqns=clean_fqns)

        # Walk config tree and swap Linear.Config → Float8Linear.Config
        for fqn, linear_config, parent, attr in model_config.walk(Linear.Config):
            if filter_fn(linear_config, fqn):
                new_config = Float8Linear.Config(
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


class Float8MoEConverter(QuantizationConverter):
    """Apply FP8 quantization to MoE expert grouped GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        pass

    def __init__(self, config: Config):
        self.config = config

    def convert(self, model_config) -> None:
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

        # FP8: 16 byte alignment / 1 byte per elem = 16 elements.
        pad_multiple = 16

        for _fqn, config, parent, attr in model_config.walk(GroupedExperts.Config):
            swap_token_dispatcher(config, pad_multiple)
            new_config = Float8GroupedExperts.from_config(config)
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            "Converted GroupedExperts to use dynamic float8 rowwise quantization "
            "with scaled grouped GEMMs"
        )


class MXFP8LinearConverter(QuantizationConverter):
    """Apply MXFP8 quantization to modules matching FQNs (e.g. Flux blocks)."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal["mxfp8_rceil"] = "mxfp8_rceil"
        """
        Quantization recipe name for grouped GEMMs. Options: ["mxfp8_rceil"]

        - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode when computing the e8m0 scale factors.
        """

        fqns: list[str] = field(default_factory=list)
        """
        *Prototype feature, performance optimization still in progress*
        Comma-separated list of fully qualified names of MoE modules to apply MXFP8 dynamic quantization
        on grouped GEMM operations.
        This is a prototype feature that requires the torchao nightly build.
        """

        pad_token_groups: bool = True
        """If True, TorchAO pads token groups for MXFP8 linear ops."""

    def __init__(self, config: Config):
        self.config = config

    def convert(self, model_config) -> None:
        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 quantization."
            )

        assert has_cuda_capability(
            10, 0
        ), "MXFP8 is only supported on SM100 or architectures"

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.walk(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                new_config = MXFP8Linear.Config(
                    in_features=config.in_features,
                    out_features=config.out_features,
                    bias=config.bias,
                    param_init=config.param_init,
                    _recipe_name=self.config.recipe_name,
                    _pad_token_groups=self.config.pad_token_groups,
                )
                if isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info(
            f"Converted modules to use dynamic {self.config.recipe_name} "
            "quantization for grouped_mm and linear ops"
        )


class MXFP8MoEConverter(QuantizationConverter):
    """Apply MXFP8 quantization to MoE expert grouped GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal["mxfp8_rceil"] = "mxfp8_rceil"
        """
        Quantization recipe name for grouped GEMMs. Options: ["mxfp8_rceil"]

        - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode when computing the e8m0 scale factors.
        """

    def __init__(self, config: Config):
        self.config = config

    def convert(self, model_config) -> None:
        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 MoE training."
            )

        assert has_cuda_capability(
            10, 0
        ), "MXFP8 is only supported on SM100 or architectures"

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

        # MXFP8: scaling block size is (1 x 32), so contracting dim must be divisible by 32.
        pad_multiple = 32
        for _fqn, config, parent, attr in model_config.walk(GroupedExperts.Config):
            dispatcher_handles_padding = swap_token_dispatcher(config, pad_multiple)
            new_config = MXFP8GroupedExperts.from_config(
                config,
                recipe_name=self.config.recipe_name,
                pad_token_groups=not dispatcher_handles_padding,
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            f"Converted GroupedExperts to use dynamic {self.config.recipe_name} "
            "quantization for grouped_mm and linear ops"
        )
