# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from functools import partial
from typing import ClassVar, Literal

import torch
import torch._inductor.config
import torch.nn as nn
from torchtitan.components.quantization import FP8_GROUP_ALIGNMENT_SIZE

from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.moe.utils import set_token_group_alignment_size_m
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import module_filter_fn

AUTO_FILTER_SMALL_KN_FLAG = "auto_filter_small_kn"


class Float8LinearConverter(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        _quantization_type: ClassVar[str] = "float8"

        enable_fsdp_float8_all_gather: bool = False
        """Whether enable float8 all-gather in FSDP, recommended for tensorwise scaling"""

        precompute_float8_dynamic_scale_for_fsdp: bool = False
        """Whether precompute float8 scales dynamically for FSDP, recommended for tensorwise scaling"""

        recipe_name: Literal[
            "tensorwise", "rowwise", "rowwise_with_gw_hp"
        ] | None = None
        """If specified, creates float8 config from recipe name"""

        filter_fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to skip applying float8 training to.
        nn.Linear modules with any dim size not divisible by 16 are always skipped due to hardware requirements.
        Example: filter_fqns=["attention.wq", "attention.wk", "attention.wv", "output"]
        """

        emulate: bool = False
        """
        If True, emulation is used instead of hardware accelerated gemm. This is for test purpose only,
        as the current CI does not have sm_89 capability, required by Float8.
        Not compatible with torch.compile.
        """

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ):
        self.enabled = False
        float8_config = config

        if has_cuda_capability(8, 9) or (
            float8_config.emulate and not model_compile_enabled
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

        if float8_config.recipe_name is not None and not hasattr(
            TorchAOFloat8LinearConfig, "from_recipe_name"
        ):
            logger.warning(
                "Failed to swap to Float8Linear with recipe lookup because the torchao version "
                "is too old, please install torchao v0.9.0 or later and try again",
            )
            return

        self.filter_fqns = float8_config.filter_fqns
        self.filter_fn = self._init_filter_fn(float8_config)

        if float8_config.recipe_name is not None:
            assert not float8_config.enable_fsdp_float8_all_gather, (
                "using `float8_config.enable_fsdp_float8_all_gather` together "
                "with `float8_config.recipe_name` is not supported"
            )

            self.torchao_config = TorchAOFloat8LinearConfig.from_recipe_name(
                float8_config.recipe_name
            )
            self.precompute_scale = False
            logger.info(
                f"Float8 training active with recipe {float8_config.recipe_name}"
            )

            # short-term solution for https://github.com/pytorch/pytorch/issues/150859
            if float8_config.recipe_name == "rowwise":
                torch._inductor.config.emulate_precision_casts = True
                logger.debug(
                    "Set torch._inductor.config.emulate_precision_casts to True"
                )
        else:
            # Mutates the model inplace replacing instances of nn.Linear with Float8Linear
            enable_fsdp_float8_all_gather = (
                parallel_dims.dp_shard_enabled
                and float8_config.enable_fsdp_float8_all_gather
            )
            self.torchao_config = TorchAOFloat8LinearConfig(
                enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
                emulate=float8_config.emulate,
            )
            # for precompute_float8_dynamic_scale_for_fsdp
            self.precompute_scale = (
                enable_fsdp_float8_all_gather
                and float8_config.precompute_float8_dynamic_scale_for_fsdp
            )
            logger.info("Float8 tensorwise scaled training active")

        self.enabled = True

    def _init_filter_fn(self, float8_config: Config):
        # use auto_filter if filter_fqns "auto_filter_small_kn" is one of the given fqns.
        use_auto_filter = AUTO_FILTER_SMALL_KN_FLAG in float8_config.filter_fqns
        if use_auto_filter:
            try:
                from torchao.float8 import _auto_filter_for_recipe

                logger.info(
                    "Using _auto_filter_for_recipe to avoid converting linear layers with dims too small "
                    "to benefit from float8 training. See docs/float8.md for more info."
                )

                recipe_name = (
                    float8_config.recipe_name
                    if float8_config.recipe_name
                    else "tensorwise"
                )

                # remove auto filter flag from filter_fqns before passing to _auto_filter_for_recipe
                float8_config.filter_fqns.remove(AUTO_FILTER_SMALL_KN_FLAG)

                return _auto_filter_for_recipe(
                    recipe_name,
                    filter_fqns=float8_config.filter_fqns,
                )
            except ImportError:
                logger.warning(
                    (
                        "Using default module_filter_fn for float8 model conversion. "
                        "To use _auto_filter_for_recipe, please install torchao nightly build."
                    )
                )

        # use default filter func
        return partial(module_filter_fn, filter_fqns=float8_config.filter_fqns)

    def convert(self, model: nn.Module):
        """
        This function converts the linear layers of `model` to `Float8Linear`.
        Note that today, only dynamic tensor scaling (the default) is supported.
        This will mutate the model inplace.
        """
        if not self.enabled:
            return

        from torchao.float8 import convert_to_float8_training

        # Mutates the model inplace replacing instances of nn.Linear with Float8Linear
        convert_to_float8_training(
            model,
            config=self.torchao_config,
            module_filter_fn=self.filter_fn,
        )
        logger.info(
            "Swapped to Float8Linear layers with enable_fsdp_float8_all_gather="
            f"{self.torchao_config.enable_fsdp_float8_all_gather}"
        )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        if not self.enabled:
            return

        if not self.precompute_scale:
            return

        from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            precompute_float8_dynamic_scale_for_fsdp(m)


class Float8GroupedMMConverter(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        _quantization_type: ClassVar[str] = "float8"

        fqns: list[str] | str = field(default_factory=list)
        """
        *Prototype feature, performance optimization still in progress*
        Comma-separated list of fully qualified names of MoE Layers to apply FP8 dynamic quantization
        on grouped GEMM operations.
        This is a prototype feature that requires the torchao nightly build.
        Example: fqns=["experts"]
        """

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ):
        self.fqns = config.fqns
        if not has_cuda_capability(8, 9):
            raise ValueError("Float8 MoE training only supported on SM89 or later.")

        if not model_compile_enabled:
            logger.warning(
                "Compile is required for high performance float8 MoE training; enable it with --compile.enable"
            )

        # Validate MoE training prototype limitations.
        assert (
            not parallel_dims.pp_enabled
        ), "Float8 MoE training prototype does not yet support pipeline parallelism"
        assert (
            not parallel_dims.cp_enabled
        ), "Float8 MoE training prototype does not yet support context parallelism"

        # For fp8 grouped GEMM, token group sizes must be multiples of 16
        # (16 byte alignment / 1 byte per elem = 16 elements)
        set_token_group_alignment_size_m(FP8_GROUP_ALIGNMENT_SIZE)
        self.enabled = True

    def convert(self, model: nn.Module):
        """
        Mutates the model inplace replacing instances of nn.Parameter with ScaledGroupedMMTensor,
        to perform dynamic float8 rowwise quantization + scaled grouped GEMMs for the target MoE FQNs.
        """
        from torchao.quantization.quant_api import quantize_

        try:
            from torchao.prototype.moe_training.conversion_utils import (
                MoETrainingConfig,
            )
        except ImportError as e:
            raise ImportError(
                "torchao installation does not have MoE training support. Please install torchao nightly build."
            ) from e

        def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
            for target_fqn in self.fqns:
                if target_fqn in cur_fqn:
                    return True
            return False

        config = MoETrainingConfig()
        quantize_(model, config=config, filter_fn=moe_module_filter_fn)
        logger.info(
            f"Converted MoE layers matching FQNS {self.fqns} "
            "to use dynamic float8 rowwise quantization with scaled grouped GEMMs"
        )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        pass


def find_float8_linear_config(
    converters: list,
) -> Float8LinearConverter.Config | None:
    """Find the Float8LinearConverter.Config in a list of converter configs, if any."""
    return next(
        (c for c in converters if isinstance(c, Float8LinearConverter.Config)),
        None,
    )
