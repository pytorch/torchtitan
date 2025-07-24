# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial

import torch
import torch.nn as nn

from torchtitan.config.job_config import Float8, JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import (
    ModelConverter,
    register_model_converter,
)
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import module_filter_fn

AUTO_FILTER_SMALL_KN_FLAG = "auto_filter_small_kn"


class Float8Converter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.enabled = False

        float8_config: Float8 = job_config.float8
        if has_cuda_capability(8, 9) or (
            float8_config.emulate and not job_config.training.compile
        ):
            pass
        else:
            raise ValueError(
                "Failed to swap to Float8Linear because float8 is only supported on SM89 or later."
                "To enable testing on older hardware, set `float8.emulate` to True in eager mode.",
            )
        try:
            from torchao.float8 import Float8LinearConfig
        except ImportError as e:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 linear layers."
            ) from e

        if float8_config.recipe_name is not None and not hasattr(
            Float8LinearConfig, "from_recipe_name"
        ):
            logger.warning(
                "Failed to swap to Float8Linear with recipe lookup because the torchao version "
                "is too old, please install torchao v0.9.0 or later and try again",
            )
            return

        self.enabled = True
        self.filter_fqns = float8_config.filter_fqns
        self.moe_fqns = float8_config.moe_fqns_prototype
        self.filter_fn = self._init_filter_fn(float8_config)

        # Validate MoE training prototype limitations.
        if self.moe_fqns:
            assert (
                job_config.parallelism.pipeline_parallel_degree == 1
            ), "Float8 MoE training prototype does not yet support pipeline parallelism"
            assert (
                job_config.parallelism.context_parallel_degree == 1
            ), "Float8 MoE training prototype does not yet support context parallelism"

        if float8_config.recipe_name is not None:
            assert not float8_config.enable_fsdp_float8_all_gather, (
                "using `float8_config.enable_fsdp_float8_all_gather` together "
                "with `float8_config.recipe_name` is not supported"
            )

            self.config = Float8LinearConfig.from_recipe_name(float8_config.recipe_name)
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
            self.config = Float8LinearConfig(
                enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
                emulate=float8_config.emulate,
            )
            # for precompute_float8_dynamic_scale_for_fsdp
            self.precompute_scale = (
                enable_fsdp_float8_all_gather
                and float8_config.precompute_float8_dynamic_scale_for_fsdp
            )
            logger.info("Float8 tensorwise scaled training active")

    def _init_filter_fn(self, float8_config: Float8):
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

        # MoE conversion must take place before Float8Linear conversion, otherwise the Float8Linears will
        # be converted back to nn.Linear:
        # https://github.com/pytorch/ao/blob/c2a6568a04075acc371a338206216bb65536fb27/torchao/quantization/quant_api.py#L294-L299
        # TODO: add warning in torchao when this happens, or find a better way to avoid this.
        if self.moe_fqns:
            self._convert_moe_layers(model)

        from torchao.float8 import convert_to_float8_training

        # Mutates the model inplace replacing instances of nn.Linear with Float8Linear
        convert_to_float8_training(
            model,
            config=self.config,
            module_filter_fn=self.filter_fn,
        )
        logger.info(
            "Swapped to Float8Linear layers with enable_fsdp_float8_all_gather="
            f"{self.config.enable_fsdp_float8_all_gather}"
        )

    def _convert_moe_layers(self, model: nn.Module):
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
            for target_fqn in self.moe_fqns:
                if target_fqn in cur_fqn:
                    return True
            return False

        config = MoETrainingConfig()
        quantize_(model, config=config, filter_fn=moe_module_filter_fn)
        logger.info(
            f"Converted MoE layers matching FQNS {self.moe_fqns} "
            "to use dynamic float8 rowwise quantization with scaled grouped GEMMs"
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


register_model_converter(Float8Converter, "float8")
