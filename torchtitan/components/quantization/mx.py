# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from importlib.util import find_spec
from typing import Any, List

import torch.nn as nn
from torchtitan.components.quantization import (
    MXFP8_GROUP_ALIGNMENT_SIZE,
    QuantizationConverter,
)

from torchtitan.config.job_config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.moe.utils import set_token_group_alignment_size_m
from torchtitan.protocols.model_converter import register_model_converter
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import module_filter_fn


class MXLinearConverter(QuantizationConverter):
    """Converts the linear layers of `model` to `MXLinear`."""

    filter_fqns: List[str]
    mx_config: Any  # MXLinearConfig type when imported

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        super().__init__(job_config, parallel_dims)
        # Ensure minimum torchao versions
        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 linear layers."
            )

        # Can be removed if we enable the emulated versions
        assert has_cuda_capability(
            10, 0
        ), "MXFP8 is only supported on SM100 or architectures"

        # TP not yet supported with torch.compile
        model_compile_enabled = (
            job_config.compile.enable and "model" in job_config.compile.components
        )
        assert not (
            model_compile_enabled and job_config.parallelism.tensor_parallel_degree > 1
        ), "TP not yet supported with torch.compile for mxfp8"

        # Configure MXFP8
        from torchao.prototype.mx_formats.config import (
            MXFP8Dim1CastKernelChoice,
            MXLinearConfig as TorchAOMXLinearConfig,
        )

        mx_job_config: TorchAOMXLinearConfig = job_config.quantize.linear.mx
        config = TorchAOMXLinearConfig.from_recipe_name(mx_job_config.recipe_name)
        config.mxfp8_dim1_cast_kernel_choice = MXFP8Dim1CastKernelChoice[
            mx_job_config.mxfp8_dim1_cast_kernel_choice.upper()
        ]
        self.filter_fqns = mx_job_config.filter_fqns
        self.config = config
        self.enabled = True
        logger.info(f"MX training active with recipe {mx_job_config.recipe_name}")

    def convert(self, model: nn.Module):
        """
        Converts the linear layers of `model` to `MXLinear`.
        Note that today, only dynamic tensor scaling (the default) is supported.
        This will mutate the model inplace.
        """
        if not self.enabled:
            return

        from torchao.prototype.mx_formats.config import (
            MXLinearConfig as TorchAOMXLinearConfig,
        )
        from torchao.quantization import quantize_

        assert isinstance(self.config, TorchAOMXLinearConfig)
        quantize_(
            model,
            config=self.config,
            filter_fn=partial(module_filter_fn, filter_fqns=self.filter_fqns),
        )
        logger.info("Swapped to MXLinear layers")

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        """
        MXFP8 doesn't require any post-optimizer hooks at the moment
        """
        return


class MXGroupedMMConverter(QuantizationConverter):
    """Converts target 3D nn.Parameters of a model, representing 'experts',
    to use MXFP8 scaled grouped GEMMs instead of a high precision grouped GEMMs."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        super().__init__(job_config, parallel_dims)
        # Ensure minimum torchao versions
        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 linear layers."
            )

        # Can be removed if we enable the emulated versions
        assert has_cuda_capability(
            10, 0
        ), "MXFP8 is only supported on SM100 or architectures"

        # Warn user if torch.compile is not enabled
        model_compile_enabled = (
            job_config.compile.enable and "model" in job_config.compile.components
        )
        if not model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance of MXFP8 dynamic quantization."
            )

        # For MoE training with mxfp8, token group sizes must be multiples of 32
        self.moe_fqns = job_config.quantize.grouped_mm.mx.fqns
        if self.moe_fqns:
            logger.info(
                f"Setting token group alignment size to {MXFP8_GROUP_ALIGNMENT_SIZE}"
            )
            set_token_group_alignment_size_m(MXFP8_GROUP_ALIGNMENT_SIZE)

        self.recipe_name = job_config.quantize.grouped_mm.mx.recipe_name
        self.enabled = True
        logger.info("MXFP8 MoE training enabled")

    def convert(self, model: nn.Module):
        """
        Mutates the model inplace replacing instances of nn.Parameter with ScaledGroupedMMTensor.
        This will use low precision grouped GEMMs with dynamic quantization using the specified MX dtype,
        rather than the default high precision grouped GEMMs, for the target MoE FQNs.
        """
        if not self.enabled:
            return
        from torchao.prototype.moe_training.conversion_utils import (
            MoEScalingType,
            MoETrainingConfig,
        )
        from torchao.quantization.quant_api import quantize_

        def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
            for target_fqn in self.moe_fqns:
                if target_fqn in cur_fqn:
                    return True
            return False

        config = MoETrainingConfig(scaling_type=MoEScalingType.MXFP8)
        quantize_(model, config=config, filter_fn=moe_module_filter_fn)
        logger.info(
            f"Converted MoE layers matching FQNS {self.moe_fqns} "
            f"to use dynamic {self.recipe_name} quantization with scaled grouped GEMMs"
        )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        """
        MXFP8 MoE training doesn't require any post-optimizer hooks at the moment
        """
        return


register_model_converter(MXLinearConverter, "quantize.linear.mx")
register_model_converter(MXGroupedMMConverter, "quantize.grouped_mm.mx")
