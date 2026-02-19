# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from functools import partial
from importlib.util import find_spec
from typing import Any, ClassVar, Literal

import torch.nn as nn
from torchtitan.components.quantization import MXFP8_GROUP_ALIGNMENT_SIZE

from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.moe.utils import set_token_group_alignment_size_m
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability, has_rocm_capability

from .utils import module_filter_fn


class MXLinearConverter(Configurable):
    """Converts the linear layers of `model` to `MXLinear`."""

    filter_fqns: list[str]
    mx_config: Any  # MXLinearConfig type when imported

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        _quantization_type: ClassVar[str] = "mx"

        mxfp8_dim1_cast_kernel_choice: Literal["triton", "cuda", "torch"] = "triton"
        """
        Temp work around for inductor performance gap.
        CUDA is recommended for best performance.
        """

        recipe_name: str = "mxfp8_cublas"
        """
        If specified, creates MX config from recipe name. See
        https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats for more information.
        """

        filter_fqns: list[str] = field(default_factory=lambda: ["output"])
        """
        Comma-separated list of fully qualified names of modules to skip applying mxfp8 training to.
        nn.Linear modules with any dim size not divisible by 16 are also always skipped due to hardware requirements.
        By default we always skip the output layer.
        """

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ):
        self.enabled = False

        # Ensure minimum torchao versions
        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 linear layers."
            )

        # Can be removed if we enable the emulated versions
        assert has_cuda_capability(10, 0) or has_rocm_capability(
            9, 5
        ), "MXFP8 is only supported on CUDA SM100 or later, or ROCm gfx950 or later"

        # TP not yet supported with torch.compile
        assert not (
            model_compile_enabled and parallel_dims.tp_enabled
        ), "TP not yet supported with torch.compile for mxfp8"

        # Configure MXFP8
        from torchao.prototype.mx_formats.config import (
            MXFP8Dim1CastKernelChoice,
            MXLinearConfig as TorchAOMXLinearConfig,
        )

        torchao_config = TorchAOMXLinearConfig.from_recipe_name(config.recipe_name)
        # pyrefly: ignore [missing-attribute]
        torchao_config.mxfp8_dim1_cast_kernel_choice = MXFP8Dim1CastKernelChoice[
            config.mxfp8_dim1_cast_kernel_choice.upper()
        ]
        self.filter_fqns = config.filter_fqns
        self.torchao_config = torchao_config
        self.enabled = True
        logger.info(f"MX training active with recipe {config.recipe_name}")

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

        assert isinstance(self.torchao_config, TorchAOMXLinearConfig)
        quantize_(
            model,
            config=self.torchao_config,
            filter_fn=partial(module_filter_fn, filter_fqns=self.filter_fqns),
        )
        logger.info("Swapped to MXLinear layers")

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        """
        MXFP8 doesn't require any post-optimizer hooks at the moment
        """
        return


class MXGroupedMMConverter(Configurable):
    """Converts target 3D nn.Parameters of a model, representing 'experts',
    to use MXFP8 scaled grouped GEMMs instead of a high precision grouped GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        _quantization_type: ClassVar[str] = "mx"

        recipe_name: Literal["mxfp8"] = "mxfp8"
        """
        Quantization recipe name for grouped GEMMs. Options: ["mxfp8"]
        """

        fqns: list[str] | str = field(default_factory=list)
        """
        *Prototype feature, performance optimization still in progress*
        Comma-separated list of fully qualified names of MoE modules to apply MXFP8 dynamic quantization
        on grouped GEMM operations.
        This is a prototype feature that requires the torchao nightly build.
        """

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ):
        self.enabled = False

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
        if not model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance of MXFP8 dynamic quantization."
            )

        # For MoE training with mxfp8, token group sizes must be multiples of 32
        self.moe_fqns = config.fqns
        if self.moe_fqns:
            logger.info(
                f"Setting token group alignment size to {MXFP8_GROUP_ALIGNMENT_SIZE}"
            )
            set_token_group_alignment_size_m(MXFP8_GROUP_ALIGNMENT_SIZE)

        self.recipe_name = config.recipe_name
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
