# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import ClassVar, Literal

import torch.nn as nn
from torchtitan.components.quantization import (
    MXFP8_GROUP_ALIGNMENT_SIZE,
    QuantizationConverter,
)

from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.moe.utils import set_token_group_alignment_size_m
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


class MXFP8Converter(Configurable):
    """
    Wraps the weight tensors of target nn.Linears or 3D nn.Parameters with a tensor subclass
    that overrides grouped_mm and linear ops, dispatching to autograd functions that implement
    dynamic quantization and MXFP8 grouped_m/linear ops, based on the given config.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        _quantization_type: ClassVar[str] = "mxfp8"

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

        # If using mxfp8 on grouped_mm, we need to pad the token group sizes to a multiple of 32
        # TODO: remove once (1) hybridEP landed and we require it to use mxfp8 on experts, or (2) torchao
        # has an efficient token group padding implementation for mxfp8 grouped_mm
        def _has_experts_fqn(fqns: list[str]) -> bool:
            for fqn in fqns:
                if "experts" in fqn and "shared_experts" not in fqn:
                    return True
            return False

        if _has_experts_fqn(config.fqns):
            logger.info(
                f"Setting token group alignment size to {MXFP8_GROUP_ALIGNMENT_SIZE}"
            )
            set_token_group_alignment_size_m(MXFP8_GROUP_ALIGNMENT_SIZE)

        self.recipe_name = config.recipe_name
        self.fqns = config.fqns
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

        from torchao.prototype.moe_training.config import (
            MXFP8TrainingOpConfig,
            MXFP8TrainingRecipe,
        )
        from torchao.quantization.quant_api import quantize_

        def module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
            for target_fqn in self.fqns:
                if target_fqn in cur_fqn:
                    return True
            return False

        # TODO: use sensible defaults configs for now, add more configurability later if needed
        recipe = MXFP8TrainingRecipe(self.recipe_name)
        config = MXFP8TrainingOpConfig.from_recipe(recipe)

        quantize_(model, config=config, filter_fn=module_filter_fn)

        logger.info(
            f"Converted layers matching FQNS {self.fqns} "
            f"to use dynamic {self.recipe_name} quantization for grouped_mm and linear ops"
        )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        """
        MXFP8 training doesn't require any post-optimizer hooks at the moment
        """
        return
