# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import ClassVar, Literal

import torch.nn as nn
from torchtitan.components.quantization import QuantizationConverter
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.linear import Linear
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .module_utils import (
    capture_module_attrs,
    inject_module_protocol,
    verify_module_protocol,
)


class MXFP8Converter(QuantizationConverter):
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

        if not model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance of MXFP8 dynamic quantization."
            )

        # If EP is enabled, TorchTitan handles the token group padding for MXFP8 grouped GEMM
        # as part of the EP implementation (except for DeepEP backend).
        # Otherwise, if EP is not enabled, we need TorchAO to pad the token groups.
        self.pad_token_groups_for_grouped_mm = not parallel_dims.ep_enabled

        self.config = config
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
            for target_fqn in self.config.fqns:
                if target_fqn in cur_fqn:
                    return True
            return False

        # Capture Module attrs before conversion (MX may swap classes, losing them).
        # We need to first verify if all nn.Linear have been converted to Linear.
        verify_module_protocol(model, nn.Linear, Linear)
        saved_attrs = capture_module_attrs(
            model, ["_init_mean", "_init_std"], nn_module_cls=nn.Linear
        )

        recipe = MXFP8TrainingRecipe(self.config.recipe_name)
        mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
        mxfp8_op_config.pad_token_groups_for_grouped_mm = (
            self.pad_token_groups_for_grouped_mm
        )

        quantize_(model, config=mxfp8_op_config, filter_fn=module_filter_fn)

        # Re-inject Linear protocol and re-attach attrs
        inject_module_protocol(model, Linear, saved_attrs)
        verify_module_protocol(model, nn.Linear, Linear)

        logger.info(
            f"Converted layers matching FQNS {self.config.fqns} "
            f"to use dynamic {self.config.recipe_name} quantization for grouped_mm and linear ops"
        )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        """
        MXFP8 training doesn't require any post-optimizer hooks at the moment
        """
        return
