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
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


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

        pad_token_groups_for_grouped_mm: bool = True
        """If True, TorchAO pads token groups for MXFP8 grouped GEMM.
        Set to False when EP is enabled (TorchTitan handles padding instead,
        except for DeepEP backend)."""

    def __init__(
        self,
        config: Config,
        *,
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

        self.config = config
        self.enabled = True
        logger.info("MXFP8 MoE training enabled")

    def convert_config(self, model_config) -> None:
        """Convert MoE layers in the model config to use MXFP8 scaled grouped GEMMs.

        Walks the model config tree and sets ``_convert_fn`` on modules
        matching the target FQNs so that ``build()`` replaces instances of
        nn.Parameter with MXFP8TrainingWeightWrapperTensor. This will use low precision grouped
        GEMMs with dynamic quantization using the specified MX dtype, rather than
        the default high precision grouped GEMMs, for the target MoE FQNs.
        """
        if not self.enabled:
            return

        from torchao.prototype.moe_training.config import (
            MXFP8TrainingOpConfig,
            MXFP8TrainingRecipe,
        )
        from torchao.quantization.quant_api import quantize_

        def convert_fn(mod: nn.Module) -> nn.Module:
            recipe = MXFP8TrainingRecipe(self.config.recipe_name)
            mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
            mxfp8_op_config.pad_token_groups_for_grouped_mm = (
                self.config.pad_token_groups_for_grouped_mm
            )
            quantize_(mod, config=mxfp8_op_config)
            return mod

        from torchtitan.protocols.module import Module

        for fqn, config in model_config.walk(Module.Config):
            if any(target_fqn in fqn for target_fqn in self.config.fqns):
                config._convert_fn = convert_fn


        logger.info(
            f"Converted layers matching FQNS {self.config.fqns} "
            f"to use dynamic {self.config.recipe_name} quantization for grouped_mm and linear ops"
        )

