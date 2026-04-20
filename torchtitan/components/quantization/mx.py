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
        from torchtitan.components.quantization import PAD_MULTIPLE_MAP
        from torchtitan.models.common.token_dispatcher import (
            AllToAllTokenDispatcher,
            DeepEPTokenDispatcher,
            TorchAOTokenDispatcher,
        )
        from torchtitan.protocols.module import Module

        pad_token_groups = self.pad_token_groups_for_grouped_mm
        recipe_name = self.config.recipe_name

        def convert_fn(mod: nn.Module) -> nn.Module:
            recipe = MXFP8TrainingRecipe(recipe_name)
            mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
            mxfp8_op_config.pad_token_groups_for_grouped_mm = pad_token_groups
            quantize_(mod, config=mxfp8_op_config)
            return mod

        fqns = self.config.fqns
        for fqn, config in model_config.walk(Module.Config):
            if any(target_fqn in fqn for target_fqn in fqns):
                config._convert_fn = convert_fn
                if hasattr(config, "token_dispatcher") and isinstance(
                    config.token_dispatcher, AllToAllTokenDispatcher.Config
                ):
                    td = config.token_dispatcher
                    config.token_dispatcher = TorchAOTokenDispatcher.Config(
                        num_experts=td.num_experts,
                        top_k=td.top_k,
                        score_before_experts=td.score_before_experts,
                        pad_multiple=PAD_MULTIPLE_MAP["mxfp8"],
                    )
                elif hasattr(config, "token_dispatcher") and isinstance(
                    config.token_dispatcher, DeepEPTokenDispatcher.Config
                ):
                    config.token_dispatcher.pad_multiple = PAD_MULTIPLE_MAP["mxfp8"]

        logger.info(
            f"Converted layers matching FQNS {self.config.fqns} "
            f"to use dynamic {self.config.recipe_name} quantization for grouped_mm and linear ops"
        )

    def convert(self, model: nn.Module):
        pass

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        """
        MXFP8 training doesn't require any post-optimizer hooks at the moment
        """
        return
