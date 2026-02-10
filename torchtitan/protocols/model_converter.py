# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Protocol, Union

import torch.nn as nn

from torchtitan.config import JobConfig
from torchtitan.config.configurable import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


class ModelConverter(Protocol):
    """General model converter interface.

    A model converter is applying a modification to PyTorch model.
    Typical use cases are:
        - Quantization: using QAT, FP8, ... specialized linear layers;
        - Fused optimized layers (e.g. flash-attention, norms, ...)
    """

    def convert(self, model: nn.Module):
        """Inplace conversion of the model."""
        ...

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        """Post-optimizer (optional) hook (e.g. compute weights statistics)."""
        ...


class ModelConvertersContainer(ModelConverter):
    """Model converters sequential container.

    Builds converters from their Config objects and applies them
    to the model sequentially.
    """

    def __init__(
        self,
        *,
        converter_configs: list[Configurable.Config],
        print_after_conversion: bool,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ):
        _validate_quantization(converter_configs)
        self.converters: list[ModelConverter] = [
            config.build(
                parallel_dims=parallel_dims,
                model_compile_enabled=model_compile_enabled,
            )
            for config in converter_configs
        ]
        self.print_after_conversion = print_after_conversion

    def convert(self, model: nn.Module):
        for mh in self.converters:
            mh.convert(model)
        if self.print_after_conversion:
            logger.info(f"Model definition after conversion:\n\n{model}\n\n")

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        for mh in self.converters:
            mh.post_optimizer_hook(model)


def _validate_quantization(converter_configs: list[Configurable.Config]):
    """Validates that all quantization converters use the same quantization type.

    Each quantization converter Config defines a `_quantization_type` ClassVar
    (e.g. "float8" or "mx"). This function asserts they are all the same.
    """
    existing_type: str | None = None
    for config in converter_configs:
        qt = getattr(config, "_quantization_type", None)
        if qt is not None:
            if existing_type is None:
                existing_type = qt
            else:
                assert qt == existing_type, (
                    "Cannot combine model converters with different quantization types: "
                    f"'{qt}' and '{existing_type}'"
                )


def build_model_converters(
    *, job_config: JobConfig, parallel_dims: ParallelDims
) -> ModelConvertersContainer:
    """Build the collection of model converters to apply to the model."""
    compile_config = job_config.compile
    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )
    return ModelConvertersContainer(
        converter_configs=job_config.model.converters,
        print_after_conversion=job_config.model.print_after_conversion,
        parallel_dims=parallel_dims,
        model_compile_enabled=model_compile_enabled,
    )
