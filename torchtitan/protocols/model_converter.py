# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Protocol, Union

import torch.nn as nn

from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


class ModelConverter(Protocol):
    """General model converter interface.

    A model converter is applying a modification to PyTorch model.
    Typical use cases are:
        - Quantization: using QAT, FP8, ... specialized linear layers;
        - Fused optimized layers (e.g. flash-attention, norms, ...)
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        ...

    def convert(self, model: nn.Module):
        """Inplace convertion of the model."""
        ...

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        """Post-optimizer (optional) hook (e.g. compute weights statistics)."""
        ...


_registry_model_converter_cls: Dict[str, type[ModelConverter]] = {}
"""Registry of model converter classes.
"""


def register_model_converter(converter_cls: type[ModelConverter], name: str):
    """Register a model converter class.

    A registered model converter can be applied on any model
    using the `model.converters` config parameter.
    """
    assert (
        name not in _registry_model_converter_cls
    ), f"A model converter '{name}' is already registered."
    _registry_model_converter_cls[name] = converter_cls


class ModelConvertersContainer(ModelConverter):
    """Model converters sequential container.

    The class build the sequence of model converters defined in `model.converters`
    job config, and apply them to the model sequentially.
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        converter_classes = [
            _registry_model_converter_cls[name] for name in job_config.model.converters
        ]
        self.converters = [
            mh_cls(job_config, parallel_dims) for mh_cls in converter_classes
        ]
        self.print_after_conversion = job_config.model.print_after_conversion

    def convert(self, model: nn.Module):
        for mh in self.converters:
            mh.convert(model)
        if self.print_after_conversion:
            logger.info(f"Model definion after conversion:\n\n{model}\n\n")

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        for mh in self.converters:
            mh.post_optimizer_hook(model)


def build_model_converters(
    job_config: JobConfig, parallel_dims: ParallelDims
) -> ModelConvertersContainer:
    """Build the collection of model converters to apply to the model."""
    return ModelConvertersContainer(job_config, parallel_dims)
