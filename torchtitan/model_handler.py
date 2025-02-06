# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Protocol, Union

import torch.nn as nn

from torchtitan.config_manager import JobConfig
from torchtitan.parallelisms import ParallelDims


class ModelHandler(Protocol):
    """General model handler interface.

    A model handler is applying a modification to PyTorch model.
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


_registry_model_handler_cls: Dict[str, type[ModelHandler]] = {}
"""Registry of model handler classes.
"""


def register_model_handler(handler_cls: type[ModelHandler], name: str):
    """Register a model handler class.

    A registered model handler can be applied on any TorchTitan model
    using the `model.handlers` config parameter.
    """
    assert (
        name not in _registry_model_handler_cls
    ), f"A TorchTitan model handler '{name}' is already registered."
    _registry_model_handler_cls[name] = handler_cls


class ModelHandlersContainer(ModelHandler):
    """Model handlers sequential container.

    The class build the sequence of model handlers defined in `model.handlers`
    job config, and apply them to the model sequentially.
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        handler_names = parse_model_handlers(job_config)
        handler_classes = [_registry_model_handler_cls[name] for name in handler_names]
        self.handlers = [
            mh_cls(job_config, parallel_dims) for mh_cls in handler_classes
        ]

    def convert(self, model: nn.Module):
        for mh in self.handlers:
            mh.convert(model)

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        for mh in self.handlers:
            mh.post_optimizer_hook(model)


def parse_model_handlers(job_config: JobConfig) -> List[str]:
    """Parse the list of model handlers to apply."""
    handler_names = [v.strip() for v in job_config.model.handlers.split(",")]
    handler_names = [v for v in handler_names if len(v) > 0]
    return handler_names


def build_model_handlers_container(
    job_config: JobConfig, parallel_dims: ParallelDims
) -> ModelHandlersContainer:
    """Build the collection of model handlers to apply to the model."""
    return ModelHandlersContainer(job_config, parallel_dims)
