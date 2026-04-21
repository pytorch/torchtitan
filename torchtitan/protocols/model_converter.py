# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch.nn as nn

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.config import Configurable
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

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        """Post-optimizer (optional) hook (e.g. compute weights statistics)."""
        ...

    def key_filter(self) -> Callable[[str], bool] | None:
        """Return a filter that identifies state dict keys owned by this converter.

        The returned callable takes a key name and returns ``True`` if the
        key belongs to this converter.  Return ``None`` if the converter
        doesn't introduce new keys.

        Used by ``ModelWrapper`` to exclude converter keys in BASE mode
        (for HF container building).
        """
        return None

    def state_dict_transform(
        self,
    ) -> Callable[[dict[str, Any], bool], dict[str, Any]] | None:
        """Return a transform for the model state dict during saves.

        The returned callable takes ``(state_dict, last_step)`` and returns
        the transformed state dict.  Behavior depends on ``last_step``:

        - ``last_step=False`` (interval save): e.g. LoRA filters to adapter
          keys only, QAT returns as-is.
        - ``last_step=True`` (export save): e.g. LoRA merges adapter into
          base weights, QAT dequantizes.

        Return ``None`` if the converter doesn't need any transform.
        """
        return None


class ModelConvertersContainer(Configurable, ModelConverter):
    """Model converters sequential container.

    Builds converters from their Config objects and applies them
    to the model sequentially.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configuration for model converters (quantization, etc.).

        Each entry in converters should be a Configurable.Config instance
        (e.g. Float8LinearConverter.Config) whose build() constructs the converter.
        """

        converters: list = field(default_factory=list)
        """List of converter Config objects to apply to the model."""

        print_after_conversion: bool = False
        """If true, model definition will be printed after converters are applied."""

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ):
        _validate_quantization(config.converters)
        self.converters: list[ModelConverter] = [
            cc.build(
                parallel_dims=parallel_dims,
                model_compile_enabled=model_compile_enabled,
            )
            for cc in config.converters
        ]
        self.print_after_conversion = config.print_after_conversion

    def convert(self, model: nn.Module):
        for mh in self.converters:
            mh.convert(model)
        if self.print_after_conversion:
            logger.info(f"Model definition after conversion:\n\n{model}\n\n")

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        for mh in self.converters:
            mh.post_optimizer_hook(model)

    def state_dict_transform(
        self,
    ) -> Callable[[dict[str, Any], bool], dict[str, Any]] | None:
        """Compose state_dict_transform from all converters."""
        transforms = [
            t for c in self.converters if (t := c.state_dict_transform()) is not None
        ]
        if not transforms:
            return None
        if len(transforms) == 1:
            return transforms[0]

        def composed(sd: dict[str, Any], last_step: bool = False) -> dict[str, Any]:
            # Reverse order: undo transforms in the opposite order they were
            # applied during model construction (last converter undone first).
            for t in reversed(transforms):
                sd = t(sd, last_step)
            return sd

        return composed

    def key_filter(self) -> Callable[[str], bool] | None:
        """Compose key_filter from all converters (union / OR)."""
        filters = [f for c in self.converters if (f := c.key_filter()) is not None]
        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]

        def composed(key: str) -> bool:
            return any(f(key) for f in filters)

        return composed


def _validate_quantization(converters: list[Configurable.Config]):
    """Validates that all quantization converters use the same quantization type.

    Each quantization converter Config inherits from QuantizationConverter.Config
    and defines a `_quantization_type` ClassVar (e.g. "float8" or "mx").
    This function asserts they are all the same.
    """
    existing_type: str | None = None
    for config in converters:
        if isinstance(config, QuantizationConverter.Config):
            qt = config._quantization_type
            if existing_type is None:
                existing_type = qt
            else:
                assert qt == existing_type, (
                    "Cannot combine model converters with different quantization types: "
                    f"'{qt}' and '{existing_type}'"
                )
