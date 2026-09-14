# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Legacy configurable model-config converters."""

from abc import abstractmethod
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.protocols.module import Module

__all__ = ["ModelConfigConverter", "validate_converter_order"]


class ModelConfigConverter(Configurable):
    """Base class for converters that transform the model config tree.

    Subclasses implement ``convert()`` to modify configs before model build.
    Converters may return a replacement root config when the transform needs
    to wrap the model config itself.

    TODO: Replace this legacy interface with ``ModelConfigTransform``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def convert(self, model_config: Module.Config) -> Module.Config:
        raise NotImplementedError


def validate_converter_order(
    converters: list[ModelConfigConverter.Config],
) -> None:
    """Validate converter compatibility and ordering before model conversion."""
    from .cast_linear import LMHeadCastConverter
    from .lora import LoRAConverter
    from .quantization import QuantizationConverter

    has_quantization = any(
        isinstance(converter, QuantizationConverter.Config) for converter in converters
    )
    has_lm_head_cast = any(
        isinstance(converter, LMHeadCastConverter.Config) for converter in converters
    )
    # TODO: Allow this combination once linear quantization and CastLinear can
    # preserve each other's config and compute semantics.
    if has_quantization and has_lm_head_cast:
        raise ValueError(
            "QuantizationConverter and LMHeadCastConverter cannot be combined."
        )

    seen_lora = False
    for converter in converters:
        if isinstance(converter, LoRAConverter.Config):
            seen_lora = True
        elif seen_lora and isinstance(
            converter,
            (QuantizationConverter.Config, LMHeadCastConverter.Config),
        ):
            raise ValueError(
                f"{type(converter).__name__} must be applied before "
                "LoRAConverter. Reorder the converters list."
            )
