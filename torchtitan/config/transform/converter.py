# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Legacy configurable model-config converters."""

from abc import abstractmethod
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.models.base.module import Module

__all__ = ["ModelConfigConverter"]


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
