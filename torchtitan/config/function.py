# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torchtitan.config.configurable import Configurable

R = TypeVar("R")


class Function(Generic[R], Configurable, ABC):
    """Base class for configurable callable objects."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> R:
        pass


class CallableFunction(Function[R]):
    """Configurable wrapper around an arbitrary callable.

    Use for config fields whose values are computed lazily -- e.g., per-layer
    parameter initialization that varies by depth::

        CallableFunction.Config(
            fn=lambda layer_id: {
                "weight": partial(trunc_normal_, std=0.02 / (2 * (layer_id + 1)) ** 0.5),
            }
        )
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Function.Config):
        fn: Callable[..., R]

    def __init__(self, config: Config) -> None:
        self.fn = config.fn

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        return self.fn(*args, **kwargs)
