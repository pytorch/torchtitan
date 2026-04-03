# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torchtitan.config.configurable import Configurable

R = TypeVar("R")


class Function(Generic[R], Configurable):
    """A Configurable wrapper around any callable.

    Wraps a callable in a ``Config`` with ``.build()`` support, fitting into
    torchtitan's config system.  The built instance is itself callable.

    Use for config fields whose values are computed lazily — e.g., per-layer
    parameter initialization that varies by depth::

        Function.Config(
            fn=lambda layer_id: {
                "weight": partial(trunc_normal_, std=0.02 / (2 * (layer_id + 1)) ** 0.5),
            }
        )
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        fn: Callable[..., R]

    def __init__(self, config: Config) -> None:
        self.fn = config.fn

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        return self.fn(*args, **kwargs)
