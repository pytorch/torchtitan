# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan integration for the standalone Dist-MoE package."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, TYPE_CHECKING


class _LazyModule:
    def __init__(self, name: str) -> None:
        self._name = name
        self._module: ModuleType | None = None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            try:
                self._module = import_module(self._name)
            except ModuleNotFoundError as error:
                if error.name != self._name:
                    raise
                raise ModuleNotFoundError(
                    "Dist-MoE support requires the optional dist_moe package. "
                    "Install it from "
                    "git+https://github.com/meta-pytorch/dist_moe.git@main.",
                    name=self._name,
                ) from error
        return getattr(self._module, name)


if TYPE_CHECKING:
    import dist_moe

    _dist_moe = dist_moe
else:
    _dist_moe = _LazyModule("dist_moe")


# These imports must remain after _dist_moe is defined because the child modules
# import the proxy from this partially initialized package.
from .mxfp8 import MXFP8DistMoeRoutedExperts  # noqa: E402
from .routed_experts import DistMoeRoutedExperts  # noqa: E402
from .runtime import DistMoeRuntime  # noqa: E402


__all__ = ["DistMoeRoutedExperts", "DistMoeRuntime", "MXFP8DistMoeRoutedExperts"]
