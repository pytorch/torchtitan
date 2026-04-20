# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch.nn as nn

from torchtitan.protocols.module import Module

# Cache: maps original class -> patched class (with Linear protocol).
_patched_classes: dict[type, type] = {}


class Linear(nn.Linear, Module):
    """Configurable nn.Linear.

    Uses diamond inheritance (nn.Linear + Module) so that:
    - The module hierarchy stays flat (no extra wrapper layer).
    - All nn.Linear logic (forward, state_dict, etc.) is reused as-is.
    - The Module protocol is satisfied and ``build()`` is inherited from
      ``Configurable.Config``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        out_features: int
        bias: bool = False

    def __init__(self, config: Config):
        super().__init__(
            config.in_features,
            config.out_features,
            bias=config.bias,
        )


def inject_linear_protocol(mod: nn.Module) -> None:
    """Patch *mod*'s class to also inherit from ``Linear``."""
    orig_cls = type(mod)
    if orig_cls not in _patched_classes:

        def _raise_error(self, *args, **kwargs):
            raise RuntimeError(
                f"{type(self).__name__} should not be constructed directly."
            )

        _patched_classes[orig_cls] = type(
            f"{orig_cls.__name__}_WithLinear",
            (orig_cls, Linear),
            {"__init__": _raise_error},
        )
    mod.__class__ = _patched_classes[orig_cls]
