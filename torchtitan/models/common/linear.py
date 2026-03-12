# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch.nn as nn

from torchtitan.protocols.module import Module


class Linear(nn.Linear, Module):
    """Configurable nn.Linear with init_weights support.

    Uses diamond inheritance (nn.Linear + Module) so that:
    - The module hierarchy stays flat (no extra wrapper layer).
    - All nn.Linear logic (forward, state_dict, etc.) is reused as-is.
    - The Module protocol (``init_weights``) is satisfied and ``build()``
      is inherited from ``Configurable.Config``.

    ``in_features`` and ``out_features`` use ``field(init=False)`` so
    they are excluded from ``Config.__init__()``.  They are typically supplied
    via ``build()`` kwargs from the parent model.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int = field(init=False)
        out_features: int = field(init=False)
        bias: bool = False
        init_mean: float = 0.0
        init_std: float = 0.02

    def __init__(self, config: Config):
        if not hasattr(config, "in_features") or not hasattr(config, "out_features"):
            raise TypeError(
                "Linear.Config requires 'in_features' and 'out_features' to be set. "
                "Use Config.build(in_features=..., out_features=...) or set them "
                "on the Config instance before constructing Linear directly."
            )
        super().__init__(
            config.in_features,
            config.out_features,
            bias=config.bias,
        )
        self._init_mean = config.init_mean
        self._init_std = config.init_std

    def init_weights(self, **kwargs) -> None:
        init_std: float | None = kwargs.pop("init_std", None)
        std = init_std if init_std is not None else self._init_std
        nn.init.trunc_normal_(self.weight, mean=self._init_mean, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
