# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from torchtitan.protocols.module import Module
from torchtitan.protocols.state_initializer import StateInitializer


class LinearStateInitializer(StateInitializer):
    @dataclass(kw_only=True, slots=True)
    class Config(StateInitializer.Config):
        init_mean: float = 0.0
        init_std: float = 0.02
        cutoff_factor: float = 0.0

    def __init__(self, config: Config):
        self.init_mean = config.init_mean
        self.init_std = config.init_std
        self.cutoff_factor = config.cutoff_factor

    def init_states(self, module: nn.Module, *, buffer_device=None) -> None:
        weight = module.weight
        assert isinstance(weight, torch.Tensor)
        std = self.init_std
        if self.cutoff_factor > 0:
            a = -self.cutoff_factor * std
            b = self.cutoff_factor * std
            nn.init.trunc_normal_(
                weight,
                mean=self.init_mean,
                std=std,
                a=a,
                b=b,
            )
        else:
            nn.init.trunc_normal_(weight, mean=self.init_mean, std=std)
        bias = module.bias
        if isinstance(bias, torch.Tensor):
            nn.init.zeros_(bias)


class Linear(Module, nn.Linear):
    """Configurable nn.Linear with StateInitializer support.

    Uses diamond inheritance (nn.Linear + Module) so that:
    - The module hierarchy stays flat (no extra wrapper layer).
    - All nn.Linear logic (forward, state_dict, etc.) is reused as-is.
    - The Module protocol (``init_states``) is satisfied and ``build()``
      is inherited from ``Configurable.Config``.

    ``in_features`` and ``out_features`` use ``field(init=False)`` so
    they are excluded from ``Config.__init__``.  They are typically supplied
    via ``build()`` kwargs from the parent model.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int = field(init=False)
        out_features: int = field(init=False)
        bias: bool = False
        state_initializer: StateInitializer.Config = field(
            default_factory=LinearStateInitializer.Config
        )

    def __init__(self, config: Config):
        if not hasattr(config, "in_features") or not hasattr(config, "out_features"):
            raise TypeError(
                "Linear.Config requires 'in_features' and 'out_features' to be set. "
                "Use Config.build(in_features=..., out_features=...) or set them "
                "on the Config instance before constructing Linear directly."
            )
        super().__init__(
            config, config.in_features, config.out_features, bias=config.bias
        )
