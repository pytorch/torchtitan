# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from torchtitan.protocols.module import Module


class RMSNorm(nn.RMSNorm, Module):
    """Configurable nn.RMSNorm with init_weights support.

    Uses diamond inheritance (nn.RMSNorm + Module) so that:
    - The module hierarchy stays flat (no extra wrapper layer).
    - All nn.RMSNorm logic (forward, state_dict, etc.) is reused as-is.
    - The Module protocol is satisfied and ``build()`` is inherited from
      ``Configurable.Config``.

    ``normalized_shape`` uses ``field(init=False)`` so it is excluded from
    ``Config.__init__``.  It is typically supplied via ``build()`` kwargs
    from the parent model. See ``Configurable`` docstring to understand
    the design pattern.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        normalized_shape: int = field(init=False)
        eps: float = 1e-5
        elementwise_affine: bool = True

    def __init__(self, config: Config):
        super().__init__(
            config.normalized_shape,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
        )
        self.config = config

    def init_weights(self, **kwargs) -> None:
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).to(x.dtype)
