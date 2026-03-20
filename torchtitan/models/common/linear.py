# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch.nn as nn

from torchtitan.protocols.module import Module


class Linear(nn.Linear, Module):
    """Configurable nn.Linear.

    Uses diamond inheritance (nn.Linear + Module) so that:
    - The module hierarchy stays flat (no extra wrapper layer).
    - All nn.Linear logic (forward, state_dict, etc.) is reused as-is.
    - The Module protocol is satisfied and ``build()`` is inherited from
      ``Configurable.Config``.

    ``in_features`` and ``out_features`` use ``field(init=False)`` so
    they are excluded from ``Config.__init__()``.  They are typically supplied
    via ``build()`` kwargs from the parent model.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int = field(init=False)
        out_features: int = field(init=False)
        bias: bool = False

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
