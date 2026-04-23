# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch.nn as nn

from torchtitan.protocols.module import Module


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

    def __init__(self, config_or_in_features=None, out_features=None, bias=False, *, config=None):
        if config is not None:
            config_or_in_features = config
        if isinstance(config_or_in_features, Linear.Config):
            c = config_or_in_features
            super().__init__(c.in_features, c.out_features, bias=c.bias)
        else:
            super().__init__(config_or_in_features, out_features, bias=bias)
