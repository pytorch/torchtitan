# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch.nn as nn

from torchtitan.protocols.module import Module


class RMSNorm(nn.RMSNorm, Module):
    """Configurable nn.RMSNorm with init_weights support.

    Uses diamond inheritance (nn.RMSNorm + Module) so that:
    - The module hierarchy stays flat (no extra wrapper layer).
    - All nn.RMSNorm logic (forward, state_dict, etc.) is reused as-is.
    - The Module protocol is satisfied and ``build()`` is inherited from
      ``Configurable.Config``.

    ``normalized_shape`` lives in ``Config`` (defaulting to ``None``).
    It is typically supplied via ``build()`` kwargs from the parent model
    and passed into a cloned config.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # ``normalized_shape`` are usually passed by the parent modules
        # through build(). So the default values are None to fit
        # ``Configurable.Config`` convention.
        normalized_shape: int | None = None
        eps: float = 1e-5
        elementwise_affine: bool = True

    def __init__(self, config: Config):
        if config.normalized_shape is None:
            raise TypeError(
                "RMSNorm requires normalized_shape. "
                "Either set it in Config or pass it to build()."
            )
        super().__init__(
            config.normalized_shape,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
        )
        self.config = config

    def init_weights(self, **kwargs) -> None:
        if self.weight is not None:
            self.reset_parameters()
