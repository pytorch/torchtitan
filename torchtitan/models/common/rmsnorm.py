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
    - The Module protocol (``init_weights``) is satisfied and ``build()``
      is inherited from ``Configurable.Config``.

    ``normalized_shape`` is passed as a kwarg to ``build()`` (and forwarded
    to ``__init__``) because it is derived from the parent model config
    (e.g. ``dim``, ``head_dim``, ``lora_rank``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        eps: float = 1e-5
        elementwise_affine: bool = True

    def __init__(self, config: Config, *, normalized_shape: int):
        super().__init__(
            normalized_shape,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
        )

    def init_weights(self, **kwargs) -> None:
        self.reset_parameters()
