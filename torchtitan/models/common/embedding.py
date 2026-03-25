# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch.nn as nn

from torchtitan.protocols.module import Module

__all__ = ["Embedding"]


class Embedding(nn.Embedding, Module):
    """Configurable nn.Embedding with init_weights support.

    Uses diamond inheritance (nn.Embedding + Module) so that:
    - The module hierarchy stays flat (no extra wrapper layer).
    - All nn.Embedding logic (forward, state_dict, etc.) is reused as-is.
    - The Module protocol is satisfied and ``build()`` is inherited from
      ``Configurable.Config``.

    ``num_embeddings`` and ``embedding_dim`` use ``field(init=False)`` so
    they are excluded from ``Config.__init__()``.  They are typically supplied
    via ``build()`` kwargs from the parent model.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int = field(init=False)
        embedding_dim: int = field(init=False)
        init_mean: float = 0.0
        init_std: float = 1.0

    def __init__(self, config: Config):
        if not hasattr(config, "num_embeddings") or not hasattr(
            config, "embedding_dim"
        ):
            raise TypeError(
                "Embedding.Config requires 'num_embeddings' and 'embedding_dim' to be set. "
                "Use Config.build(num_embeddings=..., embedding_dim=...) or set them "
                "on the Config instance before constructing Embedding directly."
            )
        super().__init__(config.num_embeddings, config.embedding_dim)
        self.config = config

    def init_weights(self, **kwargs) -> None:
        nn.init.normal_(
            self.weight, mean=self.config.init_mean, std=self.config.init_std
        )
