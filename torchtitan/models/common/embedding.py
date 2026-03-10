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

__all__ = ["Embedding", "EmbeddingStateInitializer"]


class EmbeddingStateInitializer(StateInitializer):
    @dataclass(kw_only=True, slots=True)
    class Config(StateInitializer.Config):
        init_mean: float = 0.0
        init_std: float = 1.0

    def __init__(self, config: Config):
        self.init_mean = config.init_mean
        self.init_std = config.init_std

    def init_states(self, module: nn.Module, *, buffer_device=None) -> None:
        weight = module.weight
        assert isinstance(weight, torch.Tensor)
        nn.init.normal_(
            weight,
            mean=self.init_mean,
            std=self.init_std,
        )


class Embedding(Module, nn.Embedding):
    """Configurable nn.Embedding with StateInitializer support.

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
        state_initializer: StateInitializer.Config = field(
            default_factory=EmbeddingStateInitializer.Config
        )

    def __init__(self, config: Config):
        if not hasattr(config, "num_embeddings") or not hasattr(
            config, "embedding_dim"
        ):
            raise TypeError(
                "Embedding.Config requires 'num_embeddings' and 'embedding_dim' to be set. "
                "Use Config.build(num_embeddings=..., embedding_dim=...) or set them "
                "on the Config instance before constructing Embedding directly."
            )
        super().__init__(config, config.num_embeddings, config.embedding_dim)
        self.config = config
