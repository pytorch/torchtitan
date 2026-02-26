# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch.nn as nn

from torchtitan.protocols.module import Module


class Embedding(nn.Embedding, Module):
    """Configurable nn.Embedding with init_weights support.

    Uses diamond inheritance (nn.Embedding + Module) so that:
    - The module hierarchy stays flat (no extra wrapper layer).
    - All nn.Embedding logic (forward, state_dict, etc.) is reused as-is.
    - The Module protocol is satisfied and ``build()`` is inherited from
      ``Configurable.Config``.

    ``num_embeddings`` and ``embedding_dim`` live in ``Config`` (defaulting
    to ``None``).  They are typically supplied via ``build()`` kwargs from
    the parent model and passed into a cloned config.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # ``num_embeddings`` and ``embedding_dim`` are usually passed by the
        # parent modules through build(). So the default values are None to
        # fit Configurable.Config convention.
        num_embeddings: int | None = None
        embedding_dim: int | None = None
        init_mean: float = 0.0
        init_std: float = 1.0

    def __init__(self, config: Config):
        if config.num_embeddings is None or config.embedding_dim is None:
            raise TypeError(
                "Embedding requires num_embeddings and embedding_dim. "
                "Either set them in Config or pass them to build()."
            )
        super().__init__(config.num_embeddings, config.embedding_dim)
        self.config = config

    def init_weights(self, **kwargs) -> None:
        nn.init.normal_(
            self.weight, mean=self.config.init_mean, std=self.config.init_std
        )
