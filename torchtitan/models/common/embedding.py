# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch.nn as nn

from torchtitan.protocols.module import Module


class Embedding(nn.Embedding, Module):
    """Configurable nn.Embedding with init_weights support.

    Uses diamond inheritance (nn.Embedding + Module) so that:
    - The module hierarchy stays flat (no extra wrapper layer).
    - All nn.Embedding logic (forward, state_dict, etc.) is reused as-is.
    - The Module protocol (``init_weights``) is satisfied and ``build()``
      is inherited from ``Configurable.Config``.

    ``num_embeddings`` and ``embedding_dim`` are ``init=False`` because they
    are derived from the parent model config (``vocab_size`` and ``dim``) in
    ``Decoder.Config.__post_init__``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int = field(init=False)
        embedding_dim: int = field(init=False)
        init_mean: float = 0.0
        init_std: float = 1.0

    def __init__(self, config: Config):
        try:
            num_embeddings = config.num_embeddings
            embedding_dim = config.embedding_dim
        except AttributeError:
            raise ValueError(
                "Embedding.Config requires 'num_embeddings' and "
                "'embedding_dim' to be set before building. "
                "These are typically set by the parent model config "
                "(e.g. Decoder.Config.__post_init__)."
            ) from None
        super().__init__(num_embeddings, embedding_dim)
        self._init_mean = config.init_mean
        self._init_std = config.init_std

    def init_weights(self, **kwargs) -> None:
        nn.init.normal_(self.weight, mean=self._init_mean, std=self._init_std)
