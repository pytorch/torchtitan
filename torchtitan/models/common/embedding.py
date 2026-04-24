# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch.nn as nn

from torchtitan.protocols.module import Module

__all__ = ["Embedding"]


class Embedding(nn.Embedding, Module):
    """Configurable nn.Embedding.

    Uses diamond inheritance (nn.Embedding + Module) so that:
    - The module hierarchy stays flat (no extra wrapper layer).
    - All nn.Embedding logic (forward, state_dict, etc.) is reused as-is.
    - The Module protocol is satisfied and ``build()`` is inherited from
      ``Configurable.Config``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int
        embedding_dim: int

    def __init__(self, config: Config):
        super().__init__(config.num_embeddings, config.embedding_dim)
