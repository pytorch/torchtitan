# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurable linear modules.

``Linear`` uses diamond inheritance (``nn.Linear`` + ``Module``) so that:
- The module hierarchy stays flat (no extra wrapper layer).
- All ``nn.Linear`` logic (forward, state_dict, etc.) is reused as-is.
- The ``Module`` protocol is satisfied and ``build()`` is inherited
  from ``Configurable.Config``.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.protocols.module import Module


class Linear(nn.Linear, Module):
    """Configurable nn.Linear."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        out_features: int
        bias: bool = False

    def __init__(self, config: Config):
        super().__init__(
            config.in_features,
            config.out_features,
            bias=config.bias,
        )


class ScaledBiasRowwiseLinear(Linear):
    """
    Rowwise linear whose local bias contribution is scaled by TP degree.
    TODO(pianpwk): this should work in decomposition in spmd_types, or as Partial
    init in DTensor. Today the local SPMD typecheck errors on the TP-axis
    input:V, weight:V, bias:P case; decomposing to input @ weight -> P, then P + P should pass.
    For DTensor, this errors because FSDP does not want to redistribute the incoming gradient
    from Replicate -> storage-time Partial.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.tp_degree = 1

    def parallelize(self, parallel_dims) -> None:
        self.tp_degree = parallel_dims.tp
        super().parallelize(parallel_dims)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = (
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
        bias = bias / self.tp_degree
        return F.linear(input, weight, bias)


__all__ = [
    "Linear",
    "ScaledBiasRowwiseLinear",
]
