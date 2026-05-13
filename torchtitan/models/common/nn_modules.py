# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module wrappers around standard ``nn.Module`` subclasses.

Each wrapper participates in the torchtitan ``Module`` protocol
(``init_states``, ``sharding_config``, etc.) via ``from_nn_module``.

Two usage patterns:

1. Direct construction (no declarative sharding):

       norm = LayerNorm(dim, eps)

   Behaves like plain ``nn.LayerNorm``; the instance has no
   ``sharding_config``. Use this when the layer's params don't need a
   declarative TP/SP/EP plan.

2. Config-based construction (declarative sharding):

       norm = LayerNorm.Config(
           sharding_config=norm_config(enable_sp=False),
       ).build(dim, eps)

   The Config carries the sharding plan; ``build(*args, **kwargs)``
   forwards positional / keyword args to the underlying
   ``nn.Module`` constructor and applies ``sharding_config`` /
   ``param_init`` to the resulting instance. Use this when the
   parent's Config tree carries the sharding plan -- the standard
   torchtitan flow.
"""

import torch.nn as nn

from torchtitan.protocols.module import Module


Conv2d = Module.from_nn_module(nn.Conv2d)
GELU = Module.from_nn_module(nn.GELU)
GroupNorm = Module.from_nn_module(nn.GroupNorm)
Identity = Module.from_nn_module(nn.Identity)
LayerNorm = Module.from_nn_module(nn.LayerNorm)
SiLU = Module.from_nn_module(nn.SiLU)


__all__ = [
    "Conv2d",
    "GELU",
    "GroupNorm",
    "Identity",
    "LayerNorm",
    "SiLU",
]
