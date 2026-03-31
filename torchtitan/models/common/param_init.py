# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parameter initialization utilities for torchtitan models.

``param_init`` is a ``dict[str, Callable]`` mapping local parameter names
to init functions.  Set directly on every sub-config in the model config
registry::

    Linear.Config(param_init={"weight": partial(trunc_normal_, std=0.02), "bias": zeros_})
    RMSNorm.Config(param_init={"weight": nn.init.ones_})

``DeferredCallable.Config`` wraps a factory for init that varies by layer
depth.  ``resolve_deferred`` walks a config tree and resolves all deferred
markers during layer expansion.
"""

import dataclasses

import torch.nn as nn

from torchtitan.config import DeferredCallable


def resolve_deferred(cfg: object, layer_id: int) -> None:
    """Walk a dataclass config tree and resolve all ``DeferredCallable.Config`` fields.

    Mutates *cfg* in place (intended for use on a deep-copied config).
    """
    if not dataclasses.is_dataclass(cfg):
        return
    for f in dataclasses.fields(cfg):
        try:
            val = getattr(cfg, f.name)
        except AttributeError:
            continue
        if isinstance(val, DeferredCallable.Config):
            object.__setattr__(cfg, f.name, val.build()(layer_id))
        elif dataclasses.is_dataclass(val):
            resolve_deferred(val, layer_id)


def skip_param_init(param: nn.Parameter) -> None:
    """No-op initializer: explicitly skip initialization for a parameter.

    Useful when a parameter is tied to another (e.g., weight tying).
    """
    pass


def depth_scaled_std(base_std: float, layer_id: int) -> float:
    """Compute depth-dependent std: base_std / sqrt(2 * (layer_id + 1))."""
    return base_std / (2 * (layer_id + 1)) ** 0.5
