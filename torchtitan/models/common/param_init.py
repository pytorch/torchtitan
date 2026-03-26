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

``PerLayer`` (defined in ``protocols.module``) wraps a factory
``(layer_id) -> dict`` for init that varies by layer depth.
``resolve_per_layer`` walks a config tree and resolves all ``PerLayer``
markers during layer expansion.
"""

import dataclasses

import torch.nn as nn

from torchtitan.protocols.module import PerLayer  # noqa: F401


def resolve_per_layer(cfg: object, layer_id: int) -> None:
    """Walk a dataclass config tree and resolve all ``PerLayer`` param_init fields.

    Mutates *cfg* in place (intended for use on a deep-copied config).
    """
    if not dataclasses.is_dataclass(cfg):
        return
    for f in dataclasses.fields(cfg):
        try:
            val = getattr(cfg, f.name)
        except AttributeError:
            continue
        if isinstance(val, PerLayer):
            object.__setattr__(cfg, f.name, val.resolve(layer_id))
        elif dataclasses.is_dataclass(val):
            resolve_per_layer(val, layer_id)


def skip_param_init(param: nn.Parameter) -> None:
    """No-op initializer: explicitly skip initialization for a parameter.

    Useful when a parameter is tied to another (e.g., weight tying).
    """
    pass


def depth_scaled_std(base_std: float, layer_id: int) -> float:
    """Compute depth-dependent std: base_std / sqrt(2 * (layer_id + 1))."""
    return base_std / (2 * (layer_id + 1)) ** 0.5


def expand_shared_experts(moe_config: object) -> None:
    """Scale ``shared_experts.hidden_dim`` by ``num_shared_experts``.

    Mutates *moe_config*.shared_experts in place.  Call once on the layer
    template before per-layer expansion.  No-op when *moe_config* is None
    or has no shared_experts.
    """
    if moe_config is None or getattr(moe_config, "shared_experts", None) is None:
        return
    moe_config.shared_experts = dataclasses.replace(
        moe_config.shared_experts,
        hidden_dim=moe_config.hidden_dim * moe_config.num_shared_experts,
    )
