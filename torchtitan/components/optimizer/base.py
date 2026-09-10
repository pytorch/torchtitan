# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""torchtitan-native optimizers.

An optimizer is a Configurable whose Config splits into two channels:

- ``to_param_group_kwargs()`` are hyperparameters recorded on each PyTorch
  parameter group. Two groups using the same optimizer class may differ here,
  for example a different ``lr`` per group.
- ``to_factory_kwargs()`` are instance-wide constructor arguments such as
  communication bucket specs. Groups batched into one optimizer instance must
  agree on them.

This module must not import from ``torchtitan.distributed``: ``dist_muon``
imports ``Optimizer`` from here, and the reverse edge would close a cycle.
"""

from dataclasses import dataclass, fields
from typing import Any, ClassVar

import torch

from torchtitan.config import Configurable

__all__ = ["Adam", "AdamW", "Optimizer", "from_torch"]


class Optimizer(Configurable, torch.optim.Optimizer):
    """Base class for torchtitan-native optimizers."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Per-parameter-group optimizer configuration.

        Subclasses list their instance-wide fields in ``_FACTORY_FIELDS``;
        every other public field is a parameter-group hyperparameter.
        """

        _FACTORY_FIELDS: ClassVar[frozenset[str]] = frozenset()

        def to_param_group_kwargs(self) -> dict[str, Any]:
            """Hyperparameters to record on each PyTorch parameter group."""
            return {
                f.name: getattr(self, f.name)
                for f in fields(self)
                if not f.name.startswith("_")
                and f.name not in self._FACTORY_FIELDS
            }

        def to_factory_kwargs(self) -> dict[str, Any]:
            """Instance-wide arguments passed once to the constructor."""
            return {name: getattr(self, name) for name in self._FACTORY_FIELDS}


def from_torch(torch_optimizer_cls: type[torch.optim.Optimizer]) -> type:
    """Adapt a ``torch.optim`` optimizer into a torchtitan ``Optimizer``.

    Per-group hyperparameters ride on the parameter-group dicts, so the torch
    constructor receives only the instance-wide factory kwargs. The torch class
    name is copied onto the adapter so logging that reports
    ``type(optimizer).__name__`` is unchanged.
    """

    class _FromTorch(Optimizer, torch_optimizer_cls):
        def __init__(self, config: Optimizer.Config, *, params: Any) -> None:
            torch_optimizer_cls.__init__(self, params, **config.to_factory_kwargs())

    _FromTorch.__name__ = torch_optimizer_cls.__name__
    _FromTorch.__qualname__ = torch_optimizer_cls.__name__
    return _FromTorch


class Adam(from_torch(torch.optim.Adam)):
    @dataclass(kw_only=True, slots=True)
    class Config(Optimizer.Config):
        lr: float
        betas: tuple[float, float] = (0.9, 0.95)
        eps: float = 1e-8


class AdamW(from_torch(torch.optim.AdamW)):
    @dataclass(kw_only=True, slots=True)
    class Config(Optimizer.Config):
        lr: float
        betas: tuple[float, float] = (0.9, 0.95)
        eps: float = 1e-8
        weight_decay: float = 0.1
