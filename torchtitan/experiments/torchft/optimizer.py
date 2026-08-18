# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import Optimizer

from torchtitan.components.checkpoint_utils import init_optim_state
from torchtitan.components.optimizer import OptimizersContainer

if TYPE_CHECKING:
    from torchtitan.experiments.torchft.manager import TorchFTManager

__all__ = ["TorchFTOptimizersContainer"]

has_torchft = importlib.util.find_spec("torchft") is not None
if has_torchft:
    import torchft


class _OptimizerAdapter:
    """Call the base container without re-entering TorchFT dispatch."""

    def __init__(self, owner: "TorchFTOptimizersContainer") -> None:
        self._owner = owner

    def step(self, *args, **kwargs) -> None:
        return OptimizersContainer.step(self._owner, *args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        return OptimizersContainer.zero_grad(self._owner, *args, **kwargs)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        self._owner.add_param_group(param_group)

    def state_dict(self) -> dict[str, Any]:
        return self._owner.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._owner.load_state_dict(state_dict)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._owner.param_groups

    @property
    def state(self) -> Mapping[torch.Tensor, object]:
        return self._owner.state


class TorchFTOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[nn.Module],
        ft_manager: "TorchFTManager",
    ) -> None:
        super().__init__(config, model_parts=model_parts)

        # Force to initialize the optimizer state so that `optim.step()`
        # won't be called by state_dict() and load_state_dict().
        for optim in self.optimizers:
            init_optim_state(optim)
        self.cache_state_dict: dict[str, Any] = {}
        self._inner_optimizer = cast(Optimizer, _OptimizerAdapter(self))
        self._ft_optimizer = torchft.Optimizer(
            ft_manager.manager, self._inner_optimizer
        )
        # Whether to determine quorum using FT.optimizer,
        # in semi-sync training we use the synchronization step to start quorum
        # A single-replica quorum always commits, so wrapping it only adds
        # control-plane work without providing failure isolation.
        self._use_ft_optimizer: bool = (
            ft_manager.use_async_quorum and ft_manager.group_size > 1
        )

    def init_cache_state_dict(self) -> None:
        self.cache_state_dict = super().state_dict()

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # We have to invalidate the `cache_state_dict` because optimizer uses
        # assign instead of copy when doing `load_state_dict()`. Without
        # invalidating the `cache_state_dict`, there will be memory leakage.
        self.cache_state_dict = {}
        super().load_state_dict(state_dict)
        self.init_cache_state_dict()

    def step(self, *args, **kwargs) -> None:
        """Dispatch one container step through TorchFT when required.

        TorchFT's OptimizerWrapper.step() is designed to be called only once
        per train step per torchft.Manager regardless how many optimizers are used.
        The inner adapter calls the base implementation directly, so TorchFT cannot
        recursively enter this method.
        """
        if self._use_ft_optimizer:
            return self._ft_optimizer.step(*args, **kwargs)
        return self._inner_optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        """Calling the correct zero_grad() depending on the caller.

        Check the comment in ``step()``.
        """
        if self._use_ft_optimizer:
            return self._ft_optimizer.zero_grad(*args, **kwargs)
        return self._inner_optimizer.zero_grad(*args, **kwargs)
