# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch.nn as nn

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.optimizer.utils import (
    get_flat_optim_state_dict,
    init_optim_state,
)

if TYPE_CHECKING:
    from torchtitan.experiments.torchft.manager import TorchFTManager

__all__ = ["TorchFTOptimizersContainer"]


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
        # Semi-sync algorithms manage quorum in their own synchronization hooks.
        self._quorum_manager = (
            ft_manager.manager if ft_manager.use_async_quorum else None
        )

    def init_cache_state_dict(self) -> None:
        self.cache_state_dict = super().state_dict()

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def _refresh_cached_state_dict(self) -> None:
        if not self.cache_state_dict:
            return

        # Refresh scalar metadata while preserving the cache and tensor references.
        for optimizer in self.optimizers:
            self.cache_state_dict.update(get_flat_optim_state_dict(optimizer))

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # We have to invalidate the `cache_state_dict` because optimizer uses
        # assign instead of copy when doing `load_state_dict()`. Without
        # invalidating the `cache_state_dict`, there will be memory leakage.
        self.cache_state_dict = {}
        super().load_state_dict(state_dict)
        self.init_cache_state_dict()

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        assert closure is None, "OptimizersContainer does not support closures"
        if (
            self._quorum_manager is not None
            and not self._quorum_manager.should_commit()
        ):
            return None
        # Call inner optimizers directly to avoid re-entering container hooks.
        for optimizer in self.optimizers:
            optimizer.step()
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self._quorum_manager is not None:
            self._quorum_manager.start_quorum()
        super().zero_grad(set_to_none=set_to_none)
