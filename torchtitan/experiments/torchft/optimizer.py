# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn

from torchtitan.components.optim import Optim, OptimizersContainer
from torchtitan.components.optim.utils import (
    get_flat_optim_state_dict,
    init_optim_state,
)
from torchtitan.distributed import utils as dist_utils

if TYPE_CHECKING:
    from torchtitan.experiments.torchft.manager import TorchFTManager

__all__ = ["TorchFTOptim", "TorchFTOptimizersContainer"]

logger = logging.getLogger(__name__)


class TorchFTOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[nn.Module],
        enable_cuda_graph: bool = False,
    ) -> None:
        super().__init__(
            config,
            model_parts=model_parts,
            enable_cuda_graph=enable_cuda_graph,
        )

        # Force to initialize the optimizer state so that `optim.step()`
        # won't be called by state_dict() and load_state_dict().
        for optim in self.optimizers:
            init_optim_state(optim)
        self.cache_state_dict: dict[str, Any] = {}
        self._quorum_manager = None

    def configure_fault_tolerance(self, ft_manager: "TorchFTManager") -> None:
        """Configure quorum handling after the optimizer is built."""
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


class TorchFTOptim(Optim):
    """Vote out non-finite steps instead of asserting."""

    # A divergence that persists across retries is not a communication fault.
    MAX_CONSECUTIVE_NON_FINITE_STEPS = 10

    @dataclass(kw_only=True, slots=True)
    class Config(Optim.Config):
        optimizer: OptimizersContainer.Config = field(
            default_factory=TorchFTOptimizersContainer.Config
        )

    def __init__(self, config: Config, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._quorum_manager = None
        self._consecutive_non_finite_steps = 0

    def configure_fault_tolerance(self, ft_manager: "TorchFTManager") -> None:
        """Configure quorum handling after the optimizer is built."""
        # Semi-sync algorithms manage quorum in their own synchronization hooks.
        self._quorum_manager = (
            ft_manager.manager if ft_manager.use_async_quorum else None
        )

    def _update(self, loss: torch.Tensor) -> torch.Tensor:
        if self._quorum_manager is None:
            return super()._update(loss)
        # Duplicates Optim._update except for the finite check.
        grad_norm = dist_utils.clip_grad_norm_(
            self.parameters,
            self.config.max_norm,
            foreach=True,
            pp_mesh=self.parallelism_context.get_optional_mesh("pp"),
            ep_enabled=self.parallelism_context.ep_enabled,
        )
        loss_is_finite = torch.isfinite(loss).all().to(torch.int32)
        if not self.parallelism_context.pp_enabled or self.pp_has_last_stage:
            loss_mesh = self.parallelism_context.get_optional_mesh("loss")
            if loss_mesh is not None:
                torch.distributed.all_reduce(
                    loss_is_finite,
                    op=torch.distributed.ReduceOp.MIN,
                    group=loss_mesh.get_group(),
                )
        pp_mesh = self.parallelism_context.get_optional_mesh("pp")
        if pp_mesh is not None:
            torch.distributed.all_reduce(
                loss_is_finite,
                op=torch.distributed.ReduceOp.MIN,
                group=pp_mesh.get_group(),
            )
        step_is_finite = loss_is_finite.logical_and(torch.isfinite(grad_norm).all())
        # Gradients are garbage after a swallowed communication error. Vote the
        # step out so the replica heals instead of killing its CUDA context.
        if step_is_finite.item():
            self._consecutive_non_finite_steps = 0
        else:
            self._consecutive_non_finite_steps += 1
            if (
                self._consecutive_non_finite_steps
                > self.MAX_CONSECUTIVE_NON_FINITE_STEPS
            ):
                raise RuntimeError(
                    "Loss or gradient norm is not finite for "
                    f"{self._consecutive_non_finite_steps} consecutive steps."
                )
            logger.warning(
                "Loss or gradient norm is not finite; skipping step "
                f"({self._consecutive_non_finite_steps} consecutive)."
            )
            self._quorum_manager.report_error(
                RuntimeError("Loss or gradient norm is not finite.")
            )
        # should_commit() in the container step returns False after report_error.
        self.optimizers.step()
        return grad_norm
