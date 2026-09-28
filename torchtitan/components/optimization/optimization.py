# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.cuda_graph import (
    cuda_graphs_supported,
    NUM_CUDA_GRAPH_WARMUP_STEPS,
    wrap_with_cuda_graph,
)

from .ema import EMA
from .lr_scheduler import LRSchedulersContainer
from .optimizer import OptimizersContainer


logger = logging.getLogger(__name__)


class Optimization(Configurable):
    """Own the parameter update and its eager state.

    ``optimizer_build_kwargs`` forwards runtime dependencies required by a
    custom ``OptimizersContainer`` implementation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        ema: EMA.Config | None = None
        """Optional online EMA of model weights."""
        max_norm: float | int = 1.0
        """Maximum norm used for gradient clipping."""
        enable_cuda_graph: bool = False
        """Capture gradient clipping and optimizer updates in one CUDA graph."""

        def __post_init__(self) -> None:
            if self.max_norm < 0:
                raise ValueError("max_norm must be greater than or equal to 0.")

    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    ema: EMA | None

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
        training_steps: int,
        pp_has_last_stage: bool,
        optimizer_build_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.parallel_dims = parallel_dims
        self.pp_has_last_stage = pp_has_last_stage
        self.parameters = [
            parameter
            for model_part in model_parts
            for parameter in model_part.parameters()
        ]
        enable_cuda_graph = (
            config.enable_cuda_graph
            and cuda_graphs_supported()
            and all(parameter.device.type == "cuda" for parameter in self.parameters)
        )
        if config.enable_cuda_graph and not enable_cuda_graph:
            logger.warning(
                "Optimization CUDA graph is disabled because the runtime or model "
                "parameter device does not support CUDA graphs."
            )
        optimizer_build_kwargs = dict(optimizer_build_kwargs or {})
        if enable_cuda_graph:
            optimizer_build_kwargs["enable_cuda_graph"] = True
        self.optimizers = config.optimizer.build(
            model_parts=model_parts,
            **optimizer_build_kwargs,
        )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=training_steps,
        )
        self.ema = (
            config.ema.build(model_parts=model_parts)
            if config.ema is not None
            else None
        )

        self._run_update = self._update
        if enable_cuda_graph:
            self._run_update = wrap_with_cuda_graph(
                self._update,
                num_warmup_iterations=NUM_CUDA_GRAPH_WARMUP_STEPS,
            )

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        """Clear gradients owned by the optimizers."""
        self.optimizers.zero_grad(set_to_none=set_to_none)

    def step(self, loss: torch.Tensor, *, current_step: int) -> torch.Tensor:
        """Validate loss and gradients, update parameters, then advance eager state."""
        grad_norm = self._run_update(loss)
        self.lr_schedulers.step()
        if self.ema is not None:
            self.ema.step(current_step)
        return grad_norm

    def _update(self, loss: torch.Tensor) -> torch.Tensor:
        """Clip gradients, check loss and gradient norm, then update parameters."""
        grad_norm = dist_utils.clip_grad_norm_(
            self.parameters,
            self.config.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )
        loss_is_finite = torch.isfinite(loss).all().to(torch.int32)
        if not self.parallel_dims.pp_enabled or self.pp_has_last_stage:
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
            if loss_mesh is not None:
                torch.distributed.all_reduce(
                    loss_is_finite,
                    op=torch.distributed.ReduceOp.MIN,
                    group=loss_mesh.get_group(),
                )
        pp_mesh = self.parallel_dims.get_optional_mesh("pp")
        if pp_mesh is not None:
            torch.distributed.all_reduce(
                loss_is_finite,
                op=torch.distributed.ReduceOp.MIN,
                group=pp_mesh.get_group(),
            )
        step_is_finite = loss_is_finite.logical_and(torch.isfinite(grad_norm).all())
        torch._assert_async(
            step_is_finite,
            "Loss or gradient norm is not finite. Stopping before the update.",
        )
        self.optimizers.step()
        return grad_norm
