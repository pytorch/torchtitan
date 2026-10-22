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
from torchtitan.distributed import ParallelismContext, utils as dist_utils
from torchtitan.distributed.accelerator_graph import (
    accelerator_graphs_supported,
    NUM_ACCELERATOR_GRAPH_WARMUP_STEPS,
    wrap_with_accelerator_graph,
)
from torchtitan.tools.utils import device_module, device_type

from .ema import EMA
from .lr_scheduler import LRSchedulersContainer
from .optimizer import OptimizersContainer


logger = logging.getLogger(__name__)


class GradientClipper(Configurable):
    """Default optimizer-owned gradient clipping implementation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config) -> None:
        self.config = config

    def clip_grad_norm_(
        self,
        parameters: list[nn.Parameter],
        max_norm: float | int,
        *,
        parallelism_context: ParallelismContext,
    ) -> torch.Tensor:
        """Clip gradients with ``dist_utils.clip_grad_norm_``.

        Use ``foreach=True`` and the PP mesh from ``parallelism_context``.
        Return the pre-clip global L2 norm as a local zero-dimensional tensor
        with an equivalent value on every rank. Preserve non-finite values and
        do not reuse the returned norm as the clip coefficient or scratch.
        """
        return dist_utils.clip_grad_norm_(
            parameters,
            max_norm,
            foreach=True,
            pp_mesh=parallelism_context.get_optional_mesh("pp"),
        )


class Optim(Configurable):
    """Own the parameter update and its eager state."""

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
        gradient_clipper: GradientClipper.Config = field(
            default_factory=GradientClipper.Config
        )
        """Optimizer-owned gradient clipping implementation."""
        enable_cuda_graph: bool = False
        """Capture gradient clipping and optimizer updates in one CUDA graph."""

        def __post_init__(self) -> None:
            if self.max_norm < 0:
                raise ValueError("max_norm must be greater than or equal to 0.")

    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    ema: EMA | None
    gradient_clipper: GradientClipper

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[nn.Module],
        parallelism_context: ParallelismContext,
        training_steps: int,
        pp_has_last_stage: bool,
        cuda_graph_annotation_config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.parallelism_context = parallelism_context
        self.pp_has_last_stage = pp_has_last_stage
        self.parameters = [
            parameter
            for model_part in model_parts
            for parameter in model_part.parameters()
        ]
        self._is_first_step_call = True
        enable_cuda_graph = (
            config.enable_cuda_graph
            and accelerator_graphs_supported()
            and all(
                parameter.device.type == device_type for parameter in self.parameters
            )
        )
        if config.enable_cuda_graph and not enable_cuda_graph:
            logger.warning(
                "Optim graph capture is disabled because the runtime or model "
                "parameter device does not support graph capture."
            )
        self.optimizers = config.optimizer.build(
            model_parts=model_parts,
            enable_cuda_graph=enable_cuda_graph,
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
        self.gradient_clipper = config.gradient_clipper.build()

        self._run_update = self._update
        if enable_cuda_graph:
            self._run_update = wrap_with_accelerator_graph(
                self._update,
                num_warmup_iterations=NUM_ACCELERATOR_GRAPH_WARMUP_STEPS,
                name="optimizer",
                annotation_config=cuda_graph_annotation_config,
            )

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        """Clear gradients owned by the optimizers."""
        self.optimizers.zero_grad(set_to_none=set_to_none)

    def step(self, loss: torch.Tensor, *, current_step: int) -> torch.Tensor:
        """Validate loss and gradients, update parameters, then advance eager state."""
        if self._is_first_step_call:
            # Finish GPU work before freeing cached memory for NCCL buffers.
            # https://github.com/pytorch/torchtitan/issues/5019
            if torch.accelerator.is_available() and device_module.is_initialized():
                device_module.synchronize()
            device_module.empty_cache()
            self._is_first_step_call = False
        grad_norm = self._run_update(loss)
        self.lr_schedulers.step()
        if self.ema is not None:
            self.ema.step(current_step)
        return grad_norm

    def _update(self, loss: torch.Tensor) -> torch.Tensor:
        """Clip gradients, check loss and gradient norm, then update parameters."""
        grad_norm = self.gradient_clipper.clip_grad_norm_(
            self.parameters,
            self.config.max_norm,
            parallelism_context=self.parallelism_context,
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
        torch._assert_async(
            step_is_finite,
            "Loss or gradient norm is not finite. Stopping before the update.",
        )
        self.optimizers.step()
        return grad_norm
