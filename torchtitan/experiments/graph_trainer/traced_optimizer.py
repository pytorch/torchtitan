# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Optimizer update traced into the SPMD joint train-step graph."""

import torch
import torch.distributed._functional_collectives as funcol

from torchtitan.components.optim import Optim
from torchtitan.components.optim.utils import init_optim_state
from torchtitan.distributed import ParallelismContext


class TracedOptimizerUpdate:
    """Clip gradients, check the step, and apply the optimizer inside the graph.

    ``minimal_fx_tracer`` reparametrizes the module and swaps the traced
    parameters and state into every optimizer, so ``optimizer.step()`` traces
    into graph ops on those inputs. This mirrors ``Optim._update`` with two
    differences required by tracing:

    - Gradients are clipped over the traced parameters passed in.
      ``Optim.parameters`` holds the live parameters, which the tracer does not
      swap.
    - The loss-finite flag is reduced with a functional collective so the
      reduction becomes a graph op instead of an in-place eager collective.

    The learning rate scheduler and EMA stay eager. Capturable optimizers keep
    ``lr`` as a device tensor that the scheduler updates in place, so the graph
    reads the current value on every replay.

    The graph returns the pre-clip gradient norm. The bound executor records
    it with :meth:`record_grad_norm`, and the engine's optimizer step returns
    it through :meth:`recorded_grad_norm`.
    """

    def __init__(
        self,
        optim: Optim,
        *,
        parallelism_context: ParallelismContext,
    ) -> None:
        if parallelism_context.pp_enabled:
            raise ValueError(
                "Tracing the optimizer update is not supported with pipeline "
                "parallelism."
            )
        hooked = [
            type(optimizer).__name__
            for optimizer in (optim.optimizers, *optim.optimizers)
            if optimizer._optimizer_step_pre_hooks
            or optimizer._optimizer_step_post_hooks
        ]
        if hooked:
            raise ValueError(
                "Tracing the optimizer update does not support optimizer step "
                f"hooks (registered on: {hooked}), e.g. MoE "
                "load balancing, qk_clip, or bf16 moments."
            )
        self.optim = optim
        self.parallelism_context = parallelism_context
        self._grad_norm: torch.Tensor | None = None

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return list(self.optim.optimizers)

    def initialize_state(self) -> None:
        """Materialize optimizer state, which the tracer requires to exist."""
        for optimizer in self.optimizers:
            init_optim_state(optimizer)

    def __call__(
        self,
        loss: torch.Tensor,
        parameters: list[torch.Tensor],
    ) -> torch.Tensor:
        grad_norm = self.optim.gradient_clipper.clip_grad_norm_(
            parameters,
            self.optim.config.max_norm,
            parallelism_context=self.parallelism_context,
        )
        loss_is_finite = torch.isfinite(loss).all().to(torch.int32)
        loss_mesh = self.parallelism_context.get_optional_mesh("loss")
        if loss_mesh is not None:
            loss_is_finite = funcol.all_reduce(loss_is_finite, "min", loss_mesh)
        step_is_finite = loss_is_finite.logical_and(torch.isfinite(grad_norm).all())
        torch._assert_async(
            step_is_finite,
            "Loss or gradient norm is not finite. Stopping before the update.",
        )
        self.optim.optimizers.step()
        return grad_norm

    def record_grad_norm(self, grad_norm: torch.Tensor) -> None:
        self._grad_norm = grad_norm

    def recorded_grad_norm(self) -> torch.Tensor:
        """Return the gradient norm of the update the graph applied."""
        assert self._grad_norm is not None
        return self._grad_norm
