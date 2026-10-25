# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Optimizer update traced into the SPMD joint train-step graph."""

import torch
from torch.fx.traceback import annotate

from torchtitan.components.optim import Optim
from torchtitan.components.optim.utils import init_optim_state
from torchtitan.experiments.graph_trainer.common_utils import OPTIMIZER_UPDATE_META


class TracedOptimizerUpdate:
    """Trace ``Optim``'s update into the GraphTrainer train-step graph.

    ``minimal_fx_tracer`` swaps the traced parameters and state into every
    optimizer, so ``Optim._update`` traces into graph ops on those inputs.
    ``Optim.parameters`` still holds the live parameters, so it points at the
    traced ones while the update is traced.

    The update's ops are annotated with ``OPTIMIZER_UPDATE_META`` so dead-code
    elimination keeps its non-finite step assert.

    The learning rate scheduler and EMA stay eager in ``Optim.step``.
    Capturable optimizers keep ``lr`` as a device tensor that the scheduler
    updates in place, so the graph reads the current value on every replay.

    The graph returns the pre-clip gradient norm. The bound executor records
    it with :meth:`record_grad_norm`, and ``Optim.step`` returns it through
    :meth:`recorded_grad_norm` in place of its eager update.
    """

    def __init__(self, optim: Optim) -> None:
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
        live_parameters = self.optim.parameters
        self.optim.parameters = parameters
        try:
            with annotate({OPTIMIZER_UPDATE_META: True}):
                return self.optim._update(loss)
        finally:
            self.optim.parameters = live_parameters

    def record_grad_norm(self, grad_norm: torch.Tensor) -> None:
        self._grad_norm = grad_norm

    def recorded_grad_norm(self, loss: torch.Tensor) -> torch.Tensor:
        """Return the gradient norm of the update the graph applied."""
        assert self._grad_norm is not None
        return self._grad_norm
