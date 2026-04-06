# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_ac_regions,
    register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.cudagraph import cudagraph_teardown
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_default_graph_passes,
)
from torchtitan.trainer import Trainer

# BlockMask must be registered as a pytree node so its tensor children
# are properly traced as graph inputs instead of opaque leaves.
register_blockmask_pytree_node()


def make_fwd_bwd_step(loss_fn):
    """Return a plain function that traces the entire fwd+loss+bwd step.

    ``loss_fn`` is captured in the closure so it is not a graph input.
    """

    def fwd_bwd_step(
        model, inputs, labels, global_valid_tokens, extra_inputs, extra_kwargs
    ):
        pred = model(inputs, **extra_inputs, **extra_kwargs)
        loss = loss_fn(pred, labels) / global_valid_tokens
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params)
        return [loss] + list(grads)

    return fwd_bwd_step


class GraphTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    def __init__(self, config):
        super().__init__(config)

        if self.config.compile.mode == "aot_fx_trace" and self.parallel_dims.pp_enabled:
            raise ValueError(
                "aot_fx_trace compile mode does not support Pipeline Parallel"
            )

        # Lazy state for aot_fx_trace mode
        self._traced_step: TracedResult | None = None

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.compile.mode != "aot_fx_trace":
            return super().forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )

        assert len(self.model_parts) == 1
        model = self.model_parts[0]

        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )

        params = [p for p in model.parameters() if p.requires_grad]
        return self._make_fx_forward_backward_step(
            model,
            inputs,
            labels,
            global_valid_tokens,
            params,
            extra_inputs,
            extra_kwargs,
        )

    def _make_fx_forward_backward_step(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        params: list[torch.Tensor],
        extra_inputs: dict[str, torch.Tensor],
        extra_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        if self._traced_step is None:
            fwd_bwd_fn = make_fwd_bwd_step(self.loss_fn)
            if self.config.activation_checkpoint.mode != "none":
                annotate_ac_regions(model)
            with self.train_context(), self.maybe_enable_amp:
                self._traced_step = minimal_fx_tracer(fwd_bwd_fn)(
                    model,
                    inputs,
                    labels,
                    global_valid_tokens,
                    extra_inputs,
                    extra_kwargs,
                )
            self._traced_step.gm = apply_default_graph_passes(
                self._traced_step.gm,
                self._traced_step.example_inputs,
                enable_graph_ac=self.config.activation_checkpoint.mode != "none",
            )
        with self.train_context(), self.maybe_enable_amp:
            outputs = run_traced(
                self._traced_step,
                model,
                inputs,
                labels,
                global_valid_tokens,
                extra_inputs,
                extra_kwargs,
            )
        loss = outputs[0]
        grads = outputs[1:]

        for param, grad in zip(params, grads):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad

        return loss

    def close(self) -> None:
        super().close()

        # See Note [explicit cudagraph teardown] in cudagraph.py
        cudagraph_teardown()
