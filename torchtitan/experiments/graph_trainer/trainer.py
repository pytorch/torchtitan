# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.nonstrict_tracer import (
    rewrap_outputs,
    run_traced_module,
    trace_module,
)
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer


class TrainStepModule(nn.Module):
    """Wraps model + loss_fn + autograd.grad into a single traceable forward.

    This allows make_fx to trace through the entire fwd+loss+bwd as one graph.
    """

    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, inputs, labels, global_valid_tokens):
        pred = self.model(inputs)
        loss = self.loss_fn(pred, labels) / global_valid_tokens
        params = [p for _, p in self.model.named_parameters(remove_duplicate=False)]
        grads = torch.autograd.grad(loss, params)
        return [loss] + list(grads)


class GraphTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    def __init__(self, config):
        super().__init__(config)
        # Lazy state for make_fx mode
        self._train_step_module: TrainStepModule | None = None
        self._traced_step: torch.fx.GraphModule | None = None

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        parallel_dims = self.parallel_dims

        if parallel_dims.pp_enabled:
            # Pipeline Parallel is not supported with autograd.grad;
            # fall back to the base implementation.
            return super().forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )

        # Non-PP forward / backward using torch.autograd.grad
        assert len(self.model_parts) == 1
        model = self.model_parts[0]

        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )

        params = [p for p in model.parameters() if p.requires_grad]

        if self.config.compile.mode == "make_fx":
            return self._make_fx_forward_backward_step(
                model, inputs, labels, global_valid_tokens, params
            )

        with self.train_context():
            with self.maybe_enable_amp:
                pred = model(inputs, **extra_inputs, **extra_kwargs)
                loss_sum = self.loss_fn(pred, labels)
                loss = loss_sum / global_valid_tokens

            # Free pred before computing gradients to reduce peak memory
            del pred
            grads = torch.autograd.grad(loss, params)

        # Manually accumulate gradients (for gradient accumulation across microbatches)
        for param, grad in zip(params, grads):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad

        return loss

    def _make_fx_forward_backward_step(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        params: list[torch.Tensor],
    ) -> torch.Tensor:
        if self._traced_step is None:
            # If loss_fn was wrapped by torch.compile, unwrap it — make_fx
            # traces at the ATen level and cannot nest inside dynamo.
            loss_fn = self.loss_fn
            orig = getattr(loss_fn, "_torchdynamo_orig_callable", None)
            if orig is not None:
                loss_fn = orig
            self._train_step_module = TrainStepModule(model, loss_fn)
            logger.info("make_fx: tracing fwd+loss+bwd graph...")
            self._traced_step = trace_module(
                self._train_step_module,
                (inputs, labels, global_valid_tokens),
            )
            logger.info("make_fx: tracing complete")

        outputs = run_traced_module(
            self._traced_step,
            self._train_step_module,
            (inputs, labels, global_valid_tokens),
        )
        wrapped = rewrap_outputs(outputs, self._traced_step._output_subclass_metas)
        loss = wrapped[0]
        grads = wrapped[1:]

        for param, grad in zip(params, grads):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad

        return loss

    def close(self) -> None:
        super().close()

        # Note [explicit cudagraph close]
        # cudagraph holds reference to nccl which prevents destroy nccl
        # group. so we need to explicitly delete cudagraph which is held
        # in joint_graph_module. An explicit gc.collect() is necessary
        # to clean up reference cycles.
        for part in self.model_parts:
            if hasattr(part, "joint_graph_module"):
                part.joint_graph_module = None
        gc.collect()
