# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn

from torchtitan.distributed.cudagraph import cudagraph_teardown
from torchtitan.experiments.graph_trainer.common_utils import (
    compute_annotated_loss,
    compute_parameter_gradients,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
    make_pp1_vpp1_graph_pipeline_runtime,
    resolve_graph_pp_runtime_policy,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    validate_memory_policy_config,
)
from torchtitan.experiments.graph_trainer.registry import (
    POST_INIT_HOOKS,
    PRE_TRAIN_STEP_HOOKS,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols import BaseModel
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer


def _maybe_apply_numa_binding(device_index: int, device_type: str) -> None:
    """Pin this process to the NUMA node of its GPU for local memory bandwidth.

    On multi-NUMA machines (e.g. GB200 NVLink-C2C), pinned-memory allocations
    that land on the GPU's local NUMA node get ~350 GB/s D2H bandwidth vs
    ~120 GB/s cross-NUMA. Must run before any pinned memory is allocated.
    """
    if device_type != "cuda":
        return
    from torch.numa.binding import (
        _maybe_apply_numa_binding_to_current_process,
        AffinityMode,
        NumaOptions,
    )

    _maybe_apply_numa_binding_to_current_process(
        device_index=device_index,
        numa_options=NumaOptions(
            affinity_mode=AffinityMode.NODE,
            should_fall_back_if_binding_fails=True,
        ),
    )
    logger.info("NUMA binding applied for GPU %d", device_index)


def make_fwd_bwd_step(model, loss_fn, *, accumulate_gradients: bool = False):
    """Return a plain function that traces the entire fwd+loss+bwd step.

    ``model`` and ``loss_fn`` are captured in the closure so neither shows up
    as a graph input. Pass ``model`` through ``minimal_fx_tracer(fn, module=model)``
    to thread its parameters/buffers as static graph inputs.
    """

    def compute_step(inputs, labels, global_valid_tokens, extra_kwargs):
        pred = model(inputs, **extra_kwargs)
        # The loss function is not a submodule of the model, so
        # annotate_module_fqns won't tag it. Annotate it here so that
        # downstream passes (bucketing, SAC, kernel annotations) can
        # attribute loss nodes in the traced graph.
        loss = compute_annotated_loss(
            loss_fn,
            pred,
            labels,
            {"global_valid_tokens": global_valid_tokens},
        )
        named_params = [
            (name, parameter)
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        ]
        grads = compute_parameter_gradients(loss, named_params)
        return loss, named_params, grads

    if not accumulate_gradients:

        def fwd_bwd_step(inputs, labels, global_valid_tokens, extra_kwargs):
            loss, _named_params, grads = compute_step(
                inputs, labels, global_valid_tokens, extra_kwargs
            )
            return [loss, *grads]

        return fwd_bwd_step

    def fwd_bwd_step(
        gradient_buffers, inputs, labels, global_valid_tokens, extra_kwargs
    ):
        loss, named_params, grads = compute_step(
            inputs, labels, global_valid_tokens, extra_kwargs
        )
        for (fqn, _parameter), grad in zip(named_params, grads, strict=True):
            gradient_buffers[fqn].add_(grad)
        return [loss]

    return fwd_bwd_step


class GraphTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    def __init__(self, config):
        super().__init__(config)

        validate_memory_policy_config(self.config.compile)

        _maybe_apply_numa_binding(self.device.index, self.device.type)

        if self.config.compile.memory_policy == "sac_and_offload":
            from torch._functorch._activation_offloading.offload_ops import (
                pinned_memory_pool,
            )

            self._pinned_pool_ctx = pinned_memory_pool()
            self._pinned_pool_ctx.__enter__()
        else:
            self._pinned_pool_ctx = None

        # Run post-init hook for the active pass pipeline
        POST_INIT_HOOKS.get(self.config.compile.pass_pipeline, lambda _: None)(self)

    def _select_fwd_bwd_fn(
        self, default: Callable[..., torch.Tensor]
    ) -> Callable[..., torch.Tensor]:
        runtime_policy = resolve_graph_pp_runtime_policy(
            self.config.compile,
            pp_enabled=self.parallel_dims.pp_enabled,
            fsdp_enabled=self.parallel_dims.fsdp_enabled,
        )
        if self.config.compile.mode != "aot_fx_trace":
            return default

        if self.parallel_dims.pp_enabled:
            return default

        if (
            runtime_policy.extract_fsdp_grad_reduction
            and not runtime_policy.accumulate_in_schedule
            and self.gradient_accumulation_steps > 1
        ):
            raise ValueError(
                "Scheduled FSDP gradient synchronization requires scheduled "
                "gradient accumulation"
            )
        num_microbatches = (
            self.gradient_accumulation_steps
            if runtime_policy.accumulate_in_schedule
            else 1
        )
        self.pp_schedule = make_pp1_vpp1_graph_pipeline_runtime(
            self.model_parts[0],
            num_microbatches=num_microbatches,
            training=self.config.training,
            parallel_dims=self.parallel_dims,
            parallelism=self.config.parallelism,
            compile_config=self.config.compile,
            device=self.device,
            model_config=self.model_config,
            loss_fn=self.loss_fn,
            loss_config=self.config.loss,
        )
        self.pp_has_last_stage = True
        if runtime_policy.accumulate_in_schedule:
            self.num_pp_microbatches = num_microbatches
            self.gradient_accumulation_steps = 1

        return self._pp_forward_backward_body

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, Any] | list[dict[str, Any]],
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.parallel_dims.pp_enabled or self.config.compile.mode != "aot_fx_trace":
            return super().forward_backward_step(
                input_dict=input_dict,
                global_valid_tokens=global_valid_tokens,
            )

        model = self.model_parts[0]
        microbatches = input_dict if isinstance(input_dict, list) else [input_dict]
        prepared = [
            self._preprocess_fwd_bwd_inputs(model, microbatch)
            for microbatch in microbatches
        ]
        return self.fwd_bwd_fn(
            [(inputs,) for inputs, _labels, _kwargs in prepared],
            [kwargs for _inputs, _labels, kwargs in prepared],
            [labels for _inputs, labels, _kwargs in prepared],
            global_valid_tokens,
        )

    def _preprocess_fwd_bwd_inputs(
        self,
        model: nn.Module,
        input_dict: dict[str, Any],
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, ...],
        torch.Tensor | tuple[torch.Tensor, ...],
        dict[str, Any],
    ]:
        with sl.log_trace_span("preprocess_inputs"):
            inputs, labels, extra_kwargs = cast(BaseModel, model).preprocess_inputs(
                input_dict,
                parallel_dims=self.parallel_dims,
                parallelism=self.config.parallelism,
            )
            # MTP returns one labels tensor per prediction; index 0 contains
            # the complete main-model labels used for token accounting.
            self.ntokens_seen += (
                labels[0].numel() if isinstance(labels, tuple) else labels.numel()
            )
        return inputs, labels, extra_kwargs

    def train_step(self, data_iterator: Iterator[dict[str, Any]]):
        PRE_TRAIN_STEP_HOOKS.get(self.config.compile.pass_pipeline, lambda _: None)(
            self
        )
        super().train_step(data_iterator)

    def close(self) -> None:
        if self._pinned_pool_ctx is not None:
            self._pinned_pool_ctx.__exit__(None, None, None)
            self._pinned_pool_ctx = None

        super().close()

        # See Note [explicit cudagraph teardown] in cudagraph.py
        cudagraph_teardown()
