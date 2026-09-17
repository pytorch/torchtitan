# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn

from torchtitan.distributed.cudagraph import cudagraph_teardown
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
    make_spmd_graph_runtime,
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
from torchtitan.trainer import Trainer


logger = logging.getLogger(__name__)


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

    def _select_fwd_bwd_fn(self) -> Callable[..., torch.Tensor]:
        if (
            self.config.compile.mode == "aot_fx_trace"
            and not self.parallel_dims.pp_enabled
        ):
            self.pp_schedule = make_spmd_graph_runtime(
                self.model_parts[0],
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                parallel_dims=self.parallel_dims,
                parallelism=self.config.parallelism,
                compile_config=self.config.compile,
                device=self.device,
                model_config=self.model_config,
                loss_fn=self.loss_fn,
                trainer_config=self.config,
            )
            stages = self.pp_schedule.schedule._stages
            self.pp_has_first_stage = any(stage.is_first for stage in stages)
            self.pp_has_last_stage = any(stage.is_last for stage in stages)
            assert self.pp_has_first_stage and self.pp_has_last_stage
            if self.pp_schedule.num_microbatches > 1:
                self.num_pp_microbatches = self.pp_schedule.num_microbatches
                self.gradient_accumulation_steps = 1
            return self._pp_forward_backward_body

        return super()._select_fwd_bwd_fn()

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
