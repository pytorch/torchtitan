# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn

from torchtitan.distributed.cudagraph import cudagraph_teardown
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    trace_input_preparer_keys,
)
from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
    graph_train_step_runtime,
)
from torchtitan.experiments.graph_trainer.graph_pp.runner import GraphPipelineStepRunner
from torchtitan.experiments.graph_trainer.memory_policy import (
    validate_memory_policy_config,
)
from torchtitan.experiments.graph_trainer.registry import (
    PASS_PIPELINE_REGISTRY,
    POST_INIT_HOOKS,
    PRE_TRAIN_STEP_HOOKS,
    TRACE_CALL_INPUT_PREPARERS,
    TRACE_INPUT_PREPARERS,
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

        # PP > 1 stores GraphPipelineRuntime in self.pp_schedule. For PP=1/VPP=1,
        # GraphTrainer builds the same runtime below and keeps its step adapter
        # here. Stage graphs are built lazily and reused; close() drops the adapter.
        self._graph_pipeline_runner: GraphPipelineStepRunner | None = None
        self._validate_graph_pipeline_config()
        if (
            self.config.compile.mode == "aot_fx_trace"
            and not self.parallel_dims.pp_enabled
        ):
            # Execute the complete non-PP train step as a one-stage pipeline
            # schedule, including all gradient-accumulation microbatches.
            num_microbatches = self.gradient_accumulation_steps
            use_cuda_graph = (
                self.config.compile.enable
                and self.config.compile.enable_passes
                and not self.config.training.disable_cuda_graphs
                and "cudagraph_pass" not in self.config.compile.disable_passes
            )
            runtime = graph_train_step_runtime(
                self.model_parts[0],
                num_microbatches=num_microbatches,
                parallel_dims=self.parallel_dims,
                parallelism=self.config.parallelism,
                compile_config=self.config.compile,
                device=self.device,
                model_config=self.model_config,
                loss_fn=self.loss_fn,
                use_cuda_graph=use_cuda_graph,
            )
            sdc_config = self.config.sdc_replayer
            self._graph_pipeline_runner = GraphPipelineStepRunner(
                runtime,
                num_microbatches=num_microbatches,
                use_cuda_graph=use_cuda_graph,
                sdc_num_steps=(sdc_config.num_steps if sdc_config is not None else 0),
                sdc_num_replays=(
                    sdc_config.num_replays if sdc_config is not None else 0
                ),
            )
            # One outer trainer group is one complete runtime schedule. The
            # former gradient-accumulation groups are its microbatches.
            self.num_pp_microbatches = num_microbatches
            self.gradient_accumulation_steps = 1

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

        assert len(self.model_parts) == 1
        model = self.model_parts[0]
        microbatches = input_dict if isinstance(input_dict, list) else [input_dict]
        prepared = tuple(
            self._preprocess_fwd_bwd_inputs(model, microbatch)
            for microbatch in microbatches
        )
        assert self._graph_pipeline_runner is not None
        with self.train_context():
            return self._graph_pipeline_runner(prepared, global_valid_tokens)

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
                max_num_documents=self.dataloader.max_num_documents,
                max_context_length=self.config.training.max_context_length,
            )
            # MTP returns one labels tensor per prediction; index 0 contains
            # the complete main-model labels used for token accounting.
            self.ntokens_seen += (
                labels[0].numel() if isinstance(labels, tuple) else labels.numel()
            )
        return inputs, labels, extra_kwargs

    def _validate_graph_pipeline_config(self) -> None:
        if self.config.compile.mode != "aot_fx_trace":
            if self.config.compile.enable_deferred_fsdp_gradient_sync:
                raise ValueError(
                    "Deferred FSDP gradient synchronization requires "
                    "compile.mode='aot_fx_trace'"
                )
            return
        if self.parallel_dims.pp_enabled:
            if self.config.compile.enable_deferred_fsdp_gradient_sync:
                raise ValueError(
                    "--compile.enable_deferred_fsdp_gradient_sync currently "
                    "supports only one-stage GraphPipelineRuntime execution; "
                    "multi-stage GraphPP schedules own their reduction placement"
                )
            return
        if len(self.model_parts) != 1:
            raise ValueError(
                "One-stage GraphPipelineRuntime requires exactly one model part"
            )
        pipeline_name = self.config.compile.pass_pipeline
        if any(
            pipeline_name in registry
            for registry in (
                PASS_PIPELINE_REGISTRY,
                POST_INIT_HOOKS,
                PRE_TRAIN_STEP_HOOKS,
            )
        ):
            raise ValueError(
                "GraphPipelineRuntime does not support custom pass pipelines yet"
            )
        trace_preparer_names = set(trace_input_preparer_keys(self.config.compile))
        unsupported_preparers = trace_preparer_names.intersection(
            TRACE_INPUT_PREPARERS.keys() | TRACE_CALL_INPUT_PREPARERS.keys()
        )
        if unsupported_preparers:
            raise ValueError(
                "GraphPipelineRuntime does not support trace-input preparers yet: "
                f"{sorted(unsupported_preparers)}"
            )

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

        self._graph_pipeline_runner = None

        # See Note [explicit cudagraph teardown] in cudagraph.py
        cudagraph_teardown()
