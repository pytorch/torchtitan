# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch

from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.distributed.cuda_graph import cuda_graph_teardown
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
from torchtitan.training_engine import ForwardBackwardResult, TrainingEngine


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


class GraphTrainingEngine(TrainingEngine):
    """Training engine for the experimental whole-step graph backend.

    TODO: Validate this as an optional execution backend for other workflows,
    such as RL training.
    """

    def __init__(
        self,
        config: "GraphTrainer.Config",
        *,
        model_config: BaseModel.Config,
        max_num_documents: int | None,
        output_dir: str,
    ) -> None:
        validate_memory_policy_config(config.compile)
        super().__init__(
            config,
            model_config=model_config,
            max_num_documents=max_num_documents,
            output_dir=output_dir,
        )
        self._pinned_pool_ctx = None

    def _initialize_forward_backward(self) -> None:
        if self.config.parallelism.fsdp_defer_gradient_reduction:
            raise ValueError(
                "GraphTrainer does not support fsdp_defer_gradient_reduction."
            )

        if (
            self.config.compile.mode == "aot_fx_trace"
            and not self.parallel_dims.pp_enabled
        ):
            num_tokens_per_train_step = self.config.training.num_tokens_per_train_step
            if num_tokens_per_train_step < 0:
                num_microbatches = 1
            else:
                num_tokens_per_microbatch = (
                    self.config.training.num_tokens_per_microbatch_per_dp_rank
                    * self.parallel_dims.dp_replicate
                    * self.parallel_dims.dp_shard
                )
                num_microbatches = (
                    num_tokens_per_train_step // num_tokens_per_microbatch
                )
            graph_runtime = make_spmd_graph_runtime(
                self.model_parts[0],
                gradient_accumulation_steps=num_microbatches,
                parallel_dims=self.parallel_dims,
                parallelism=self.config.parallelism,
                compile_config=self.config.compile,
                device=self.device,
                model_config=self.model_config,
                loss_fn=self.loss_fn,
                trainer_config=self.config,
            )
            # The inherited PP execution path calls `pp_schedule.step`
            # GraphRuntime implements that interface and owns the underlying schedule
            # in graph_runtime.schedule.
            self.pp_schedule = graph_runtime
            stages = graph_runtime.schedule._stages
            self.pp_has_first_stage = any(stage.is_first for stage in stages)
            self.pp_has_last_stage = any(stage.is_last for stage in stages)
            assert self.pp_has_first_stage and self.pp_has_last_stage

        sdc_config = self.config.sdc_replayer
        self.sdc_replayer = None
        if sdc_config is not None:
            self.sdc_replayer = sdc_config.build(
                modules=self.model_parts,
                device=self.device,
            )

        if self.parallel_dims.pp_enabled:
            self._pp_loss_sentinel_on_non_last_stage = torch.full(
                (1,), -1.0, device=self.device
            )
        self._run_forward_backward = partial(
            self._forward_backward_body,
            defer_fsdp_gradient_reduction=False,
        )

        _maybe_apply_numa_binding(self.device.index, self.device.type)

        if self.config.compile.memory_policy == "sac_and_offload":
            from torch._functorch._activation_offloading.offload_ops import (
                pinned_memory_pool,
            )

            self._pinned_pool_ctx = pinned_memory_pool()
            self._pinned_pool_ctx.__enter__()
        else:
            self._pinned_pool_ctx = None

    def _preprocess_microbatch_groups(
        self,
        microbatch_groups: list[list[TrainingMicrobatch]],
    ) -> list[tuple[Any, ...]]:
        """Prepare GraphRuntime schedule inputs for AOT single-stage execution."""
        if self.parallel_dims.pp_enabled or self.config.compile.mode != "aot_fx_trace":
            return super()._preprocess_microbatch_groups(microbatch_groups)

        preprocessed_microbatch_groups: list[tuple[Any, ...]] = []
        for microbatch_group in microbatch_groups:
            if any(microbatch.loss_kwargs() for microbatch in microbatch_group):
                raise ValueError(
                    "Per-microbatch loss arguments are not supported with "
                    "GraphRuntime yet."
                )

            # Calling convention:
            # The runtime receives one positional tuple, keyword dictionary,
            # and target per schedule microbatch.
            arg_mbs: list[tuple[torch.Tensor, ...]] = []
            kwarg_mbs: list[dict[str, Any]] = []
            target_mbs: list[torch.Tensor] = []
            for microbatch in microbatch_group:
                input_dict = microbatch.to_input_dict(self.device, non_blocking=True)
                with sl.log_trace_span("preprocess_inputs"):
                    inputs_mb, labels_mb, extra_kwargs_mb = self.model_parts[
                        0
                    ].preprocess_inputs(
                        input_dict,
                        parallel_dims=self.parallel_dims,
                        parallelism=self.config.parallelism,
                        max_num_documents=self.max_num_documents,
                        max_context_length=self.config.training.max_context_length,
                        **self.preprocess_inputs_kwargs,
                    )
                    assert isinstance(inputs_mb, torch.Tensor)
                    assert isinstance(labels_mb, torch.Tensor)
                    self.ntokens_seen += (
                        self.config.training.num_tokens_per_microbatch_per_dp_rank
                        // self.parallel_dims.cp
                    )
                arg_mbs.append((inputs_mb,))
                kwarg_mbs.append(extra_kwargs_mb)
                target_mbs.append(labels_mb)
            preprocessed_microbatch_groups.append((arg_mbs, kwarg_mbs, target_mbs))

        return preprocessed_microbatch_groups

    def _forward_backward_body(
        self,
        microbatch_groups: list[tuple[Any, ...]],
        global_valid_tokens: torch.Tensor,
        *,
        defer_fsdp_gradient_reduction: bool,
    ) -> ForwardBackwardResult:
        """Run AOT single-stage groups through GraphRuntime."""
        if self.parallel_dims.pp_enabled or self.config.compile.mode != "aot_fx_trace":
            return super()._forward_backward_body(
                microbatch_groups,
                global_valid_tokens,
                defer_fsdp_gradient_reduction=defer_fsdp_gradient_reduction,
            )

        assert not defer_fsdp_gradient_reduction
        accumulated_loss: torch.Tensor | None = None
        loss_metrics: list[dict[str, torch.Tensor]] = []
        for inputs, model_kwargs, labels in microbatch_groups:
            self.loss_metrics = {}
            loss = self._pp_forward_backward_microbatch_group(
                inputs=inputs,
                model_kwargs=model_kwargs,
                labels=labels,
                loss_kwargs={"global_valid_tokens": global_valid_tokens},
                finalize_gradients=True,
            )
            detached_loss = loss.detach()
            if accumulated_loss is None:
                accumulated_loss = detached_loss.clone()
            else:
                accumulated_loss.add_(detached_loss)
            loss_metrics.append(
                {
                    key: value.detach().clone()
                    for key, value in self.loss_metrics.items()
                }
            )

        assert accumulated_loss is not None
        return ForwardBackwardResult(accumulated_loss, loss_metrics)

    def close(self) -> None:
        if self._pinned_pool_ctx is not None:
            self._pinned_pool_ctx.__exit__(None, None, None)
            self._pinned_pool_ctx = None

        super().close()

        cuda_graph_teardown()


class GraphTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    engine_cls = GraphTrainingEngine
    engine: GraphTrainingEngine

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if (
            self.config.compile.mode == "aot_fx_trace"
            and not self.engine.parallel_dims.pp_enabled
            and self.engine.pp_schedule.num_microbatches > 1
        ):
            self.num_pp_microbatches = self.engine.pp_schedule.num_microbatches
            self.gradient_accumulation_steps = 1
        POST_INIT_HOOKS.get(self.config.compile.pass_pipeline, lambda _: None)(self)

    def train_step(self, data_iterator: Iterator[TrainingMicrobatch]) -> None:
        PRE_TRAIN_STEP_HOOKS.get(self.config.compile.pass_pipeline, lambda _: None)(
            self
        )
        super().train_step(data_iterator)
