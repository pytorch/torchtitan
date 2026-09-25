# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
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
from torchtitan.training_engine import TrainingEngine


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

        super()._initialize_forward_backward()
        _maybe_apply_numa_binding(self.device.index, self.device.type)

        if self.config.compile.memory_policy == "sac_and_offload":
            from torch._functorch._activation_offloading.offload_ops import (
                pinned_memory_pool,
            )

            self._pinned_pool_ctx = pinned_memory_pool()
            self._pinned_pool_ctx.__enter__()
        else:
            self._pinned_pool_ctx = None

    def forward_backward_microbatch(
        self,
        *,
        microbatch_group: list[TrainingMicrobatch],
        global_valid_tokens: torch.Tensor,
        accumulation_index: int = 0,
    ) -> torch.Tensor:
        if self.parallel_dims.pp_enabled or self.config.compile.mode != "aot_fx_trace":
            return super().forward_backward_microbatch(
                microbatch_group=microbatch_group,
                global_valid_tokens=global_valid_tokens,
                accumulation_index=accumulation_index,
            )

        if any(microbatch.loss_kwargs() for microbatch in microbatch_group):
            raise ValueError(
                "Per-microbatch loss arguments are not supported with GraphRuntime yet."
            )

        if accumulation_index == 0:
            self.loss_is_finite = torch.ones((), dtype=torch.int32, device=self.device)

        if self.parallel_dims.dp_replicate_enabled and (
            self.num_accumulation_steps == 1 or self.config.training.disable_cuda_graphs
        ):
            is_last = accumulation_index == self.num_accumulation_steps - 1
            for part in self.model_parts:
                part.set_requires_all_reduce(is_last)  # pyrefly: ignore[not-callable]

        def forward_backward() -> torch.Tensor:
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

            return self.forward_backward_body_fn(
                inputs=arg_mbs,
                model_kwargs=kwarg_mbs,
                labels=target_mbs,
                loss_kwargs={"global_valid_tokens": global_valid_tokens},
            )

        if self.sdc_replayer is not None and accumulation_index == 0:
            loss = self.sdc_replayer.run_fwd_bwd(
                forward_backward, step=self.num_completed_steps + 1
            )
        else:
            loss = forward_backward()
        detached_loss = loss.detach()
        self.loss_is_finite.logical_and_(torch.isfinite(detached_loss).all())
        return detached_loss

    def _non_pp_forward_backward_body(
        self,
        *,
        inputs: Any,
        labels: Any,
        model_kwargs: Any,
        loss_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Route AOT PP=1 through the runtime body used by pipeline parallelism."""
        if self.config.compile.mode == "aot_fx_trace":
            return self._pp_forward_backward_body(
                inputs=inputs,
                labels=labels,
                model_kwargs=model_kwargs,
                loss_kwargs=loss_kwargs,
            )
        return super()._non_pp_forward_backward_body(
            inputs=inputs,
            labels=labels,
            model_kwargs=model_kwargs,
            loss_kwargs=loss_kwargs,
        )

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
